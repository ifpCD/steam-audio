//
// Copyright 2017-2023 Valve Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "path_effect.h"

#include "ambisonics_panning_effect.h"
#include "array_math.h"
#include "context.h"
#include "float4.h"

namespace ipl {

static int clampedOrder(int32_t order)
{
    return std::min(std::max(static_cast<int>(order), 0), static_cast<int>(AmbisonicField::kMaxOrder));
}

// rotate (if rotation) + maxRE weight (if maxRE) one band
static void prepareCoeffs(int order,
                          const float* coeffs,
                          const SHRotation* rotation,
                          bool maxRE,
                          float* prepared)
{
    if (rotation)
    {
        rotation->apply(order, coeffs, prepared);
    }
    else
    {
        memcpy(prepared, coeffs, SphericalHarmonics::numCoeffsForOrder(order) * sizeof(float));
    }

    if (!maxRE)
        return;

    float weights[AmbisonicField::kMaxOrder + 1];
    SphericalHarmonics::maxREWeights(order, weights);

    for (auto l = 0, i = 0; l <= order; ++l)
    {
        for (auto m = -l; m <= l; ++m, ++i)
        {
            prepared[i] *= weights[l];
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

PathEffect::PathEffect(const AudioSettings& audioSettings,
                       const PathEffectSettings& effectSettings)
    : mSamplingRate(audioSettings.samplingRate)
    , mFrameSize(audioSettings.frameSize)
    , mSpatialize(effectSettings.spatialize)
    , mPrevBinaural(false)
    , mSharedShape(false)
    , mNumFieldBlocks(0)
    , mAccumulateFieldHRTF(&PathEffect::accumulateFieldHRTF_float4)
    , mBandSignals(Bands::kNumBands, audioSettings.frameSize)
    , mPrevGains(Bands::kNumBands, kMaxChannels)
{
    IIR filters[Bands::kNumBands];
    IIR::bandFilters(filters, audioSettings.samplingRate);

    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        mBandFilters[i].setFilter(filters[i]);
    }

#if defined(IPL_ENABLE_FLOAT8)
    if (gSIMDLevel() >= SIMDLevel::AVX2)
    {
        mAccumulateFieldHRTF = &PathEffect::accumulateFieldHRTF_float8;
    }
#endif

    if (mSpatialize)
    {
        AmbisonicsPanningEffect::buildDecoder(*effectSettings.speakerLayout, AmbisonicField::kMaxOrder, mDecoder);
    }

    if (mSpatialize && effectSettings.hrtf)
    {
        mFieldFFT = ipl::make_unique<FFT>(effectSettings.hrtf->fieldIRSize());

        OverlapAddConvolutionEffectSettings convolutionSettings{};
        convolutionSettings.numChannels = IHRTFMap::kNumEars;
        convolutionSettings.irSize = mFieldFFT->numRealSamples;

        mConvolution = ipl::make_unique<OverlapAddConvolutionEffect>(audioSettings, convolutionSettings);
        mConvolutionFFT = ipl::make_unique<FFT>(mConvolution->wetAudioSize());

        mNumFieldBlocks = effectSettings.hrtf->numFieldBlocks();
        const auto numBlockFloats = mNumFieldBlocks * HRTFDatabase::kFieldEarFloats;

        mCrossover.resize(Bands::kNumBands, numBlockFloats);
        mSharedWeights.resize(numBlockFloats);
        mFieldHRTF.resize(IHRTFMap::kNumEars, mNumFieldBlocks * HRTFDatabase::kFieldBlockBins);
        mHRIR.resize(mConvolutionFFT->numRealSamples);
        mHRTF.resize(IHRTFMap::kNumEars, mConvolutionFFT->numComplexSamples);

        buildCrossover();
    }

    reset();
}

void PathEffect::reset()
{
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        mBandFilters[i].reset();
    }

    mPrevGains.zero();

    if (mConvolution)
    {
        mConvolution->reset();
    }

    mPrevBinaural = false;
}

AudioEffectState PathEffect::apply(const PathEffectParams& params,
                                   const AudioBuffer& in,
                                   AudioBuffer& out)
{
    assert(in.numSamples() == out.numSamples());
    assert(in.numChannels() == 1);

    out.makeSilent();

    if (!params.field)
        return AudioEffectState::TailComplete;

    // an effect created without an HRTF (or handed one with another field FFT size) has nothing to render into.
    // decode to speakers for that frame.
    auto binaural = mSpatialize && params.binaural && params.hrtf && mConvolution &&
                    params.hrtf->numFieldBlocks() == mNumFieldBlocks;

    if (binaural)
    {
        loadField(*params.field, &listenerRotation(*params.listener), true);
        mPrevBinaural = true;
        return applyBinaural(*params.hrtf, in, out);
    }

    loadField(*params.field, mSpatialize ? &listenerRotation(*params.listener) : nullptr, mSpatialize);
    applyGains(mSpatialize, in, out);
    mPrevBinaural = false;
    return AudioEffectState::TailComplete;
}

AudioEffectState PathEffect::tail(AudioBuffer& out)
{
    if (mPrevBinaural)
        return mConvolution->tail(out);

    out.makeSilent();
    return AudioEffectState::TailComplete;
}

int PathEffect::numTailSamplesRemaining() const
{
    return mPrevBinaural ? mConvolution->numTailSamplesRemaining() : 0;
}

// Every path effect on an audio thread renders against the same listener, so the order-10 rotation is built once.
const SHRotation& PathEffect::listenerRotation(const CoordinateSpace3f& listener)
{
    // TODO: first use allocs + builds the uvw table on the audio thread (one-off spike). prewarm before shipping.
    static thread_local SHRotation rotation(AmbisonicField::kMaxOrder);
    static thread_local Vector3f ahead(0.0f, 0.0f, 0.0f);
    static thread_local Vector3f up(0.0f, 0.0f, 0.0f);

    if (memcmp(&ahead, &listener.ahead, sizeof(Vector3f)) != 0 || memcmp(&up, &listener.up, sizeof(Vector3f)) != 0)
    {
        rotation.setRotation(listener);
        ahead = listener.ahead;
        up = listener.up;
    }

    return rotation;
}

void PathEffect::loadField(const AmbisonicField& field,
                           const SHRotation* rotation,
                           bool maxRE)
{
    mSharedShape = field.sharedShape && 0 <= field.shapeBand && field.shapeBand < Bands::kNumBands;

    if (mSharedShape)
    {
        // rotation + maxRE are linear -> prep the shape once, scale per band
        auto order = clampedOrder(field.orders[field.shapeBand]);
        prepareCoeffs(order, field.coeffs[field.shapeBand], rotation, maxRE, mShape);

        for (auto band = 0; band < Bands::kNumBands; ++band)
        {
            mOrders[band] = order;
            mBandGains[band] = field.bandGains[band];
            ArrayMath::scale(SphericalHarmonics::numCoeffsForOrder(order), mShape, mBandGains[band], mCoeffs[band]);
        }

        return;
    }

    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        mOrders[band] = clampedOrder(field.orders[band]);
        prepareCoeffs(mOrders[band], field.coeffs[band], rotation, maxRE, mCoeffs[band]);
    }
}

// Raised-cosine crossovers, one octave wide in log frequency, centered on the phonon band edges. Complementary weights
// keep a field that is identical across bands exactly equal to the single-band rendering.
void PathEffect::buildCrossover()
{
    auto numBins = mFieldFFT->numComplexSamples;
    auto binWidth = static_cast<float>(mSamplingRate) / mFieldFFT->numRealSamples;

    // padding bins in the last block keep weight 0
    mCrossover.zero();

    for (auto bin = 0; bin < numBins; ++bin)
    {
        auto frequency = bin * binWidth;
        auto lowerPass = 0.0f;

        for (auto band = 0; band < Bands::kNumBands; ++band)
        {
            auto pass = 1.0f;

            if (band < Bands::kNumBands - 1 && bin > 0)
            {
                auto position = log2f(frequency / Bands::kHighCutoffFrequencies[band]) / kCrossoverOctaves;
                auto transition = std::min(std::max(position + 0.5f, 0.0f), 1.0f);
                pass = 0.5f + 0.5f * cosf(Math::kPi * transition);
            }

            mCrossover[band][2 * bin] = pass - lowerPass;
            mCrossover[band][2 * bin + 1] = pass - lowerPass;
            lowerPass = pass;
        }
    }

    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        auto first = 0;
        while (first < numBins && mCrossover[band][2 * first] <= 0.0f)
            ++first;

        auto end = numBins;
        while (end > first && mCrossover[band][2 * (end - 1)] <= 0.0f)
            --end;

        // whole blocks, extra bins have weight 0
        mBlockRange[band][0] = first / HRTFDatabase::kFieldBlockBins;
        mBlockRange[band][1] = (end + HRTFDatabase::kFieldBlockBins - 1) / HRTFDatabase::kFieldBlockBins;
    }
}

AudioEffectState PathEffect::applyBinaural(const HRTFDatabase& hrtf,
                                           const AudioBuffer& in,
                                           AudioBuffer& out)
{
    assert(hrtf.numFieldBlocks() == mNumFieldBlocks);

    auto* left = reinterpret_cast<float*>(mFieldHRTF[0]);
    auto* right = reinterpret_cast<float*>(mFieldHRTF[1]);

    mFieldHRTF.zero();

    if (mSharedShape)
    {
        // one synthesis pass, crossover folds into a per-bin gain curve
        auto numFloats = static_cast<int>(mSharedWeights.size(0));

        ArrayMath::scale(numFloats, mCrossover[0], mBandGains[0], mSharedWeights.data());
        for (auto band = 1; band < Bands::kNumBands; ++band)
        {
            ArrayMath::scaleAccumulate(numFloats, mCrossover[band], mBandGains[band], mSharedWeights.data());
        }

        mAccumulateFieldHRTF(hrtf.fieldHRTFBlocks(mOrders[0]), SphericalHarmonics::numCoeffsForOrder(mOrders[0]), mShape,
                             0, mNumFieldBlocks, mSharedWeights.data(), left, right);
    }
    else
    {
        for (auto band = 0; band < Bands::kNumBands; ++band)
        {
            mAccumulateFieldHRTF(hrtf.fieldHRTFBlocks(mOrders[band]), SphericalHarmonics::numCoeffsForOrder(mOrders[band]),
                                 mCoeffs[band], mBlockRange[band][0], mBlockRange[band][1], mCrossover[band], left, right);
        }
    }

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        mFieldFFT->applyInverse(mFieldHRTF[ear], mHRIR.data());
        mConvolutionFFT->applyForward(mHRIR.data(), mHRTF[ear]);
    }

    OverlapAddConvolutionEffectParams convolutionParams{};
    convolutionParams.fftIR = mHRTF.data();

    return mConvolution->apply(convolutionParams, in, out);
}

void PathEffect::applyGains(bool decode,
                            const AudioBuffer& in,
                            AudioBuffer& out)
{
    auto numChannels = out.numChannels();
    auto rampStep = 1.0f / mFrameSize;

    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        mBandFilters[band].apply(mFrameSize, in[0], mBandSignals[band]);

        auto numCoeffs = SphericalHarmonics::numCoeffsForOrder(mOrders[band]);
        auto prevGains = mPrevGains[band];
        auto signal = mBandSignals[band];

        for (auto channel = 0; channel < numChannels; ++channel)
        {
            auto target = 0.0f;

            if (decode)
            {
                for (auto i = 0; i < numCoeffs; ++i)
                {
                    target += mDecoder(channel, i) * mCoeffs[band][i];
                }
            }
            else if (channel < numCoeffs)
            {
                target = mCoeffs[band][channel];
            }

            auto gain = prevGains[channel];
            auto gainStep = (target - gain) * rampStep;

            for (auto k = 0; k < mFrameSize; ++k, gain += gainStep)
            {
                out[channel][k] += gain * signal[k];
            }

            prevGains[channel] = target;
        }
    }
}

// 16 bins x 2 ears won't fit in 16 xmm without spilling -> two 8-bin halves per block. table still read once.
void PathEffect::accumulateFieldHRTF_float4(const float* blocks,
                                            int numCoeffs,
                                            const float* coeffs,
                                            int firstBlock,
                                            int endBlock,
                                            const float* weights,
                                            float* left,
                                            float* right)
{
    const auto rowFloats = HRTFDatabase::kFieldRowFloats;
    const auto earFloats = HRTFDatabase::kFieldEarFloats;
    const auto halfFloats = earFloats / 2;

    for (auto block = firstBlock; block < endBlock; ++block)
    {
        const auto* rows = &blocks[static_cast<size_t>(block) * numCoeffs * rowFloats];

        for (auto half = 0; half < 2; ++half)
        {
            const auto* row = rows + half * halfFloats;

            auto l0 = float4::zero();
            auto l1 = float4::zero();
            auto l2 = float4::zero();
            auto l3 = float4::zero();
            auto r0 = float4::zero();
            auto r1 = float4::zero();
            auto r2 = float4::zero();
            auto r3 = float4::zero();

            for (auto i = 0; i < numCoeffs; ++i, row += rowFloats)
            {
                const auto c = float4::set1(coeffs[i]);

                l0 = float4::add(l0, float4::mul(float4::load(row + 0), c));
                l1 = float4::add(l1, float4::mul(float4::load(row + 4), c));
                l2 = float4::add(l2, float4::mul(float4::load(row + 8), c));
                l3 = float4::add(l3, float4::mul(float4::load(row + 12), c));
                r0 = float4::add(r0, float4::mul(float4::load(row + earFloats + 0), c));
                r1 = float4::add(r1, float4::mul(float4::load(row + earFloats + 4), c));
                r2 = float4::add(r2, float4::mul(float4::load(row + earFloats + 8), c));
                r3 = float4::add(r3, float4::mul(float4::load(row + earFloats + 12), c));
            }

            const auto offset = block * earFloats + half * halfFloats;
            const auto* w = weights + offset;
            auto* outLeft = left + offset;
            auto* outRight = right + offset;

            float4::store(outLeft + 0, float4::add(float4::load(outLeft + 0), float4::mul(l0, float4::load(w + 0))));
            float4::store(outLeft + 4, float4::add(float4::load(outLeft + 4), float4::mul(l1, float4::load(w + 4))));
            float4::store(outLeft + 8, float4::add(float4::load(outLeft + 8), float4::mul(l2, float4::load(w + 8))));
            float4::store(outLeft + 12, float4::add(float4::load(outLeft + 12), float4::mul(l3, float4::load(w + 12))));
            float4::store(outRight + 0, float4::add(float4::load(outRight + 0), float4::mul(r0, float4::load(w + 0))));
            float4::store(outRight + 4, float4::add(float4::load(outRight + 4), float4::mul(r1, float4::load(w + 4))));
            float4::store(outRight + 8, float4::add(float4::load(outRight + 8), float4::mul(r2, float4::load(w + 8))));
            float4::store(outRight + 12, float4::add(float4::load(outRight + 12), float4::mul(r3, float4::load(w + 12))));
        }
    }
}

}
