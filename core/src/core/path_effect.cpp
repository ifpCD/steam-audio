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
    , mConvolutionIRSize(0)
    , mGroupDelayFirstBin(0)
    , mGroupDelayLinearBin(0)
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

        // the fitted HRIR sits in [1/4, 5/8) of the field IR (fieldDelay() minus its lead-in, plus the HRIR length).
        // crossover ringing and the restored group delay stay within 1/8 of it, and past 3/4 there is only crossover
        // pre-ringing wrapped around from before t = 0 (below -60 dB). convolving 3/4 keeps it out of the output and
        // drops the convolution FFT a size for frames like 512 (640 + 383 -> 1024 instead of 640 + 511 -> 2048).
        mConvolutionIRSize = 3 * mFieldFFT->numRealSamples / 4;

        OverlapAddConvolutionEffectSettings convolutionSettings{};
        convolutionSettings.numChannels = IHRTFMap::kNumEars;
        convolutionSettings.irSize = mConvolutionIRSize;

        mConvolution = ipl::make_unique<OverlapAddConvolutionEffect>(audioSettings, convolutionSettings);
        mConvolutionFFT = ipl::make_unique<FFT>(mConvolution->wetAudioSize());

        mNumFieldBlocks = effectSettings.hrtf->numFieldBlocks();
        const auto numBlockFloats = mNumFieldBlocks * HRTFDatabase::kFieldEarFloats;

        mCrossover.resize(Bands::kNumBands, numBlockFloats);
        mSharedWeights.resize(numBlockFloats);
        mFieldHRTF.resize(IHRTFMap::kNumEars, mNumFieldBlocks * HRTFDatabase::kFieldBlockBins);
        mHRIR.resize(mConvolutionFFT->numRealSamples);
        mHRTF.resize(IHRTFMap::kNumEars, mConvolutionFFT->numComplexSamples);
        mGroupDelayPhase.resize(mFieldFFT->numComplexSamples);

        buildCrossover();
        buildGroupDelay();
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
        // no max-rE: the MagLS tables are fitted for the unweighted field
        loadField(*params.field, &listenerRotation(*params.listener), false);
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

// Group delay d (samples) means phase -d * sum_{k' <= k} w(k') 2pi / N at bin k: continuous in frequency, with group
// delay d * w(k) ramping from 0 at the MagLS cutoff to d half an octave above it.
void PathEffect::buildGroupDelay()
{
    auto numBins = mFieldFFT->numComplexSamples;
    auto binWidth = static_cast<float>(mSamplingRate) / mFieldFFT->numRealSamples;
    auto step = 2.0f * Math::kPi / mFieldFFT->numRealSamples;
    auto start = HRTFDatabase::kFieldMagnitudeCutoffMax;

    auto phase = 0.0f;
    mGroupDelayFirstBin = numBins;
    mGroupDelayLinearBin = numBins;

    for (auto bin = 0; bin < numBins; ++bin)
    {
        auto frequency = bin * binWidth;
        auto weight = 0.0f;

        if (frequency > start)
        {
            auto position = std::min(log2f(frequency / start) / kGroupDelayRampOctaves, 1.0f);
            weight = (position >= 1.0f) ? 1.0f : 0.5f - 0.5f * cosf(Math::kPi * position);
        }

        if (weight > 0.0f && mGroupDelayFirstBin == numBins)
            mGroupDelayFirstBin = bin;

        if (weight >= 1.0f && mGroupDelayLinearBin == numBins)
            mGroupDelayLinearBin = bin;

        phase += step * weight;
        mGroupDelayPhase[bin] = phase;
    }

    // per-band fields: each band speaks for the share of [cutoff, kGroupDelayBandHigh) its crossover passes
    auto numWeighted = 0;
    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        mGroupDelayBandWeights[band] = 0.0f;
    }

    for (auto bin = 0; bin < numBins; ++bin)
    {
        auto frequency = bin * binWidth;
        if (frequency < start || frequency >= kGroupDelayBandHigh)
            continue;

        for (auto band = 0; band < Bands::kNumBands; ++band)
        {
            mGroupDelayBandWeights[band] += mCrossover[band][2 * bin];
        }

        ++numWeighted;
    }

    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        mGroupDelayBandWeights[band] /= std::max(numWeighted, 1);
    }
}

void PathEffect::continueGroupDelay(const HRTFDatabase& hrtf)
{
    const auto kSqrtFourPi = 3.5449077f;

    // arrival time per ear = integral(g tau) / integral(g) = (c . T) / (sqrt(4pi) c_00), over the bands that carry
    // the frequencies the group delay is restored at
    float numerator[IHRTFMap::kNumEars] = { 0.0f, 0.0f };
    auto omni = 0.0f;
    auto norm = 0.0f;

    auto accumulate = [&](const float* coeffs, int order, float weight)
    {
        auto numCoeffs = std::min(SphericalHarmonics::numCoeffsForOrder(order), HRTFDatabase::kFieldDelayCoeffs);

        for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
        {
            const auto* delays = hrtf.fieldDelayCoeffs(ear);
            auto dot = 0.0f;

            for (auto i = 0; i < numCoeffs; ++i)
            {
                dot += coeffs[i] * delays[i];
            }

            numerator[ear] += weight * dot;
        }

        auto sumSquares = 0.0f;
        for (auto i = 0; i < numCoeffs; ++i)
        {
            sumSquares += coeffs[i] * coeffs[i];
        }

        omni += weight * coeffs[0];
        norm += weight * sqrtf(sumSquares);
    };

    if (mSharedShape)
    {
        accumulate(mShape, mOrders[0], 1.0f);
    }
    else
    {
        for (auto band = 0; band < Bands::kNumBands; ++band)
        {
            if (mGroupDelayBandWeights[band] > 0.0f)
            {
                accumulate(mCoeffs[band], mOrders[band], mGroupDelayBandWeights[band]);
            }
        }
    }

    // a point source has c_00 / |c| = 1 / (order + 1) <= 0.2 over the first 25 coefficients, a field with any extent
    // more. one whose omni term has (nearly) cancelled has no meaningful mean arrival: leave the shared delay.
    if (!(omni > 0.05f * norm))
        return;

    auto numBins = mFieldFFT->numComplexSamples;
    auto step = 2.0f * Math::kPi / mFieldFFT->numRealSamples;

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        auto delay = numerator[ear] / (kSqrtFourPi * omni);
        delay = std::min(std::max(delay, hrtf.fieldDelayMin(ear)), hrtf.fieldDelayMax(ear));

        if (fabsf(delay) < 1e-3f)
            continue;

        auto* bins = reinterpret_cast<float*>(mFieldHRTF[ear]);

        for (auto bin = mGroupDelayFirstBin; bin < mGroupDelayLinearBin; ++bin)
        {
            auto angle = -delay * mGroupDelayPhase[bin];
            auto c = cosf(angle);
            auto s = sinf(angle);
            auto re = bins[2 * bin];
            auto im = bins[2 * bin + 1];
            bins[2 * bin] = re * c - im * s;
            bins[2 * bin + 1] = re * s + im * c;
        }

        if (mGroupDelayLinearBin >= numBins)
            continue;

        // past the ramp the phase is linear in bin: rotate by running phasors, four independent chains (bins k + j,
        // j < 4, each stepping 4 bins) so the recurrence pipelines instead of serializing on multiply latency.
        // renormalized every 64 bins.
        const auto kLanes = 4;
        float zr[kLanes];
        float zi[kLanes];

        for (auto j = 0; j < kLanes; ++j)
        {
            auto angle = -delay * (mGroupDelayPhase[mGroupDelayLinearBin] + j * step);
            zr[j] = cosf(angle);
            zi[j] = sinf(angle);
        }

        auto dr = cosf(-delay * step * kLanes);
        auto di = sinf(-delay * step * kLanes);
        auto bin = mGroupDelayLinearBin;

        for (auto count = 0; bin + kLanes <= numBins; bin += kLanes, ++count)
        {
            for (auto j = 0; j < kLanes; ++j)
            {
                auto re = bins[2 * (bin + j)];
                auto im = bins[2 * (bin + j) + 1];
                bins[2 * (bin + j)] = re * zr[j] - im * zi[j];
                bins[2 * (bin + j) + 1] = re * zi[j] + im * zr[j];

                auto nr = zr[j] * dr - zi[j] * di;
                zi[j] = zr[j] * di + zi[j] * dr;
                zr[j] = nr;
            }

            if ((count & 15) == 15)
            {
                for (auto j = 0; j < kLanes; ++j)
                {
                    auto scale = 1.0f / sqrtf(zr[j] * zr[j] + zi[j] * zi[j]);
                    zr[j] *= scale;
                    zi[j] *= scale;
                }
            }
        }

        for (auto j = 0; bin < numBins; ++bin, ++j)
        {
            auto re = bins[2 * bin];
            auto im = bins[2 * bin + 1];
            bins[2 * bin] = re * zr[j] - im * zi[j];
            bins[2 * bin + 1] = re * zi[j] + im * zr[j];
        }
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

    continueGroupDelay(hrtf);

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        mFieldFFT->applyInverse(mFieldHRTF[ear], mHRIR.data());
        memset(mHRIR.data() + mConvolutionIRSize, 0, (mFieldFFT->numRealSamples - mConvolutionIRSize) * sizeof(float));
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
