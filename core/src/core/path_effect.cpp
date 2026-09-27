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

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

PathEffect::PathEffect(const AudioSettings& audioSettings,
                       const PathEffectSettings& effectSettings)
    : mSamplingRate(audioSettings.samplingRate)
    , mFrameSize(audioSettings.frameSize)
    , mSpatialize(effectSettings.spatialize)
    , mPrevBinaural(false)
    , mBandSignals(Bands::kNumBands, audioSettings.frameSize)
    , mPrevGains(Bands::kNumBands, kMaxChannels)
{
    IIR filters[Bands::kNumBands];
    IIR::bandFilters(filters, audioSettings.samplingRate);

    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        mBandFilters[i].setFilter(filters[i]);
    }

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

        mCrossover.resize(Bands::kNumBands, mFieldFFT->numComplexSamples);
        mBandHRTF.resize(Bands::kNumBands, mFieldFFT->numComplexSamples);
        mFieldHRTF.resize(mFieldFFT->numComplexSamples);
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

    if (mSpatialize && params.binaural)
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
    float weights[AmbisonicField::kMaxOrder + 1];

    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        auto order = std::min(std::max(static_cast<int>(field.orders[band]), 0), static_cast<int>(AmbisonicField::kMaxOrder));
        mOrders[band] = order;

        if (rotation)
        {
            rotation->apply(order, field.coeffs[band], mCoeffs[band]);
        }
        else
        {
            memcpy(mCoeffs[band], field.coeffs[band], SphericalHarmonics::numCoeffsForOrder(order) * sizeof(float));
        }

        if (!maxRE)
            continue;

        SphericalHarmonics::maxREWeights(order, weights);

        for (auto l = 0, i = 0; l <= order; ++l)
        {
            for (auto m = -l; m <= l; ++m, ++i)
            {
                mCoeffs[band][i] *= weights[l];
            }
        }
    }
}

// Raised-cosine crossovers, one octave wide in log frequency, centered on the phonon band edges. Complementary weights
// keep a field that is identical across bands exactly equal to the single-band rendering.
void PathEffect::buildCrossover()
{
    auto numBins = mFieldFFT->numComplexSamples;
    auto binWidth = static_cast<float>(mSamplingRate) / mFieldFFT->numRealSamples;

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

            mCrossover[band][bin] = pass - lowerPass;
            lowerPass = pass;
        }
    }

    for (auto band = 0; band < Bands::kNumBands; ++band)
    {
        auto first = 0;
        while (first < numBins && mCrossover[band][first] <= 0.0f)
            ++first;

        auto end = numBins;
        while (end > first && mCrossover[band][end - 1] <= 0.0f)
            --end;

        mSupport[band][0] = first;
        mSupport[band][1] = end;
    }
}

AudioEffectState PathEffect::applyBinaural(const HRTFDatabase& hrtf,
                                           const AudioBuffer& in,
                                           AudioBuffer& out)
{
    auto numBins = mFieldFFT->numComplexSamples;

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        mBandHRTF.zero();

        for (auto band = 0; band < Bands::kNumBands; ++band)
        {
            auto first = mSupport[band][0];
            auto numSupported = mSupport[band][1] - first;
            auto order = mOrders[band];
            auto bandHRTF = reinterpret_cast<float*>(&mBandHRTF[band][first]);

            for (auto i = 0; i < SphericalHarmonics::numCoeffsForOrder(order); ++i)
            {
                if (mCoeffs[band][i] == 0.0f)
                    continue;

                ArrayMath::scaleAccumulate(2 * numSupported, reinterpret_cast<const float*>(hrtf.fieldHRTF(ear, order, i) + first),
                                           mCoeffs[band][i], bandHRTF);
            }
        }

        for (auto bin = 0; bin < numBins; ++bin)
        {
            complex_t sum = 0.0f;

            for (auto band = 0; band < Bands::kNumBands; ++band)
            {
                sum += mCrossover[band][bin] * mBandHRTF[band][bin];
            }

            mFieldHRTF[bin] = sum;
        }

        mHRIR.zero();
        mFieldFFT->applyInverse(mFieldHRTF.data(), mHRIR.data());
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

}
