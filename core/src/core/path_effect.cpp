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
#include "array_math.h"
#include "sh.h"
#include "bands.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

PathEffect::PathEffect(const AudioSettings& audioSettings,
    const PathEffectSettings& effectSettings)
    : mMaxOrder(effectSettings.maxOrder)
    , mSpatialize(effectSettings.spatialize)
    , mPrevBinaural(false)
{
    // initialize crossover filters
    IIR filters[Bands::kNumBands];
    IIR::bandFilters(filters, audioSettings.samplingRate);

    for (int i = 0; i < Bands::kNumBands; ++i)
    {
        mCrossover[i].setFilter(filters[i]);
        mBandBuffers[i] = make_unique<AudioBuffer>(1, audioSettings.frameSize);
    }

    // scratch buffer for the max possible channels
    int maxOutChannels = std::max(2, SphericalHarmonics::numCoeffsForOrder(effectSettings.maxOrder));
    if (effectSettings.speakerLayout) {
        maxOutChannels = std::max(maxOutChannels, effectSettings.speakerLayout->numSpeakers);
    }
    mTempBuffer = make_unique<AudioBuffer>(maxOutChannels, audioSettings.frameSize);

    if (mSpatialize)
    {
        AmbisonicsRotateEffectSettings ambisonicsRotateSettings{};
        ambisonicsRotateSettings.maxOrder = effectSettings.maxOrder;

        mAmbisonicsRotateEffect = make_unique<AmbisonicsRotateEffect>(AudioSettings{audioSettings.samplingRate, 1}, ambisonicsRotateSettings);

        AmbisonicsPanningEffectSettings ambisonicsPanningSettings{};
        ambisonicsPanningSettings.speakerLayout = effectSettings.speakerLayout;
        ambisonicsPanningSettings.maxOrder = effectSettings.maxOrder;

        mAmbisonicsPanningEffect = make_unique<AmbisonicsPanningEffect>(audioSettings, ambisonicsPanningSettings);

        OverlapAddConvolutionEffectSettings overlapAddSettings{};
        overlapAddSettings.numChannels = IHRTFMap::kNumEars;
        overlapAddSettings.irSize = effectSettings.hrtf->numSamples();

        mAmbisonicsBuffer = make_unique<AudioBuffer>(SphericalHarmonics::numCoeffsForOrder(effectSettings.maxOrder), 1);
        mSpeakerBuffer = make_unique<AudioBuffer>(effectSettings.speakerLayout->numSpeakers, 1);

        for (int i = 0; i < Bands::kNumBands; ++i)
        {
            mOverlapAddEffects[i] = make_unique<OverlapAddConvolutionEffect>(audioSettings, overlapAddSettings);
            mHRTF[i].resize(2, effectSettings.hrtf->numSpectrumSamples());

            for (int j = 0; j < effectSettings.speakerLayout->numSpeakers; ++j)
            {
                mGainEffectsPan[i].push_back(make_unique<GainEffect>(audioSettings));
            }
        }
    }
    else
    {
        int numCoeffs = SphericalHarmonics::numCoeffsForOrder(effectSettings.maxOrder);
        for (int i = 0; i < Bands::kNumBands; ++i)
        {
            for (int j = 0; j < numCoeffs; ++j)
            {
                mGainEffectsRaw[i].push_back(make_unique<GainEffect>(audioSettings));
            }
        }
    }
}

void PathEffect::reset()
{
    for (int i = 0; i < Bands::kNumBands; ++i)
    {
        mCrossover[i].reset();
        
        if (mSpatialize)
        {
            mOverlapAddEffects[i]->reset();
            for (auto& gain : mGainEffectsPan[i]) gain->reset();
        }
        else
        {
            for (auto& gain : mGainEffectsRaw[i]) gain->reset();
        }
    }

    if (mSpatialize)
    {
        mAmbisonicsRotateEffect->reset();
        mAmbisonicsPanningEffect->reset();
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

    // process crossover into 3 frequency bands
    for (int i = 0; i < Bands::kNumBands; ++i)
    {
        mCrossover[i].apply(in.numSamples(), in[0], (*mBandBuffers[i])[0]);
    }

    int numCoeffs = SphericalHarmonics::numCoeffsForOrder(params.order);
    AudioEffectState state = AudioEffectState::TailComplete;

    // stack-allocated wrapper around the temp buffer so AudioBuffer::mix won't assert.
    float* channelPointers[32]; // accommodate max out channels
    for (int i = 0; i < out.numChannels(); ++i) channelPointers[i] = (*mTempBuffer)[i];
    AudioBuffer subTempBuffer(out.numChannels(), out.numSamples(), channelPointers);

    if (mSpatialize)
    {
        for (int band = 0; band < Bands::kNumBands; ++band)
        {
            const float* bandSH = &params.shCoeffs[band * numCoeffs];

            // load SH coeffs for this band
            for (int i = 0; i < numCoeffs; ++i)
                (*mAmbisonicsBuffer)[i][0] = bandSH[i];

            // rotate SH coefficients for player orientation
            AmbisonicsRotateEffectParams rotateParams{};
            rotateParams.orientation = params.listener;
            rotateParams.order = params.order;
            mAmbisonicsRotateEffect->apply(rotateParams, *mAmbisonicsBuffer, *mAmbisonicsBuffer);

            subTempBuffer.makeSilent();

            if (params.binaural)
            {
                // collapse 16-channel HRTF into 2-channel binaural HRIR using SH
                memset(mHRTF[band].flatData(), 0, mHRTF[band].totalSize() * sizeof(complex_t));
                auto cosine = cosf((137.9f * Math::kDegreesToRadians) / (params.order + 1.51f));

                for (auto l = 0, i = 0; l <= params.order; ++l)
                {
                    auto scalar = SphericalHarmonics::legendre(l, cosine);
                    for (auto m = -l; m <= l; ++m, ++i)
                    {
                        const complex_t* hrtfForChannel[2] = {nullptr, nullptr};
                        params.hrtf->ambisonicsHRTF(i, hrtfForChannel);

                        for (auto k = 0; k < IHRTFMap::kNumEars; ++k)
                        {
                            ArrayMath::scaleAccumulate(params.hrtf->numSpectrumSamples(),
                                reinterpret_cast<const float*>(hrtfForChannel[k]),
                                scalar * (*mAmbisonicsBuffer)[i][0],
                                reinterpret_cast<float*>(mHRTF[band][k]));
                        }
                    }
                }

                // convolve mono input with combined HRIR
                OverlapAddConvolutionEffectParams overlapAddParams{};
                overlapAddParams.fftIR = mHRTF[band].data();
                
                auto bandState = mOverlapAddEffects[band]->apply(overlapAddParams, *mBandBuffers[band], subTempBuffer);
                if (bandState == AudioEffectState::TailRemaining) state = AudioEffectState::TailRemaining;
                
                AudioBuffer::mix(subTempBuffer, out);
                mPrevBinaural = true;
            }
            else
            {
                AmbisonicsPanningEffectParams panParams{};
                panParams.order = params.order;
                mAmbisonicsPanningEffect->apply(panParams, *mAmbisonicsBuffer, *mSpeakerBuffer);

                for (int i = 0; i < out.numChannels(); ++i)
                {
                    AudioBuffer outChannel(subTempBuffer, i);
                    GainEffectParams gainParams{};
                    gainParams.gain = (*mSpeakerBuffer)[i][0];

                    mGainEffectsPan[band][i]->apply(gainParams, *mBandBuffers[band], outChannel);
                }
                
                AudioBuffer::mix(subTempBuffer, out);
                mPrevBinaural = false;
            }
        }
    }
    else
    {
        for (int band = 0; band < Bands::kNumBands; ++band)
        {
            subTempBuffer.makeSilent();
            const float* bandSH = &params.shCoeffs[band * numCoeffs];

            for (int i = 0; i < numCoeffs; ++i)
            {
                AudioBuffer outChannel(subTempBuffer, i);
                GainEffectParams gainParams{};
                gainParams.gain = bandSH[i];

                mGainEffectsRaw[band][i]->apply(gainParams, *mBandBuffers[band], outChannel);
            }
            
            AudioBuffer::mix(subTempBuffer, out);
        }
        mPrevBinaural = false;
    }

    return state;
}

AudioEffectState PathEffect::tail(AudioBuffer& out)
{
    out.makeSilent();

    if (mSpatialize && mPrevBinaural)
    {
        AudioEffectState state = AudioEffectState::TailComplete;

        float* channelPointers[2];
        channelPointers[0] = (*mTempBuffer)[0];
        channelPointers[1] = (*mTempBuffer)[1];
        AudioBuffer subTempBuffer(2, out.numSamples(), channelPointers);

        for (int band = 0; band < Bands::kNumBands; ++band)
        {
            subTempBuffer.makeSilent();
            auto bandState = mOverlapAddEffects[band]->tail(subTempBuffer);
            if (bandState == AudioEffectState::TailRemaining) state = AudioEffectState::TailRemaining;
            
            AudioBuffer::mix(subTempBuffer, out);
        }
        return state;
    }

    return AudioEffectState::TailComplete;
}

int PathEffect::numTailSamplesRemaining() const
{
    if (mSpatialize && mPrevBinaural)
    {
        int maxTail = 0;
        for (int band = 0; band < Bands::kNumBands; ++band)
        {
            maxTail = std::max(maxTail, mOverlapAddEffects[band]->numTailSamplesRemaining());
        }
        return maxTail;
    }
    return 0;
}

}