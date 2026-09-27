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

#pragma once

#include "ambisonic_field.h"
#include "audio_buffer.h"
#include "hrtf_database.h"
#include "iir.h"
#include "matrix.h"
#include "overlap_add_convolution_effect.h"
#include "sh.h"
#include "speaker_layout.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

struct PathEffectSettings
{
    bool spatialize = false;
    const SpeakerLayout* speakerLayout = nullptr;
    const HRTFDatabase* hrtf = nullptr;
};

struct PathEffectParams
{
    const AmbisonicField* field = nullptr;
    bool binaural = false;
    const HRTFDatabase* hrtf = nullptr;
    const CoordinateSpace3f* listener = nullptr;
};

// Renders a per-band ambisonic field. Binaural rendering is a single HRTF convolution whose spherical harmonic weights
// cross over between bands in frequency; speaker and raw Ambisonic rendering split the input with IIR band filters.
class PathEffect
{
public:
    PathEffect(const AudioSettings& audioSettings,
               const PathEffectSettings& effectSettings);

    void reset();

    AudioEffectState apply(const PathEffectParams& params,
                           const AudioBuffer& in,
                           AudioBuffer& out);

    AudioEffectState tail(AudioBuffer& out);

    int numTailSamplesRemaining() const;

private:
    static constexpr int kMaxChannels = AmbisonicField::kMaxCoeffs;
    static constexpr float kCrossoverOctaves = 1.0f;

    int mSamplingRate;
    int mFrameSize;
    bool mSpatialize;
    bool mPrevBinaural;

    int mOrders[Bands::kNumBands];
    float mCoeffs[Bands::kNumBands][AmbisonicField::kMaxCoeffs];

    unique_ptr<FFT> mFieldFFT;
    unique_ptr<FFT> mConvolutionFFT;
    unique_ptr<OverlapAddConvolutionEffect> mConvolution;
    int mSupport[Bands::kNumBands][2]; // First and one-past-last field spectrum bin with nonzero crossover weight.
    Array<float, 2> mCrossover; // Band weights summing to 1. #bands * #fieldspectrumsamples.
    Array<complex_t, 2> mBandHRTF; // #bands * #fieldspectrumsamples.
    Array<complex_t> mFieldHRTF; // #fieldspectrumsamples.
    Array<float> mHRIR; // #convolutionsamples.
    Array<complex_t, 2> mHRTF; // #ears * #convolutionspectrumsamples.

    IIRFilterer mBandFilters[Bands::kNumBands];
    Array<float, 2> mBandSignals; // #bands * #samples.
    DynamicMatrixf mDecoder; // #speakers * #coefficients.
    Array<float, 2> mPrevGains; // #bands * #channels.

    static const SHRotation& listenerRotation(const CoordinateSpace3f& listener);

    void loadField(const AmbisonicField& field,
                   const SHRotation* rotation,
                   bool maxRE);

    void buildCrossover();

    AudioEffectState applyBinaural(const HRTFDatabase& hrtf,
                                   const AudioBuffer& in,
                                   AudioBuffer& out);

    void applyGains(bool decode,
                    const AudioBuffer& in,
                    AudioBuffer& out);
};

}
