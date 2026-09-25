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

#include "ambisonics_panning_effect.h"
#include "ambisonics_rotate_effect.h"
#include "bands.h"
#include "iir.h"
#include "gain_effect.h"
#include "hrtf_database.h"
#include "overlap_add_convolution_effect.h"
#include "sh.h"
#include <vector>

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

struct PathEffectSettings
{
    int maxOrder = 0;
    bool spatialize = false;
    const SpeakerLayout* speakerLayout = nullptr;
    const HRTFDatabase* hrtf = nullptr;
};

struct PathEffectParams
{
    // Pointer to 3 consecutive sets of SH coefficients (Low, Mid, High).
    // For 3rd order, this should point to 48 floats (16 * 3).
    const float* shCoeffs = nullptr; 
    
    int order = 0;
    bool binaural = false;
    const HRTFDatabase* hrtf = nullptr;
    const CoordinateSpace3f* listener = nullptr;
};

// Renders a crossover-split sound field using 3-band individual SH coefficients.
class PathEffect
{
public:
    PathEffect(const AudioSettings& audioSettings,
               const PathEffectSettings& effectSettings);

    // Resets the effect to its initial state.
    void reset();

    // Renders an audio buffer given the 3-band SH coefficients for a sound field.
    AudioEffectState apply(const PathEffectParams& params,
                           const AudioBuffer& in,
                           AudioBuffer& out);

    AudioEffectState tail(AudioBuffer& out);

    int numTailSamplesRemaining() const;

private:
    int mMaxOrder;
    bool mSpatialize;

    // Crossover state
    IIRFilterer mCrossover[Bands::kNumBands];
    unique_ptr<AudioBuffer> mBandBuffers[Bands::kNumBands];
    unique_ptr<AudioBuffer> mTempBuffer; 

    vector<unique_ptr<GainEffect>> mGainEffectsRaw[Bands::kNumBands]; // Raw Ambisonics out for when not spatial

    unique_ptr<AmbisonicsRotateEffect> mAmbisonicsRotateEffect; // For rotating the SH coefficients when spatializing.
    unique_ptr<AmbisonicsPanningEffect> mAmbisonicsPanningEffect; // For projecting SH coefficients to speaker gains when spatializing.
    unique_ptr<OverlapAddConvolutionEffect> mOverlapAddEffects[Bands::kNumBands]; // For applying an HRTF derived from rotated SH coefficients when spatializing.
    unique_ptr<AudioBuffer> mAmbisonicsBuffer; // Temp buffer for rotating SH coefficients when spatializing.
    unique_ptr<AudioBuffer> mSpeakerBuffer; // Temp buffer for calculating speaker gains when spatializing.
    
    vector<unique_ptr<GainEffect>> mGainEffectsPan[Bands::kNumBands];
    Array<complex_t, 2> mHRTF[Bands::kNumBands]; // Temp buffer for deriving a single HRTF from rotated SH coefficients when spatializing.
    bool mPrevBinaural;
};
}