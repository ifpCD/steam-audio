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
// Binaural renders the field as given (no max-rE): the per-order MagLS tables are fitted for unweighted input, and any
// taper on top only costs magnitude, ILD and low-frequency ITD accuracy. Spatial smoothing is the encoder's call.
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

    // group delay restored above the MagLS cutoff, ramped in over this many octaves (raised cosine in log frequency)
    static constexpr float kGroupDelayRampOctaves = 0.5f;
    // bands are weighted by their crossover share of [cutoff, this) when a per-band field sets the group delay
    static constexpr float kGroupDelayBandHigh = 8000.0f;

    // left/right += weights * sum_i coeffs[i] * H_i over blocks [firstBlock, endBlock), H_i from fieldHRTFBlocks(order).
    // weights/left/right: interleaved complex, indexed from bin 0, 32 B aligned.
    typedef void (*AccumulateFieldHRTF)(const float* blocks,
                                        int numCoeffs,
                                        const float* coeffs,
                                        int firstBlock,
                                        int endBlock,
                                        const float* weights,
                                        float* left,
                                        float* right);

    int mSamplingRate;
    int mFrameSize;
    bool mSpatialize;
    bool mPrevBinaural;

    int mOrders[Bands::kNumBands];
    float mCoeffs[Bands::kNumBands][AmbisonicField::kMaxCoeffs];
    bool mSharedShape; // mCoeffs[b] == mBandGains[b] * mShape
    float mShape[AmbisonicField::kMaxCoeffs];
    float mBandGains[Bands::kNumBands];

    unique_ptr<FFT> mFieldFFT;
    unique_ptr<FFT> mConvolutionFFT;
    unique_ptr<OverlapAddConvolutionEffect> mConvolution;
    int mNumFieldBlocks; // == hrtf->numFieldBlocks()
    int mBlockRange[Bands::kNumBands][2]; // [first, end) block with nonzero crossover weight
    Array<float, 2> mCrossover; // #bands * 2 * #fieldblockbins, weights sum to 1, duplicated for re/im
    Array<float> mSharedWeights; // sum_b gain_b * crossover_b, same layout
    Array<complex_t, 2> mFieldHRTF; // #ears * #fieldblockbins, ifft only reads #fieldspectrumsamples
    Array<float> mHRIR; // #convolutionsamples. tail past the field IR stays 0, ifft only writes the head
    Array<complex_t, 2> mHRTF; // #ears * #convolutionspectrumsamples.
    AccumulateFieldHRTF mAccumulateFieldHRTF; // float4/float8, picked in ctor from gSIMDLevel()
    int mConvolutionIRSize; // leading field IR samples convolved, see ctor

    Array<float> mGroupDelayPhase; // #fieldspectrumsamples. phase (rad) per sample of group delay, 0 below the cutoff
    int mGroupDelayFirstBin; // first bin the ramp touches
    int mGroupDelayLinearBin; // first bin past the ramp: phase grows by 2pi / #fieldsamples per bin from here
    float mGroupDelayBandWeights[Bands::kNumBands];

    IIRFilterer mBandFilters[Bands::kNumBands];
    Array<float, 2> mBandSignals; // #bands * #samples.
    DynamicMatrixf mDecoder; // #speakers * #coefficients.
    Array<float, 2> mPrevGains; // #bands * #channels.

    static const SHRotation& listenerRotation(const CoordinateSpace3f& listener);

    void loadField(const AmbisonicField& field,
                   const SHRotation* rotation,
                   bool maxRE);

    void buildCrossover();

    void buildGroupDelay();

    // per ear, the high-frequency arrival time the loaded field implies, restored as group delay above the MagLS cutoff
    void continueGroupDelay(const HRTFDatabase& hrtf);

    AudioEffectState applyBinaural(const HRTFDatabase& hrtf,
                                   const AudioBuffer& in,
                                   AudioBuffer& out);

    void applyGains(bool decode,
                    const AudioBuffer& in,
                    AudioBuffer& out);

    static void accumulateFieldHRTF_float4(const float* blocks,
                                           int numCoeffs,
                                           const float* coeffs,
                                           int firstBlock,
                                           int endBlock,
                                           const float* weights,
                                           float* left,
                                           float* right);

#if defined(IPL_ENABLE_FLOAT8)
    static void accumulateFieldHRTF_float8(const float* blocks,
                                           int numCoeffs,
                                           const float* coeffs,
                                           int firstBlock,
                                           int endBlock,
                                           const float* weights,
                                           float* left,
                                           float* right);
#endif
};

}
