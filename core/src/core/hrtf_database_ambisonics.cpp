#include "hrtf_database.h"

#include "array_math.h"
#include "sh.h"
#include "virtual_speaker_layout.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// HRTFDatabase: Ambisonics
// --------------------------------------------------------------------------------------------------------------------

void HRTFDatabase::precomputeAmbisonicsHRTFs()
{
    const auto numSpeakers = VirtualSpeakerLayout::kCount;
    const auto numCoeffs = SphericalHarmonics::numCoeffsForOrder(IHRTFMap::kMaxAmbisonicsOrder);
    const auto numBins = mFFTInterpolation.numComplexSamples;
    const auto quadrature = (4.0f * Math::kPi) / numSpeakers;

    Array<complex_t, 3> sampled(IHRTFMap::kNumEars, numSpeakers, numBins);
    Array<float, 2> basis(numSpeakers, numCoeffs);

    for (auto k = 0; k < numSpeakers; ++k)
    {
        const auto& direction = VirtualSpeakerLayout::kDirections[k];

        int indices[8];
        float weights[8];
        mHRTFMap->interpolatedHRIRWeights(direction, indices, weights);
        interpolateHRIRs(indices, weights, 1.0f, HRTFPhaseType::None);

        for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
        {
            memcpy(sampled[ear][k], mInterpolatedHRTF[ear], numBins * sizeof(complex_t));
        }

        SphericalHarmonics::projectSinglePoint(direction, IHRTFMap::kMaxAmbisonicsOrder, basis[k]);
    }

    // the virtual speakers are a t-design, so quadrature is the exact least-squares projection
    Array<complex_t, 3> projected(IHRTFMap::kNumEars, numCoeffs, numBins);
    projected.zero();

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        for (auto k = 0; k < numSpeakers; ++k)
        {
            for (auto i = 0; i < numCoeffs; ++i)
            {
                ArrayMath::scaleAccumulate(2 * numBins, reinterpret_cast<const float*>(sampled[ear][k]), quadrature * basis[k][i],
                                           reinterpret_cast<float*>(projected[ear][i]));
            }
        }
    }

    Array<float> hrir(mFFTAudioProcessing.numRealSamples);

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        for (auto i = 0; i < numCoeffs; ++i)
        {
            hrir.zero();
            mFFTInterpolation.applyInverse(projected[ear][i], hrir.data());
            memset(hrir.data() + numSamples(), 0, (mFFTInterpolation.numRealSamples - numSamples()) * sizeof(float));
            mFFTAudioProcessing.applyForward(hrir.data(), mAmbisonicsHRTF[ear][i]);
        }
    }

    fitFieldHRTFs(sampled, basis, projected);
}

// Least squares below a cutoff keeps interaural phase; magnitude least squares above it keeps level and spectral cues
// that truncation would smear. The cutoff follows N = kr, clamped to the range where interaural phase stops mattering.
// Phase is continued along the HRIR set's bulk delay, so high frequencies arrive with the onset instead of ~1 ms early:
//
//  Binaural Rendering of Ambisonic Signals via Magnitude Least Squares
//  C. Schoerkhuber, M. Zaunschirm, R. Hoeldrich
//  DAGA 2018
void HRTFDatabase::fitFieldHRTFs(const Array<complex_t, 3>& sampled,
                                 const Array<float, 2>& basis,
                                 const Array<complex_t, 3>& projected)
{
    const auto kSpeedOfSound = 343.0f;
    const auto kHeadRadius = 0.0875f;
    const auto kMinMagnitudeCutoff = 1500.0f;
    const auto kMaxMagnitudeCutoff = 2000.0f;

    const auto numSpeakers = VirtualSpeakerLayout::kCount;
    const auto numBins = mFFTInterpolation.numComplexSamples;
    const auto irSize = mFFTInterpolation.numRealSamples;
    const auto leadIn = irSize / 8;
    const auto delay = fieldDelay();
    const auto binWidth = static_cast<float>(mSamplingRate) / irSize;
    const auto quadrature = (4.0f * Math::kPi) / numSpeakers;

    Array<complex_t, 2> fitted(SphericalHarmonics::numCoeffsForOrder(IHRTFMap::kMaxAmbisonicsOrder), numBins);
    Array<complex_t> target(numSpeakers);
    Array<float> hrir(irSize);
    Array<float> fieldHRIR(mFFTField.numRealSamples);

    auto bulkDelay = 0.0f;

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        for (auto k = 0; k < numSpeakers; ++k)
        {
            mFFTInterpolation.applyInverse(sampled[ear][k], hrir.data());

            auto peak = 0;
            for (auto t = 1; t < irSize; ++t)
            {
                if (fabsf(hrir[t]) > fabsf(hrir[peak]))
                    peak = t;
            }

            bulkDelay += (peak < irSize / 2) ? peak : peak - irSize;
        }
    }

    bulkDelay /= IHRTFMap::kNumEars * numSpeakers;
    const auto phaseStep = -2.0f * Math::kPi * bulkDelay / irSize;

    for (auto order = 0; order <= IHRTFMap::kMaxAmbisonicsOrder; ++order)
    {
        auto numCoeffs = SphericalHarmonics::numCoeffsForOrder(order);
        auto cutoff = std::min(std::max(order * kSpeedOfSound / (2.0f * Math::kPi * kHeadRadius), kMinMagnitudeCutoff), kMaxMagnitudeCutoff);
        auto firstMagnitudeBin = std::min(numBins, static_cast<int>(ceilf(cutoff / binWidth)));

        for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
        {
            for (auto i = 0; i < numCoeffs; ++i)
            {
                memcpy(fitted[i], projected[ear][i], firstMagnitudeBin * sizeof(complex_t));
            }

            // phase is continued from the previous bin's reconstruction, magnitude comes from the HRTF
            for (auto bin = firstMagnitudeBin; bin < numBins; ++bin)
            {
                for (auto k = 0; k < numSpeakers; ++k)
                {
                    complex_t reconstructed = 0.0f;

                    for (auto i = 0; i < numCoeffs; ++i)
                    {
                        reconstructed += basis[k][i] * fitted[i][bin - 1];
                    }

                    target[k] = std::polar(std::abs(sampled[ear][k][bin]), std::arg(reconstructed) + phaseStep);
                }

                for (auto i = 0; i < numCoeffs; ++i)
                {
                    complex_t sum = 0.0f;

                    for (auto k = 0; k < numSpeakers; ++k)
                    {
                        sum += basis[k][i] * target[k];
                    }

                    fitted[i][bin] = quadrature * sum;
                }
            }

            // circular samples past irSize - leadIn are pre-ringing: they land ahead of the delayed onset
            for (auto i = 0; i < numCoeffs; ++i)
            {
                mFFTInterpolation.applyInverse(fitted[i], hrir.data());

                fieldHRIR.zero();

                for (auto t = -leadIn; t < irSize - leadIn; ++t)
                {
                    fieldHRIR[delay + t] = hrir[(t + irSize) % irSize];
                }

                mFFTField.applyForward(fieldHRIR.data(), mFieldHRTF[ear][fieldIndex(order, i)]);
            }
        }
    }
}

}
