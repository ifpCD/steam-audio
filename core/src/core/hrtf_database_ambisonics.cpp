#include "hrtf_database.h"

#include "array_math.h"
#include "float4.h"
#include "sh.h"
#include "virtual_speaker_layout.h"

namespace ipl {

namespace {

// out[0, count) += a[0, count) * s
inline void axpy(int count, const float* a, float s, float* out)
{
    auto i = 0;
    auto v = float4::set1(s);
    for (; i + 4 <= count; i += 4)
    {
        float4::storeu(&out[i], float4::add(float4::loadu(&out[i]), float4::mul(float4::loadu(&a[i]), v)));
    }
    for (; i < count; ++i)
    {
        out[i] += a[i] * s;
    }
}

// out0 += a * s0, out1 += a * s1
inline void axpy2(int count, const float* a, float s0, float* out0, float s1, float* out1)
{
    auto i = 0;
    auto v0 = float4::set1(s0);
    auto v1 = float4::set1(s1);
    for (; i + 4 <= count; i += 4)
    {
        auto x = float4::loadu(&a[i]);
        float4::storeu(&out0[i], float4::add(float4::loadu(&out0[i]), float4::mul(x, v0)));
        float4::storeu(&out1[i], float4::add(float4::loadu(&out1[i]), float4::mul(x, v1)));
    }
    for (; i < count; ++i)
    {
        out0[i] += a[i] * s0;
        out1[i] += a[i] * s1;
    }
}

}

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
    fitFieldDelays(sampled, basis);
}

// Least squares below a cutoff keeps interaural phase; magnitude least squares above it keeps level and spectral cues
// that truncation would smear. The cutoff follows N = kr, clamped to the range where interaural phase stops mattering.
// Phase is continued along the HRIR set's bulk delay, so high frequencies arrive with the onset instead of ~1 ms early:
//
//  Binaural Rendering of Ambisonic Signals via Magnitude Least Squares
//  C. Schoerkhuber, M. Zaunschirm, R. Hoeldrich
//  DAGA 2018
//
// The magnitude fit is ILD-aware. Plain MagLS fits each ear on its own in linear magnitude: the loud (ipsilateral)
// side dominates, the shadowed side comes out several dB off, and since the two ears' errors are unrelated their
// difference lands in the ILD. Above the cutoff, per bin:
//  - relative error: directions weigh by their third-octave band level^-p (p = 1.5), so both sides fit in about the
//    same dB terms. One weighted projection per band and ear, (Y^T W Y)^-1 Y^T W.
//  - coupled ears: a few fixed-point iterations on sum_k eL^2 + eR^2 + lambda (eL - eR)^2, e the log-magnitude error.
//    Each ear's target takes beta = lambda / (1 + lambda) of the other ear's error, which turns differential (ILD) error
//    into common (timbre) error. Fewer iterations past 64 coefficients, where 240 points start to be overfit.
//  - time window: per-bin iterations add fine structure the ear does not resolve, which spreads the responses in time.
//    Each coefficient's response is windowed to the HRIR's span; the least-squares bins are kept exactly.
// On the default HRTF at its measured directions this cuts ILD error 15-27% through order 8 (5% at order 10, near the
// data's own spread), with magnitude error equal or lower and less pre-onset energy than plain MagLS:
//
//  iMagLS: Interaural Level Difference with Magnitude Least-Squares Loss for Optimized First-Order HRTF
//  O. Berebi, Z. Ben-Hur, D. L. Alon, B. Rafaely
//
//  Interaural Level Difference Optimization of Binaural Ambisonic Rendering
//  T. McKenzie, D. T. Murphy, G. Kearney
//  Applied Sciences 2019
void HRTFDatabase::fitFieldHRTFs(const Array<complex_t, 3>& sampled,
                                 const Array<float, 2>& basis,
                                 const Array<complex_t, 3>& projected)
{
    const auto kSpeedOfSound = 343.0f;
    const auto kHeadRadius = 0.0875f;
    const auto kMinMagnitudeCutoff = kFieldMagnitudeCutoffMin;
    const auto kMaxMagnitudeCutoff = kFieldMagnitudeCutoffMax;
    const auto kWeightExponent = 1.5f;
    const auto kWeightRange = 31.6227766f; // +-30 dB around each band's median weight
    const auto kCoeffsForFullIterations = 64;
    const auto kFullIterations = 4;
    const auto kReducedIterations = 2;
    // lambda = 1 -> beta = 1/2: each ear's target magnitude is scaled by exp(beta * other ear's log error), i.e. by
    // sqrt(|reconstructed| / |target|) of the other ear.

    const auto numSpeakers = VirtualSpeakerLayout::kCount;
    const auto numBins = mFFTInterpolation.numComplexSamples;
    const auto irSize = mFFTInterpolation.numRealSamples;
    const auto leadIn = irSize / 8;
    const auto delay = fieldDelay();
    const auto binWidth = static_cast<float>(mSamplingRate) / irSize;
    const auto maxCoeffs = SphericalHarmonics::numCoeffsForOrder(IHRTFMap::kMaxAmbisonicsOrder);
    const auto numEars = IHRTFMap::kNumEars;

    Array<complex_t, 3> fitted(numEars, maxCoeffs, numBins);
    Array<float, 2> magnitude(numEars, numSpeakers);
    Array<float, 2> reconstructedRe(numEars, numSpeakers);
    Array<float, 2> reconstructedIm(numEars, numSpeakers);
    Array<float, 2> reconstructedAbs(numEars, numSpeakers);
    Array<float> targetRe(numSpeakers);
    Array<float> targetIm(numSpeakers);
    Array<float> coeffsRe(maxCoeffs);
    Array<float> coeffsIm(maxCoeffs);
    Array<float> hrir(irSize);
    Array<complex_t> spectrum(numBins);
    Array<float> fieldHRIR(mFFTField.numRealSamples);
    Array<complex_t> fieldSpectrum(mFFTField.numComplexSamples);

    // third-octave band of every bin (relative to 1 kHz)
    vector<int> binBand(numBins);
    auto minBand = std::numeric_limits<int>::max();
    auto maxBand = std::numeric_limits<int>::min();
    for (auto bin = 0; bin < numBins; ++bin)
    {
        binBand[bin] = static_cast<int>(lroundf(3.0f * log2f(std::max(bin, 1) * binWidth / 1000.0f)));
        minBand = std::min(minBand, binBand[bin]);
        maxBand = std::max(maxBand, binBand[bin]);
    }

    const auto numBands = maxBand - minBand + 1;

    // per ear and band: direction weights w_k = (band level)^(-p/2), normalized to the band's median and clamped;
    // W Y laid out [speaker][coefficient]; and the Cholesky factor L of Y^T W Y at the maximum order, row-major and
    // transposed. A leading block of a Cholesky factor is the factor of the leading block, so every order reuses it.
    // Every hot loop below is an axpy over contiguous memory (no reductions), so it vectorizes under strict FP.
    vector<float> weightedBasis(static_cast<size_t>(numEars) * numBands * numSpeakers * maxCoeffs);
    vector<double> cholesky(static_cast<size_t>(numEars) * numBands * maxCoeffs * maxCoeffs);
    vector<float> factorRows(static_cast<size_t>(numEars) * numBands * maxCoeffs * maxCoeffs);
    vector<float> factorColumns(static_cast<size_t>(numEars) * numBands * maxCoeffs * maxCoeffs);

    auto weightedBasisOf = [&](int ear, int band) { return &weightedBasis[(static_cast<size_t>(ear) * numBands + band) * numSpeakers * maxCoeffs]; };
    auto choleskyOf = [&](int ear, int band) { return &cholesky[(static_cast<size_t>(ear) * numBands + band) * maxCoeffs * maxCoeffs]; };
    auto factorRowsOf = [&](int ear, int band) { return &factorRows[(static_cast<size_t>(ear) * numBands + band) * maxCoeffs * maxCoeffs]; };
    auto factorColumnsOf = [&](int ear, int band) { return &factorColumns[(static_cast<size_t>(ear) * numBands + band) * maxCoeffs * maxCoeffs]; };

    {
        Array<float, 3> bandPower(numEars, numBands, numSpeakers);
        vector<int> bandBins(numBands, 0);
        vector<float> weights(numSpeakers);
        vector<float> sorted(numSpeakers);
        vector<float> gram(static_cast<size_t>(maxCoeffs) * maxCoeffs);

        bandPower.zero();

        for (auto bin = 0; bin < numBins; ++bin)
        {
            auto band = binBand[bin] - minBand;
            ++bandBins[band];

            for (auto ear = 0; ear < numEars; ++ear)
            {
                for (auto k = 0; k < numSpeakers; ++k)
                {
                    bandPower[ear][band][k] += std::norm(sampled[ear][k][bin]);
                }
            }
        }

        for (auto ear = 0; ear < numEars; ++ear)
        {
            for (auto band = 0; band < numBands; ++band)
            {
                if (bandBins[band] == 0)
                    continue;

                for (auto k = 0; k < numSpeakers; ++k)
                {
                    auto power = std::max(bandPower[ear][band][k] / bandBins[band], 1e-30f);
                    weights[k] = powf(power, -0.5f * kWeightExponent);
                    sorted[k] = weights[k];
                }

                std::nth_element(sorted.begin(), sorted.begin() + numSpeakers / 2, sorted.end());
                auto median = sorted[numSpeakers / 2];

                for (auto k = 0; k < numSpeakers; ++k)
                {
                    weights[k] = std::min(std::max(weights[k] / median, 1.0f / kWeightRange), kWeightRange);
                }

                auto* wy = weightedBasisOf(ear, band);
                auto* factor = choleskyOf(ear, band);

                for (auto k = 0; k < numSpeakers; ++k)
                {
                    for (auto i = 0; i < maxCoeffs; ++i)
                    {
                        wy[k * maxCoeffs + i] = weights[k] * basis[k][i];
                    }
                }

                // Y^T W Y as a sum of rank-1 updates, lower triangle
                std::fill(gram.begin(), gram.end(), 0.0f);
                for (auto k = 0; k < numSpeakers; ++k)
                {
                    const auto* y = basis[k];
                    const auto* wyk = &wy[k * maxCoeffs];

                    for (auto i = 0; i < maxCoeffs; ++i)
                    {
                        axpy(i + 1, y, wyk[i], &gram[i * maxCoeffs]);
                    }
                }

                auto trace = 0.0;
                for (auto i = 0; i < maxCoeffs; ++i)
                {
                    for (auto j = 0; j <= i; ++j)
                    {
                        factor[i * maxCoeffs + j] = gram[i * maxCoeffs + j];
                    }
                    trace += factor[i * maxCoeffs + i];
                }

                auto ridge = 1e-7 * trace / maxCoeffs;
                for (auto i = 0; i < maxCoeffs; ++i)
                {
                    for (auto j = 0; j <= i; ++j)
                    {
                        auto sum = factor[i * maxCoeffs + j] + ((i == j) ? ridge : 0.0);
                        for (auto m = 0; m < j; ++m)
                        {
                            sum -= factor[i * maxCoeffs + m] * factor[j * maxCoeffs + m];
                        }
                        factor[i * maxCoeffs + j] = (i == j) ? sqrt(std::max(sum, 1e-300)) : sum / factor[j * maxCoeffs + j];
                    }
                }

                // float copies for the solves: the weights span 60 dB at most, which keeps Y^T W Y well conditioned
                auto* rows = factorRowsOf(ear, band);
                auto* columns = factorColumnsOf(ear, band);
                for (auto i = 0; i < maxCoeffs; ++i)
                {
                    for (auto j = 0; j < maxCoeffs; ++j)
                    {
                        auto value = (j <= i) ? static_cast<float>(factor[i * maxCoeffs + j]) : 0.0f;
                        rows[i * maxCoeffs + j] = value;
                        columns[j * maxCoeffs + i] = value;
                    }
                }
            }
        }
    }

    vector<float> basisT(static_cast<size_t>(maxCoeffs) * numSpeakers);
    for (auto k = 0; k < numSpeakers; ++k)
    {
        for (auto i = 0; i < maxCoeffs; ++i)
        {
            basisT[i * numSpeakers + k] = basis[k][i];
        }
    }

    // complex reconstruction at every speaker from one bin's coefficients
    auto reconstruct = [&](int ear, int bin, int numCoeffs)
    {
        for (auto i = 0; i < numCoeffs; ++i)
        {
            coeffsRe[i] = fitted[ear][i][bin].real();
            coeffsIm[i] = fitted[ear][i][bin].imag();
        }

        auto* re = reconstructedRe[ear];
        auto* im = reconstructedIm[ear];
        auto* abs = reconstructedAbs[ear];

        memset(re, 0, numSpeakers * sizeof(float));
        memset(im, 0, numSpeakers * sizeof(float));

        for (auto i = 0; i < numCoeffs; ++i)
        {
            axpy2(numSpeakers, &basisT[i * numSpeakers], coeffsRe[i], re, coeffsIm[i], im);
        }

        for (auto k = 0; k < numSpeakers; ++k)
        {
            abs[k] = std::max(sqrtf(re[k] * re[k] + im[k] * im[k]), 1e-30f);
        }
    };

    // weighted least squares: (Y^T W Y) c = Y^T W target, solved with L and L^T
    auto project = [&](int ear, int band, int bin, int numCoeffs)
    {
        const auto* wy = weightedBasisOf(ear, band);
        const auto* rows = factorRowsOf(ear, band);
        const auto* columns = factorColumnsOf(ear, band);
        auto* re = coeffsRe.data();
        auto* im = coeffsIm.data();

        memset(re, 0, numCoeffs * sizeof(float));
        memset(im, 0, numCoeffs * sizeof(float));

        for (auto k = 0; k < numSpeakers; ++k)
        {
            axpy2(numCoeffs, &wy[k * maxCoeffs], targetRe[k], re, targetIm[k], im);
        }

        // L y = rhs, column by column
        for (auto j = 0; j < numCoeffs; ++j)
        {
            const auto* column = &columns[j * maxCoeffs];
            re[j] /= column[j];
            im[j] /= column[j];
            axpy2(numCoeffs - j - 1, &column[j + 1], -re[j], &re[j + 1], -im[j], &im[j + 1]);
        }

        // L^T c = y, column by column (column j of L^T is row j of L)
        for (auto j = numCoeffs - 1; j >= 0; --j)
        {
            const auto* row = &rows[j * maxCoeffs];
            re[j] /= row[j];
            im[j] /= row[j];
            axpy2(j, row, -re[j], re, -im[j], im);
        }

        for (auto i = 0; i < numCoeffs; ++i)
        {
            fitted[ear][i][bin] = complex_t(re[i], im[i]);
        }
    };

    // the HRIR's span on the fit's circular grid (signed time, lead-in wrapped to the end): full from irSize / 64 to
    // 11/16 irSize, raised-cosine rise over irSize / 32 before it and fall over irSize / 16 after
    Array<float> window(irSize);
    {
        const auto start = irSize / 64;
        const auto end = 11 * irSize / 16;
        const auto rise = irSize / 32;
        const auto fall = irSize / 16;

        for (auto t = 0; t < irSize; ++t)
        {
            auto s = (t >= irSize - leadIn) ? t - irSize : t;
            auto w = 0.0f;

            if (start <= s && s < end)
                w = 1.0f;
            else if (start - rise <= s && s < start)
                w = 0.5f - 0.5f * cosf(Math::kPi * (s - (start - rise)) / rise);
            else if (end <= s && s < end + fall)
                w = 0.5f + 0.5f * cosf(Math::kPi * (s - end) / fall);

            window[t] = w;
        }
    }

    auto bulkDelay = 0.0f;

    for (auto ear = 0; ear < numEars; ++ear)
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

    bulkDelay /= numEars * numSpeakers;
    const auto phaseStep = -2.0f * Math::kPi * bulkDelay / irSize;
    const auto stepRe = cosf(phaseStep);
    const auto stepIm = sinf(phaseStep);

    for (auto order = 0; order <= IHRTFMap::kMaxAmbisonicsOrder; ++order)
    {
        auto numCoeffs = SphericalHarmonics::numCoeffsForOrder(order);
        auto cutoff = std::min(std::max(order * kSpeedOfSound / (2.0f * Math::kPi * kHeadRadius), kMinMagnitudeCutoff), kMaxMagnitudeCutoff);
        auto firstMagnitudeBin = std::min(numBins, static_cast<int>(ceilf(cutoff / binWidth)));
        auto iterations = (numCoeffs <= kCoeffsForFullIterations) ? kFullIterations : kReducedIterations;

        for (auto ear = 0; ear < numEars; ++ear)
        {
            for (auto i = 0; i < numCoeffs; ++i)
            {
                memcpy(fitted[ear][i], projected[ear][i], firstMagnitudeBin * sizeof(complex_t));
            }
        }

        for (auto bin = firstMagnitudeBin; bin < numBins; ++bin)
        {
            auto band = binBand[bin] - minBand;

            for (auto ear = 0; ear < numEars; ++ear)
            {
                for (auto k = 0; k < numSpeakers; ++k)
                {
                    magnitude[ear][k] = std::max(std::abs(sampled[ear][k][bin]), 1e-20f);
                }

                // phase is continued from the previous bin's reconstruction, magnitude comes from the HRTF
                reconstruct(ear, bin - 1, numCoeffs);

                for (auto k = 0; k < numSpeakers; ++k)
                {
                    auto scale = magnitude[ear][k] / reconstructedAbs[ear][k];
                    auto re = reconstructedRe[ear][k] * scale;
                    auto im = reconstructedIm[ear][k] * scale;
                    targetRe[k] = re * stepRe - im * stepIm;
                    targetIm[k] = re * stepIm + im * stepRe;
                }

                project(ear, band, bin, numCoeffs);
            }

            for (auto iteration = 0; iteration < iterations; ++iteration)
            {
                for (auto ear = 0; ear < numEars; ++ear)
                {
                    reconstruct(ear, bin, numCoeffs);
                }

                for (auto ear = 0; ear < numEars; ++ear)
                {
                    auto other = 1 - ear;

                    for (auto k = 0; k < numSpeakers; ++k)
                    {
                        auto coupled = magnitude[ear][k] * sqrtf(reconstructedAbs[other][k] / magnitude[other][k]);
                        auto scale = coupled / reconstructedAbs[ear][k];
                        targetRe[k] = reconstructedRe[ear][k] * scale;
                        targetIm[k] = reconstructedIm[ear][k] * scale;
                    }

                    project(ear, band, bin, numCoeffs);
                }
            }
        }

        for (auto ear = 0; ear < numEars; ++ear)
        {
            // window the magnitude-fit bins to the HRIR's span; the least-squares bins stay exact
            if (firstMagnitudeBin < numBins)
            {
                for (auto i = 0; i < numCoeffs; ++i)
                {
                    mFFTInterpolation.applyInverse(fitted[ear][i], hrir.data());
                    ArrayMath::multiply(irSize, hrir.data(), window.data(), hrir.data());
                    mFFTInterpolation.applyForward(hrir.data(), spectrum.data());
                    memcpy(&fitted[ear][i][firstMagnitudeBin], &spectrum[firstMagnitudeBin], (numBins - firstMagnitudeBin) * sizeof(complex_t));
                }
            }

            // circular samples past irSize - leadIn are pre-ringing: they land ahead of the delayed onset
            for (auto i = 0; i < numCoeffs; ++i)
            {
                mFFTInterpolation.applyInverse(fitted[ear][i], hrir.data());

                fieldHRIR.zero();

                for (auto t = -leadIn; t < irSize - leadIn; ++t)
                {
                    fieldHRIR[delay + t] = hrir[(t + irSize) % irSize];
                }

                mFFTField.applyForward(fieldHRIR.data(), fieldSpectrum.data());
                storeFieldHRTF(order, i, ear, fieldSpectrum.data());
            }
        }
    }
}

// Per ear and virtual speaker, when the high-frequency envelope arrives: the -10 dB onset of the 2.5-8 kHz Hilbert
// envelope, oversampled 16x by zero-padding the one-sided spectrum. Onsets are taken relative to their mean over speakers
// and ears, which is where MagLS's bulk-delay phase continuation puts every direction, and projected to
// kFieldDelayOrder. Arrival time is smooth over direction (mostly the first-order ITD lobe), so a low order holds it.
// Same idea as time-aligned / bilateral rendering, but applied as a parameter of the field rather than of the signal:
//
//  Binaural rendering of Ambisonic signals by head-related impulse response time alignment and a diffuseness constraint
//  M. Zaunschirm, C. Schoerkhuber, R. Hoeldrich
//  JASA 2018
//
//  Magnitude-Least-Squares Binaural Ambisonic Rendering with Phase Continuation
//  C. Hold, N. Meyer-Kahlen, V. Pulkki
//  DAGA 2023
void HRTFDatabase::fitFieldDelays(const Array<complex_t, 3>& sampled,
                                  const Array<float, 2>& basis)
{
    const auto kLowFrequency = 2500.0f;
    const auto kHighFrequency = 8000.0f;
    const auto kOversampling = 16;
    const auto kThreshold = 0.1f; // -10 dB, on squared envelope

    const auto numSpeakers = VirtualSpeakerLayout::kCount;
    const auto numBins = mFFTInterpolation.numComplexSamples;
    const auto irSize = mFFTInterpolation.numRealSamples;
    const auto leadIn = irSize / 8;
    const auto binWidth = static_cast<float>(mSamplingRate) / irSize;
    const auto quadrature = (4.0f * Math::kPi) / numSpeakers;

    FFT oversampled(irSize * kOversampling, FFTDomain::Complex);
    const auto numOversampled = oversampled.numComplexSamples;

    Array<complex_t> spectrum(numOversampled);
    Array<complex_t> analytic(numOversampled);
    Array<float, 2> onsets(IHRTFMap::kNumEars, numSpeakers);

    auto mean = 0.0;

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        for (auto k = 0; k < numSpeakers; ++k)
        {
            spectrum.zero();

            for (auto bin = 1; bin < numBins - 1; ++bin)
            {
                auto frequency = bin * binWidth;
                if (kLowFrequency <= frequency && frequency < kHighFrequency)
                {
                    spectrum[bin] = sampled[ear][k][bin];
                }
            }

            oversampled.applyInverse(spectrum.data(), analytic.data());

            auto peak = 0.0f;
            for (auto t = 0; t < numOversampled; ++t)
            {
                peak = std::max(peak, std::norm(analytic[t]));
            }

            // scan from leadIn samples before t = 0 (circular), like the lead-in the field placement keeps
            auto start = numOversampled - leadIn * kOversampling;
            auto onset = 0.0f;
            for (auto i = 0; i < numOversampled; ++i)
            {
                if (std::norm(analytic[(start + i) % numOversampled]) >= kThreshold * peak)
                {
                    onset = static_cast<float>(i) / kOversampling - leadIn;
                    break;
                }
            }

            onsets[ear][k] = onset;
            mean += onset;
        }
    }

    mean /= IHRTFMap::kNumEars * numSpeakers;

    for (auto ear = 0; ear < IHRTFMap::kNumEars; ++ear)
    {
        mFieldDelayRange[ear][0] = std::numeric_limits<float>::max();
        mFieldDelayRange[ear][1] = -std::numeric_limits<float>::max();

        for (auto i = 0; i < kFieldDelayCoeffs; ++i)
        {
            mFieldDelayCoeffs[ear][i] = 0.0f;
        }

        for (auto k = 0; k < numSpeakers; ++k)
        {
            auto relative = static_cast<float>(onsets[ear][k] - mean);

            mFieldDelayRange[ear][0] = std::min(mFieldDelayRange[ear][0], relative);
            mFieldDelayRange[ear][1] = std::max(mFieldDelayRange[ear][1], relative);

            for (auto i = 0; i < kFieldDelayCoeffs; ++i)
            {
                mFieldDelayCoeffs[ear][i] += quadrature * relative * basis[k][i];
            }
        }
    }
}

void HRTFDatabase::storeFieldHRTF(int order,
                                  int channel,
                                  int ear,
                                  const complex_t* spectrum)
{
    const auto numBins = mFFTField.numComplexSamples;
    const auto numCoeffs = SphericalHarmonics::numCoeffsForOrder(order);
    auto* blocks = &mFieldHRTF[static_cast<size_t>(fieldIndex(order, 0)) * mNumFieldBlocks * kFieldRowFloats];

    for (auto block = 0; block < mNumFieldBlocks; ++block)
    {
        auto* row = &blocks[(static_cast<size_t>(block) * numCoeffs + channel) * kFieldRowFloats + ear * kFieldEarFloats];

        for (auto k = 0; k < kFieldBlockBins; ++k)
        {
            const auto bin = block * kFieldBlockBins + k;
            const auto value = (bin < numBins) ? spectrum[bin] : complex_t(0.0f, 0.0f);

            row[2 * k] = value.real();
            row[2 * k + 1] = value.imag();
        }
    }
}

}
