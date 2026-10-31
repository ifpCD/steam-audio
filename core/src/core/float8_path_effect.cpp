#if defined(IPL_ENABLE_FLOAT8)

#include "float8.h"
#include "path_effect.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// PathEffect
// --------------------------------------------------------------------------------------------------------------------

// AVX2 + FMA only (gSIMDLevel() >= AVX2). 8 accumulators = 16 bins x 2 ears, enough chains to cover FMA latency.
// TODO: this TU pulls in path_effect.h, so Vector3f::kZero's dynamic init also gets built /arch:AVX2 and can run at
// load on any cpu. move the kernel behind a tiny header before shipping.
void IPL_FLOAT8_FMA_ATTR PathEffect::accumulateFieldHRTF_float8(const float* blocks,
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

    for (auto block = firstBlock; block < endBlock; ++block)
    {
        const auto* row = &blocks[static_cast<size_t>(block) * numCoeffs * rowFloats];

        auto l0 = float8::zero();
        auto l1 = float8::zero();
        auto l2 = float8::zero();
        auto l3 = float8::zero();
        auto r0 = float8::zero();
        auto r1 = float8::zero();
        auto r2 = float8::zero();
        auto r3 = float8::zero();

        for (auto i = 0; i < numCoeffs; ++i, row += rowFloats)
        {
            const auto c = float8::set1(coeffs[i]);

            l0 = float8::fmadd(float8::load(row + 0), c, l0);
            l1 = float8::fmadd(float8::load(row + 8), c, l1);
            l2 = float8::fmadd(float8::load(row + 16), c, l2);
            l3 = float8::fmadd(float8::load(row + 24), c, l3);
            r0 = float8::fmadd(float8::load(row + earFloats + 0), c, r0);
            r1 = float8::fmadd(float8::load(row + earFloats + 8), c, r1);
            r2 = float8::fmadd(float8::load(row + earFloats + 16), c, r2);
            r3 = float8::fmadd(float8::load(row + earFloats + 24), c, r3);
        }

        const auto offset = block * earFloats;
        const auto* w = weights + offset;
        auto* outLeft = left + offset;
        auto* outRight = right + offset;

        float8::store(outLeft + 0, float8::fmadd(l0, float8::load(w + 0), float8::load(outLeft + 0)));
        float8::store(outLeft + 8, float8::fmadd(l1, float8::load(w + 8), float8::load(outLeft + 8)));
        float8::store(outLeft + 16, float8::fmadd(l2, float8::load(w + 16), float8::load(outLeft + 16)));
        float8::store(outLeft + 24, float8::fmadd(l3, float8::load(w + 24), float8::load(outLeft + 24)));
        float8::store(outRight + 0, float8::fmadd(r0, float8::load(w + 0), float8::load(outRight + 0)));
        float8::store(outRight + 8, float8::fmadd(r1, float8::load(w + 8), float8::load(outRight + 8)));
        float8::store(outRight + 16, float8::fmadd(r2, float8::load(w + 16), float8::load(outRight + 16)));
        float8::store(outRight + 24, float8::fmadd(r3, float8::load(w + 24), float8::load(outRight + 24)));
    }

    float8::avoidTransitionPenalty();
}

}

#endif
