#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "bands.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// AmbisonicField
// --------------------------------------------------------------------------------------------------------------------

// World-space, orthonormal ACN spherical harmonics per band; each band is truncated at its own order.
struct AmbisonicField
{
    static constexpr int kMaxOrder = 10;
    static constexpr int kMaxCoeffs = (kMaxOrder + 1) * (kMaxOrder + 1);

    int32_t orders[Bands::kNumBands];
    float coeffs[Bands::kNumBands][kMaxCoeffs];
};


// --------------------------------------------------------------------------------------------------------------------
// AmbisonicFieldExchange
// --------------------------------------------------------------------------------------------------------------------

// Lock-free single-producer single-consumer triple buffer: the consumer always acquires the latest published field.
class AmbisonicFieldExchange
{
public:
    AmbisonicFieldExchange()
        : mShared(1)
        , mBack(0)
        , mFront(2)
    {
        memset(mSlots, 0, sizeof(mSlots));
    }

    AmbisonicField& back()
    {
        return mSlots[mBack];
    }

    void publish()
    {
        mBack = mShared.exchange(mBack | kFresh, std::memory_order_acq_rel) & kSlotMask;
    }

    const AmbisonicField& acquire()
    {
        if (mShared.load(std::memory_order_relaxed) & kFresh)
            mFront = mShared.exchange(mFront, std::memory_order_acq_rel) & kSlotMask;

        return mSlots[mFront];
    }

private:
    static constexpr uint32_t kSlotMask = 3;
    static constexpr uint32_t kFresh = 4;

    AmbisonicField mSlots[3];
    std::atomic<uint32_t> mShared;
    uint32_t mBack;
    uint32_t mFront;
};

}
