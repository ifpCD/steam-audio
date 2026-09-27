#pragma once

#include "vector.h"

namespace ipl {

// Antipodal spherical 21-design: (4pi / kCount) sum_k Y_i(s_k) Y_j(s_k) = delta_ij for every degree <= kExactOrder,
// so sampling decoders and quadrature projections through it are exact at every order phonon renders.
namespace VirtualSpeakerLayout
{
    constexpr int kCount = 240;
    constexpr int kExactOrder = 10;

    extern const Vector3f kDirections[kCount];
}

}
