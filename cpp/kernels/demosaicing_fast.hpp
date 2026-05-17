#pragma once

#include <cstddef>
#include <cstdint>

#include "kernels/demosaicing.hpp"

namespace bilinear {

void demosaic_bgr_fast(
    const std::uint8_t *bayer,
    std::uint8_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

void demosaic_bgr_fast(
    const std::uint16_t *bayer,
    std::uint16_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

void demosaic_bgr_fast(
    const float *bayer,
    float *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

void demosaic_bgr_fast(
    const double *bayer,
    double *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

}  // namespace bilinear
