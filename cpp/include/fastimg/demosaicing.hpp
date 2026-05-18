#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace fastimg {

enum class BayerPattern {
    RGGB,
    GRBG,
    BGGR,
    GBRG,
};

BayerPattern parse_pattern(std::string_view pattern);

void demosaic_bgr(
    const std::uint8_t *bayer,
    std::uint8_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

void demosaic_bgr(
    const std::uint16_t *bayer,
    std::uint16_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

void demosaic_bgr(
    const float *bayer,
    float *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

void demosaic_bgr(
    const double *bayer,
    double *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
);

}  // namespace fastimg
