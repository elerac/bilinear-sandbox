#include "bilinear/demosaicing.hpp"

#include <stdexcept>
#include <type_traits>

namespace bilinear {
namespace {

bool is_red(BayerPattern pattern, std::size_t y, std::size_t x) {
    const bool y_odd = (y % 2) != 0;
    const bool x_odd = (x % 2) != 0;

    switch (pattern) {
    case BayerPattern::RGGB:
        return !y_odd && !x_odd;
    case BayerPattern::GRBG:
        return !y_odd && x_odd;
    case BayerPattern::BGGR:
        return y_odd && x_odd;
    case BayerPattern::GBRG:
        return y_odd && !x_odd;
    }

    return false;
}

bool is_blue(BayerPattern pattern, std::size_t y, std::size_t x) {
    const bool y_odd = (y % 2) != 0;
    const bool x_odd = (x % 2) != 0;

    switch (pattern) {
    case BayerPattern::RGGB:
        return y_odd && x_odd;
    case BayerPattern::GRBG:
        return y_odd && !x_odd;
    case BayerPattern::BGGR:
        return !y_odd && !x_odd;
    case BayerPattern::GBRG:
        return !y_odd && x_odd;
    }

    return false;
}

bool horizontal_neighbors_are_red(BayerPattern pattern, std::size_t y, std::size_t x) {
    return is_red(pattern, y, x - 1) || is_red(pattern, y, x + 1);
}

template <typename T>
T avg2(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        return (a + b) / T{2};
    } else {
        return static_cast<T>((static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(b) + 1) / 2);
    }
}

template <typename T>
T avg4(T a, T b, T c, T d) {
    if constexpr (std::is_floating_point_v<T>) {
        return (a + b + c + d) / T{4};
    } else {
        return static_cast<T>(
            (
                static_cast<std::uint32_t>(a) +
                static_cast<std::uint32_t>(b) +
                static_cast<std::uint32_t>(c) +
                static_cast<std::uint32_t>(d) +
                2
            ) / 4
        );
    }
}

template <typename T>
T pixel(const T *bayer, std::size_t width, std::size_t y, std::size_t x) {
    return bayer[y * width + x];
}

template <typename T>
T *bgr_pixel(T *output, std::size_t width, std::size_t y, std::size_t x) {
    return &output[(y * width + x) * 3];
}

template <typename T>
void copy_inner_border(T *output, std::size_t height, std::size_t width) {
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t x = 0; x < width; ++x) {
            output[(0 * width + x) * 3 + channel] = output[(1 * width + x) * 3 + channel];
            output[((height - 1) * width + x) * 3 + channel] =
                output[((height - 2) * width + x) * 3 + channel];
        }

        for (std::size_t y = 0; y < height; ++y) {
            output[(y * width + 0) * 3 + channel] = output[(y * width + 1) * 3 + channel];
            output[(y * width + (width - 1)) * 3 + channel] =
                output[(y * width + (width - 2)) * 3 + channel];
        }
    }
}

template <typename T>
void demosaic_bgr_impl(
    const T *bayer,
    T *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    if (height < 3 || width < 3) {
        return;
    }

    for (std::size_t y = 1; y < height - 1; ++y) {
        for (std::size_t x = 1; x < width - 1; ++x) {
            T *out = bgr_pixel(output, width, y, x);

            if (is_red(pattern, y, x)) {
                out[2] = pixel(bayer, width, y, x);
                out[1] = static_cast<T>(avg4(
                    pixel(bayer, width, y - 1, x),
                    pixel(bayer, width, y + 1, x),
                    pixel(bayer, width, y, x - 1),
                    pixel(bayer, width, y, x + 1)
                ));
                out[0] = static_cast<T>(avg4(
                    pixel(bayer, width, y - 1, x - 1),
                    pixel(bayer, width, y - 1, x + 1),
                    pixel(bayer, width, y + 1, x - 1),
                    pixel(bayer, width, y + 1, x + 1)
                ));
            } else if (is_blue(pattern, y, x)) {
                out[0] = pixel(bayer, width, y, x);
                out[1] = static_cast<T>(avg4(
                    pixel(bayer, width, y - 1, x),
                    pixel(bayer, width, y + 1, x),
                    pixel(bayer, width, y, x - 1),
                    pixel(bayer, width, y, x + 1)
                ));
                out[2] = static_cast<T>(avg4(
                    pixel(bayer, width, y - 1, x - 1),
                    pixel(bayer, width, y - 1, x + 1),
                    pixel(bayer, width, y + 1, x - 1),
                    pixel(bayer, width, y + 1, x + 1)
                ));
            } else {
                out[1] = pixel(bayer, width, y, x);
                if (horizontal_neighbors_are_red(pattern, y, x)) {
                    out[2] = static_cast<T>(avg2(
                        pixel(bayer, width, y, x - 1),
                        pixel(bayer, width, y, x + 1)
                    ));
                    out[0] = static_cast<T>(avg2(
                        pixel(bayer, width, y - 1, x),
                        pixel(bayer, width, y + 1, x)
                    ));
                } else {
                    out[0] = static_cast<T>(avg2(
                        pixel(bayer, width, y, x - 1),
                        pixel(bayer, width, y, x + 1)
                    ));
                    out[2] = static_cast<T>(avg2(
                        pixel(bayer, width, y - 1, x),
                        pixel(bayer, width, y + 1, x)
                    ));
                }
            }
        }
    }

    copy_inner_border(output, height, width);
}

}  // namespace

BayerPattern parse_pattern(std::string_view pattern) {
    if (pattern == "RGGB") {
        return BayerPattern::RGGB;
    }
    if (pattern == "GRBG") {
        return BayerPattern::GRBG;
    }
    if (pattern == "BGGR") {
        return BayerPattern::BGGR;
    }
    if (pattern == "GBRG") {
        return BayerPattern::GBRG;
    }

    throw std::invalid_argument("unsupported Bayer pattern");
}

void demosaic_bgr(
    const std::uint8_t *bayer,
    std::uint8_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_impl(bayer, output, height, width, pattern);
}

void demosaic_bgr(
    const std::uint16_t *bayer,
    std::uint16_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_impl(bayer, output, height, width, pattern);
}

void demosaic_bgr(
    const float *bayer,
    float *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_impl(bayer, output, height, width, pattern);
}

void demosaic_bgr(
    const double *bayer,
    double *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_impl(bayer, output, height, width, pattern);
}

}  // namespace bilinear
