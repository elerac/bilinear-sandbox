#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string_view.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "kernels/demosaicing.hpp"
#include "kernels/demosaicing_fast.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using BayerImage = nb::ndarray<const T, nb::numpy, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

template <typename T>
using BgrImage = nb::ndarray<nb::numpy, T, nb::ndim<3>, nb::device::cpu, nb::c_contig>;

template <typename T>
BgrImage<T> demosaic_bgr(BayerImage<T> bayer, std::string_view pattern_name) {
    const bilinear::BayerPattern pattern = bilinear::parse_pattern(pattern_name);
    const std::size_t height = bayer.shape(0);
    const std::size_t width = bayer.shape(1);
    const std::size_t channels = 3;
    const std::size_t total = height * width * channels;

    T *output = new T[total]();
    nb::capsule owner(output, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });

    {
        nb::gil_scoped_release release;
        bilinear::demosaic_bgr(bayer.data(), output, height, width, pattern);
    }

    return BgrImage<T>(output, {height, width, channels}, owner);
}

template <typename T>
BgrImage<T> demosaic_bgr_fast(BayerImage<T> bayer, std::string_view pattern_name) {
    const bilinear::BayerPattern pattern = bilinear::parse_pattern(pattern_name);
    const std::size_t height = bayer.shape(0);
    const std::size_t width = bayer.shape(1);
    const std::size_t channels = 3;
    const std::size_t total = height * width * channels;

    T *output = new T[total];
    nb::capsule owner(output, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });

    {
        nb::gil_scoped_release release;
        bilinear::demosaic_bgr_fast(bayer.data(), output, height, width, pattern);
    }

    return BgrImage<T>(output, {height, width, channels}, owner);
}

}  // namespace

NB_MODULE(_native, m) {
    m.def(
        "_demosaic_bgr",
        &demosaic_bgr<std::uint8_t>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr",
        &demosaic_bgr<std::uint16_t>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr",
        &demosaic_bgr<float>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr",
        &demosaic_bgr<double>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_fast",
        &demosaic_bgr_fast<std::uint8_t>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_fast",
        &demosaic_bgr_fast<std::uint16_t>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_fast",
        &demosaic_bgr_fast<float>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_fast",
        &demosaic_bgr_fast<double>,
        "bayer"_a.noconvert(),
        "pattern"_a
    );
}
