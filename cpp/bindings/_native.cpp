#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string_view.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "bilinear/demosaicing.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using BayerImage = nb::ndarray<const T, nb::numpy, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

template <typename T>
using BgrOutput = nb::ndarray<T, nb::numpy, nb::ndim<3>, nb::device::cpu, nb::c_contig>;

template <typename T>
void check_output_shape(BayerImage<T> bayer, BgrOutput<T> output) {
    const std::size_t height = bayer.shape(0);
    const std::size_t width = bayer.shape(1);

    if (output.shape(0) != height || output.shape(1) != width || output.shape(2) != 3) {
        throw std::runtime_error("output must have shape (height, width, 3)");
    }
}

template <typename T>
void demosaic_bgr_into(BayerImage<T> bayer, BgrOutput<T> output, std::string_view pattern_name) {
    check_output_shape(bayer, output);

    const bilinear::BayerPattern pattern = bilinear::parse_pattern(pattern_name);
    const std::size_t height = bayer.shape(0);
    const std::size_t width = bayer.shape(1);
    const T *bayer_ptr = bayer.data();
    T *output_ptr = output.data();

    {
        nb::gil_scoped_release release;
        bilinear::demosaic_bgr(bayer_ptr, output_ptr, height, width, pattern);
    }
}

}  // namespace

NB_MODULE(_native, m) {
    m.def(
        "_demosaic_bgr_into",
        &demosaic_bgr_into<std::uint8_t>,
        "bayer"_a.noconvert(),
        "output"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_into",
        &demosaic_bgr_into<std::uint16_t>,
        "bayer"_a.noconvert(),
        "output"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_into",
        &demosaic_bgr_into<float>,
        "bayer"_a.noconvert(),
        "output"_a.noconvert(),
        "pattern"_a
    );
    m.def(
        "_demosaic_bgr_into",
        &demosaic_bgr_into<double>,
        "bayer"_a.noconvert(),
        "output"_a.noconvert(),
        "pattern"_a
    );
}
