#pragma once

#include "common/aligned_vector.hpp"
#include "fft/fftw_abi.hpp"

#include <dualsynth/mdspan.hpp>

#include <array>
#include <complex>
#include <cstddef>

namespace neo_fft3d {

template <class T>
using PlaneView = ds::PlaneView2D<T>;

using BytePlaneView = PlaneView<const byte>;
using MutableBytePlaneView = PlaneView<byte>;
using FloatPlaneView = PlaneView<float>;
using ConstFloatPlaneView = PlaneView<const float>;

inline BytePlaneView make_byte_plane_view(
  const byte* data,
  int width_bytes,
  int height,
  std::ptrdiff_t stride_bytes
) {
  return ds::make_plane_view(data, width_bytes, height, stride_bytes);
}

inline MutableBytePlaneView make_mutable_byte_plane_view(
  byte* data,
  int width_bytes,
  int height,
  std::ptrdiff_t stride_bytes
) {
  return ds::make_plane_view(data, width_bytes, height, stride_bytes);
}

inline FloatPlaneView make_float_plane_view(
  float* data,
  int width,
  int height,
  std::ptrdiff_t stride_bytes
) {
  return ds::make_plane_view(data, width, height, stride_bytes);
}

inline ConstFloatPlaneView make_float_plane_view(
  const float* data,
  int width,
  int height,
  std::ptrdiff_t stride_bytes
) {
  return ds::make_plane_view(data, width, height, stride_bytes);
}

struct ComplexBlockView {
  std::complex<float>* data {nullptr};
  int outpitch {0};
  int block_height {0};
  int block_count {0};

  std::size_t block_size() const noexcept {
    return static_cast<std::size_t>(outpitch) * static_cast<std::size_t>(block_height);
  }

  std::complex<float>* block_data(int block) const noexcept {
    return data + static_cast<std::size_t>(block) * block_size();
  }

  fftwf_complex* fftw_data() const noexcept {
    return as_fftw(data);
  }

  fftwf_complex* fftw_block_data(int block) const noexcept {
    return as_fftw(block_data(block));
  }

  FloatPlaneView block_float_view(int block) const {
    return make_float_plane_view(
      reinterpret_cast<float*>(block_data(block)),
      outpitch * 2,
      block_height,
      static_cast<std::ptrdiff_t>(outpitch) * 2 * static_cast<std::ptrdiff_t>(sizeof(float))
    );
  }
};

using TemporalComplexBlockViews = std::array<ComplexBlockView, 5>;

inline ComplexBlockView make_complex_block_view(
  std::complex<float>* data,
  int outpitch,
  int block_height,
  int block_count
) noexcept {
  return ComplexBlockView{data, outpitch, block_height, block_count};
}

inline ComplexBlockView make_complex_block_view(
  fftwf_complex* data,
  int outpitch,
  int block_height,
  int block_count
) noexcept {
  return make_complex_block_view(as_complex(data), outpitch, block_height, block_count);
}

} // namespace neo_fft3d
