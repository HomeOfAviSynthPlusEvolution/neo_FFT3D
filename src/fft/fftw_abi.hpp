#pragma once

#include "common/aligned_vector.hpp"

#include <complex>
#include <memory>

using fftwf_complex = float[2];
struct fftwf_plan_s;
using fftwf_plan = fftwf_plan_s*;

inline constexpr unsigned FFTW_MEASURE = 0U;
inline constexpr unsigned FFTW_ESTIMATE = 1U << 6;

static_assert(sizeof(std::complex<float>) == sizeof(fftwf_complex) &&
                alignof(std::complex<float>) == alignof(fftwf_complex),
              "std::complex<float> and fftwf_complex must have the identical ABI size and alignment.");

inline fftwf_complex* as_fftw(std::complex<float>* ptr) {
  return reinterpret_cast<fftwf_complex*>(ptr);
}

inline fftwf_complex* as_fftw(const std::shared_ptr<AlignedVector<std::complex<float>>>& sp) {
  return sp ? reinterpret_cast<fftwf_complex*>(sp->data()) : nullptr;
}

inline std::complex<float>* as_complex(fftwf_complex* ptr) {
  return reinterpret_cast<std::complex<float>*>(ptr);
}
