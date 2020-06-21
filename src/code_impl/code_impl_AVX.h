/*
 * Copyright 2020 Xinyue Lu
 *
 * AVX SIMD wrapper.
 *
 */

#ifndef __CODE_IMPL_AVX_H__
#define __CODE_IMPL_AVX_H__

#include "code_impl.h"
#include <immintrin.h>

inline __m256 _mm256_sign_r(__m256 data) {
  return _mm256_xor_ps(data, _mm256_set_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f));
}
inline __m256 _mm256_sign_i(__m256 data) {
  return _mm256_xor_ps(data, _mm256_set_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f));
}
inline __m256 _mm256_swap_ri(__m256 data) {
  return _mm256_shuffle_ps(data, data, 177); // 10 11 00 01
}
inline __m256 _mm256_loadu_2ps(const float* mem) {
  // TODO: _mm_load_si128 + _mm256_cvtepu32_epi64
  __m256 src1 = _mm256_loadu_ps(mem);
  __m256 src2 = _mm256_loadu_ps(mem-2);
  __m256 blended = _mm256_blend_ps(src1, src2, 0xF0);
  return _mm256_unpacklo_ps(blended, blended);
}

#ifdef _MSC_VER
  #ifndef __clang__
    inline __m256 operator+(const __m256 &a, const __m256 &b) { return _mm256_add_ps(a, b); }
    inline __m256 operator-(const __m256 &a, const __m256 &b) { return _mm256_sub_ps(a, b); }
    inline __m256 operator*(const __m256 &a, const __m256 &b) { return _mm256_mul_ps(a, b); }
    inline __m256 operator/(const __m256 &a, const __m256 &b) { return _mm256_div_ps(a, b); }
    inline __m256 &operator+=(__m256 &a, const __m256 &b) { return a = _mm256_add_ps(a, b); }
    inline __m256 &operator-=(__m256 &a, const __m256 &b) { return a = _mm256_sub_ps(a, b); }
    inline __m256 &operator*=(__m256 &a, const __m256 &b) { return a = _mm256_mul_ps(a, b); }
    inline __m256 &operator/=(__m256 &a, const __m256 &b) { return a = _mm256_div_ps(a, b); }
  #endif
#endif

struct LambdaFunctionParams {
  // Progress
  int block;
  int pos;
  // Pattern
  __m256 m_pattern2d;
  __m256 m_pattern3d;
  // Wiener
  __m256 m_sigmaSquaredNoiseNormed;
  __m256 m_wsharpen;
  __m256 m_wdehalo;
  // Grid
  __m256 m_gridsample;
  __m256 m_gridcorrection;
  // Kalman
  __m256 m_sigmaSquaredNoiseNormed2D;
  fftwf_complex *covar;
  fftwf_complex *covarProcess;
  fftwf_complex **in;
  fftwf_complex *out;

  __m256 m_lowlimit;

  inline __m256 psd(const __m256 &data) {
    // [dr0, di0, dr1, di1]
    const __m256 epsilon = _mm256_set1_ps(1.0e-15f);
    __m256 _square = data * data;
    __m256 _shuffle = _mm256_swap_ri(_square);
    return _square + _shuffle + epsilon; // power spectrum density
  }

  template <bool pattern>
  inline void wiener_factor_3d(__m256 &data) {
    __m256 _psd = psd(data);
    __m256 _sigma = pattern ? m_pattern3d : m_sigmaSquaredNoiseNormed;
    __m256 _wiener_factor = (_psd - _sigma) / _psd;
    _wiener_factor = _mm256_max_ps(_wiener_factor, m_lowlimit); // limited Wiener filter

    data *= _wiener_factor;
  }
};

template<typename Expo, typename Func>
inline void loop_wrapper_AVX(Expo &&expo, fftwf_complex** in, fftwf_complex* &out, SharedFunctionParams sfp, Func f) {
  int itemsperblock = sfp.bh * sfp.outpitch;
  const int step = 4;

#ifdef ENABLE_PAR
  constexpr auto batch_count = 4;
  const int batch_size = (sfp.howmanyblocks - 1) / batch_count + 1;

  std::for_each_n(expo, reinterpret_cast<char*>(0), batch_count, [&](char&idx)
  {
    int i = static_cast<int>(reinterpret_cast<intptr_t>(&idx));
#else
  constexpr auto batch_count = 1;
  const int batch_size = sfp.howmanyblocks;
  int i = 0;
#endif
    LambdaFunctionParams lfp;
    lfp.m_lowlimit = _mm256_set1_ps((sfp.beta - 1) / sfp.beta);
    lfp.m_sigmaSquaredNoiseNormed = _mm256_set1_ps(sfp.sigmaSquaredNoiseNormed);
    lfp.m_sigmaSquaredNoiseNormed2D = _mm256_set1_ps(sfp.sigmaSquaredNoiseNormed2D);
    auto block_start = i * batch_size;
    auto block_end = MIN(block_start + batch_size, sfp.howmanyblocks);
    auto offset = block_start * itemsperblock;

    // IO
    fftwf_complex* local_in[5];
    local_in[0] = in[0] + offset;
    local_in[1] = in[1] + offset;
    local_in[2] = in[2] + offset;
    local_in[3] = in[3] + offset;
    local_in[4] = in[4] + offset;
    lfp.in = local_in;
    lfp.out = out + offset;

    // Kalman
    lfp.covar = sfp.covar + offset;
    lfp.covarProcess = sfp.covarProcess + offset;

    for (lfp.block = block_start; lfp.block < block_end; lfp.block++)
    {
      // Pattern
      float* pattern2d = sfp.pattern2d;
      float* pattern3d = sfp.pattern3d;
      // Wiener
      float* wsharpen = sfp.wsharpen;
      float* wdehalo = sfp.wdehalo;
      // Grid
      fftwf_complex* gridsample = sfp.gridsample;
      __m256 gridfraction = _mm256_set1_ps(sfp.degrid * local_in[2][0][0] / gridsample[0][0]);

      for (lfp.pos = 0; lfp.pos < itemsperblock; lfp.pos += step)
      {
        // Pattern
        lfp.m_pattern2d = _mm256_loadu_2ps(pattern2d);
        lfp.m_pattern3d = _mm256_loadu_2ps(pattern3d);
        // Wiener
        lfp.m_wsharpen = _mm256_loadu_2ps(wsharpen);
        lfp.m_wdehalo = _mm256_loadu_2ps(wdehalo);
        // Grid
        lfp.m_gridsample = _mm256_load_ps((const float*)gridsample);
        lfp.m_gridcorrection = _mm256_mul_ps(gridfraction, lfp.m_gridsample);

        f(lfp);

        // Data
        lfp.in[0] += step;
        lfp.in[1] += step;
        lfp.in[2] += step;
        lfp.in[3] += step;
        lfp.in[4] += step;
        lfp.out += step;
        // Pattern
        pattern2d += step;
        pattern3d += step;
        // Wiener
        wsharpen += step;
        wdehalo += step;
        // Grid
        gridsample += step;
        // Kalman
        lfp.covar += step;
        lfp.covarProcess += step;
      }
    }
#ifdef ENABLE_PAR
  });
#endif
}

#endif
