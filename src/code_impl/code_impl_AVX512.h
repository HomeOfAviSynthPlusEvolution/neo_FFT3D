/*
 * Copyright 2020 Xinyue Lu
 *
 * AVX512 SIMD wrapper.
 *
 */

#ifndef __CODE_IMPL_AVX512_H__
#define __CODE_IMPL_AVX512_H__

#include "code_impl.h"
#include <immintrin.h>
#include <execution>

inline __m512 _mm512_sign_r(__m512 data) {
  return _mm512_castsi512_ps(_mm512_xor_epi32(_mm512_castps_si512(data), _mm512_set1_epi64(0x80000000LL)));
}
inline __m512 _mm512_sign_i(__m512 data) {
  return _mm512_castsi512_ps(_mm512_xor_epi32(_mm512_castps_si512(data), _mm512_set1_epi64(0x8000LL)));
}
inline __m512 _mm512_swap_ri(__m512 data) {
  return _mm512_shuffle_ps(data, data, 177); // 10 11 00 01
}
inline __m512 _mm512_loadu_4ps(const float* mem) {
  __m512 src1 = _mm512_loadu_ps(mem);
  __m512i expander_index = _mm512_set_epi32(0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7);
  return _mm512_permutexvar_ps(expander_index, src1);
}

#ifdef _MSC_VER
  #ifndef __clang__
    inline __m512 operator+(const __m512 &a, const __m512 &b) { return _mm512_add_ps(a, b); }
    inline __m512 operator-(const __m512 &a, const __m512 &b) { return _mm512_sub_ps(a, b); }
    inline __m512 operator*(const __m512 &a, const __m512 &b) { return _mm512_mul_ps(a, b); }
    inline __m512 operator/(const __m512 &a, const __m512 &b) { return _mm512_div_ps(a, b); }
    inline __m512 &operator+=(__m512 &a, const __m512 &b) { return a = _mm512_add_ps(a, b); }
    inline __m512 &operator-=(__m512 &a, const __m512 &b) { return a = _mm512_sub_ps(a, b); }
    inline __m512 &operator*=(__m512 &a, const __m512 &b) { return a = _mm512_mul_ps(a, b); }
    inline __m512 &operator/=(__m512 &a, const __m512 &b) { return a = _mm512_div_ps(a, b); }
  #endif
#endif

struct LambdaFunctionParams {
  // Progress
  int block;
  int pos;
  // Pattern
  __m512 m_pattern2d;
  __m512 m_pattern3d;
  // Wiener
  __m512 m_sigmaSquaredNoiseNormed;
  __m512 m_wsharpen;
  __m512 m_wdehalo;
  // Grid
  __m512 m_gridsample;
  __m512 m_gridcorrection;
  // Kalman
  __m512 m_sigmaSquaredNoiseNormed2D;
  fftwf_complex *covar;
  fftwf_complex *covarProcess;
  fftwf_complex **in;
  fftwf_complex *out;

  __m512 m_lowlimit;

  inline __m512 psd(const __m512 &data) {
    // [dr0, di0, dr1, di1]
    const __m512 epsilon = _mm512_set1_ps(1.0e-15f);
    __m512 _square = data * data;
    __m512 _shuffle = _mm512_swap_ri(_square);
    return _square + _shuffle + epsilon; // power spectrum density
  }

  template <bool pattern>
  inline void wiener_factor_3d(__m512 &data) {
    __m512 _psd = psd(data);
    __m512 _sigma = pattern ? m_pattern3d : m_sigmaSquaredNoiseNormed;
    __m512 _wiener_factor = (_psd - _sigma) / _psd;
    _wiener_factor = _mm512_max_ps(_wiener_factor, m_lowlimit); // limited Wiener filter

    data *= _wiener_factor;
  }
};

template<typename Expo, typename Func>
inline void loop_wrapper_AVX512(Expo &&expo, fftwf_complex** in, fftwf_complex* &out, SharedFunctionParams sfp, Func f) {
  int itemsperblock = sfp.bh * sfp.outpitch;
  const int step = 8;
  constexpr auto batch_count = 4;
  const int batch_size = (sfp.howmanyblocks - 1) / batch_count + 1;

  std::for_each_n(expo, reinterpret_cast<char*>(0), batch_count, [&](char&idx)
  {
    int i = static_cast<int>(reinterpret_cast<intptr_t>(&idx));
    LambdaFunctionParams lfp;
    lfp.m_lowlimit = _mm512_set1_ps((sfp.beta - 1) / sfp.beta);
    lfp.m_sigmaSquaredNoiseNormed = _mm512_set1_ps(sfp.sigmaSquaredNoiseNormed);
    lfp.m_sigmaSquaredNoiseNormed2D = _mm512_set1_ps(sfp.sigmaSquaredNoiseNormed2D);
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
      __m512 gridfraction = _mm512_set1_ps(sfp.degrid * in[2][0][0] / gridsample[0][0]);

      for (lfp.pos = 0; lfp.pos < itemsperblock; lfp.pos += step)
      {
        // Pattern
        lfp.m_pattern2d = _mm512_loadu_4ps(pattern2d);
        lfp.m_pattern3d = _mm512_loadu_4ps(pattern3d);
        // Wiener
        lfp.m_wsharpen = _mm512_loadu_4ps(wsharpen);
        lfp.m_wdehalo = _mm512_loadu_4ps(wdehalo);
        // Grid
        lfp.m_gridsample = _mm512_load_ps((const float*)gridsample);
        lfp.m_gridcorrection = _mm512_mul_ps(gridfraction, lfp.m_gridsample);

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
  });
}

#endif
