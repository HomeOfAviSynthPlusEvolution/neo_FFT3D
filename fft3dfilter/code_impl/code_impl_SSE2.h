#ifndef __CODE_IMPL_SSE2_H__
#define __CODE_IMPL_SSE2_H__

#include "code_impl.h"
#include <avs/config.h> // x64
#include <emmintrin.h>

inline __m128 _mm_sign_r(__m128 data) {
  return _mm_xor_ps(data, _mm_set_ps(0.0f, -0.0f, 0.0f, -0.0f));
}
inline __m128 _mm_sign_i(__m128 data) {
  return _mm_xor_ps(data, _mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f));
}
inline __m128 _mm_swap_ri(__m128 data) {
  return _mm_shuffle_ps(data, data, 177); // 10 11 00 01
}

#ifdef _MSC_VER
  #ifndef __clang__
    inline __m128 operator+(const __m128 &a, const __m128 &b) { return _mm_add_ps(a, b); }
    inline __m128 operator-(const __m128 &a, const __m128 &b) { return _mm_sub_ps(a, b); }
    inline __m128 operator*(const __m128 &a, const __m128 &b) { return _mm_mul_ps(a, b); }
    inline __m128 operator/(const __m128 &a, const __m128 &b) { return _mm_div_ps(a, b); }
    inline __m128 &operator+=(__m128 &a, const __m128 &b) { return a = _mm_add_ps(a, b); }
    inline __m128 &operator-=(__m128 &a, const __m128 &b) { return a = _mm_sub_ps(a, b); }
    inline __m128 &operator*=(__m128 &a, const __m128 &b) { return a = _mm_mul_ps(a, b); }
    inline __m128 &operator/=(__m128 &a, const __m128 &b) { return a = _mm_div_ps(a, b); }
  #endif
#endif

struct LambdaFunctionParams {
  // Progress
  int block;
  int pos;
  // Pattern
  __m128 m_pattern2d;
  __m128 m_pattern3d;
  // Wiener
  __m128 m_sigmaSquaredNoiseNormed;
  __m128 m_wsharpen;
  __m128 m_wdehalo;
  // Grid
  __m128 m_gridsample;
  __m128 m_gridcorrection;

  __m128 m_lowlimit;

  inline __m128 psd(const __m128 &data) {
    // [dr0, di0, dr1, di1]
    __m128 _square = data * data;
    __m128 _shuffle = _mm_swap_ri(_square);
    return _square + _shuffle + epsilon; // power spectrum density
  }

  template <bool pattern>
  inline void wiener_factor_3d(__m128 &data) {
    __m128 _psd = psd(data);
    __m128 _sigma = pattern ? m_pattern3d : m_sigmaSquaredNoiseNormed;
    __m128 _wiener_factor = (_psd - _sigma) / _psd;
    _wiener_factor = _mm_max_ps(_wiener_factor, m_lowlimit); // limited Wiener filter

    data *= _wiener_factor;
  }

  private:
    const __m128 epsilon = _mm_set1_ps(1e-15f);
};

template<typename ... T, typename Func>
inline void loop_wrapper_SSE2(fftwf_complex** in, fftwf_complex* &out, SharedFunctionParams sfp, Func f) {
  LambdaFunctionParams lfp;

  lfp.m_lowlimit = _mm_set1_ps((sfp.beta - 1) / sfp.beta);
  lfp.m_sigmaSquaredNoiseNormed = _mm_set1_ps(sfp.sigmaSquaredNoiseNormed);
  int itemsperblock = sfp.bh * sfp.outpitch;
  const int step = 2;

  for (lfp.block = 0; lfp.block < sfp.howmanyblocks; lfp.block++)
  {
    // Pattern
    float* pattern2d = sfp.pattern2d;
    float* pattern3d = sfp.pattern3d;
    // Wiener
    float* wsharpen = sfp.wsharpen;
    float* wdehalo = sfp.wdehalo;
    // Grid
    fftwf_complex* gridsample = sfp.gridsample;
    __m128 gridfraction = _mm_set1_ps(sfp.degrid * in[2][0][0] / gridsample[0][0]);

    for (lfp.pos = 0; lfp.pos < itemsperblock; lfp.pos += 2)
    {
      // Pattern
      lfp.m_pattern2d = _mm_loadu_ps(pattern2d);
      lfp.m_pattern2d = _mm_unpacklo_ps(lfp.m_pattern2d, lfp.m_pattern2d);
      lfp.m_pattern3d = _mm_loadu_ps(pattern3d);
      lfp.m_pattern3d = _mm_unpacklo_ps(lfp.m_pattern3d, lfp.m_pattern3d);
      // Wiener
      lfp.m_wsharpen = _mm_loadu_ps(wsharpen);
      lfp.m_wsharpen = _mm_unpacklo_ps(lfp.m_wsharpen, lfp.m_wsharpen);
      lfp.m_wdehalo = _mm_loadu_ps(wdehalo);
      lfp.m_wdehalo = _mm_unpacklo_ps(lfp.m_wdehalo, lfp.m_wdehalo);
      // Grid
      lfp.m_gridsample = _mm_load_ps((const float*)gridsample);
      lfp.m_gridcorrection = _mm_mul_ps(gridfraction, lfp.m_gridsample);

      f(lfp);

      // Data
      in[0] += step;
      in[1] += step;
      in[2] += step;
      in[3] += step;
      in[4] += step;
      out += step;
      // Pattern
      pattern2d += step;
      pattern3d += step;
      // Wiener
      wsharpen += step;
      wdehalo += step;
      // Grid
      gridsample += step;
    }
  }
}

#endif
