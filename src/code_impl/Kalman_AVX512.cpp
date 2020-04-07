/*
 * Copyright 2020 Xinyue Lu
 *
 * Kalman filter implementation, AVX SIMD code.
 *
 */

#include "code_impl_AVX512.h"

template <bool pattern>
void Kalman_AVX512(fftwf_complex *outcur, fftwf_complex *outLast, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, outLast, outcur, 0, 0};
  loop_wrapper_AVX512(dummy, outLast, sfp,
    [&](LambdaFunctionParams lfp) {
      const __m512 m_one = _mm512_set1_ps(1.0f);

      __m512 sigma;

      __m512 cur = _mm512_load_ps((const float*)(dummy[2]));
      __m512 prev = _mm512_load_ps((const float*)(dummy[1]));
      __m512 m_covar = _mm512_load_ps((const float*)lfp.covar);
      __m512 m_covarProcess = _mm512_load_ps((const float*)lfp.covarProcess);

      if constexpr (pattern) {
        // Prevent bad blocks, maybe incorrect -- by XL
        sigma = _mm512_max_ps(lfp.m_pattern2d, lfp.epsilon);
      }
      else
        sigma = lfp.m_sigmaSquaredNoiseNormed2D;

      __m512 motion = (cur - prev) * (cur - prev);
      __m512 motion_threshold = sigma * _mm512_set1_ps(sfp.kratio2);

      __mmask16 is_motion1 = _mm512_cmp_ps_mask(_mm512_movehdup_ps(motion), motion_threshold, _CMP_GT_OQ);
      __mmask16 is_motion2 = _mm512_cmp_ps_mask(_mm512_moveldup_ps(motion), motion_threshold, _CMP_GT_OQ);
      __mmask16 is_motion = _kor_mask16(is_motion1, is_motion2);
      unsigned int motion_mask = _cvtmask16_u32(is_motion);

      if (motion_mask == 0xFFFF) {
        _mm512_store_ps((float*)lfp.covar, sigma);
        _mm512_store_ps((float*)lfp.covarProcess, sigma);
        _mm512_store_ps((float*)outLast, cur);
        return;
      }

      __m512 sum = m_covar + m_covarProcess;
      __m512 gain = sum / (sum + sigma);

      __m512 out_covarProcess = gain * gain * sigma;
      __m512 out_covar = (m_one - gain) * sum;
      __m512 out_value = gain * cur + (m_one - gain) * prev;
      if (motion_mask > 0) {
        out_covar = _mm512_mask_blend_ps(is_motion, out_covar, sigma);
        out_covarProcess = _mm512_mask_blend_ps(is_motion, out_covarProcess, sigma);
        out_value = _mm512_mask_blend_ps(is_motion, out_value, cur);
      }

      _mm512_store_ps((float*)lfp.covar, out_covar);
      _mm512_store_ps((float*)lfp.covarProcess, out_covarProcess);
      _mm512_store_ps((float*)outLast, out_value);
    }
  );
}

template void Kalman_AVX512<true>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
template void Kalman_AVX512<false>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
