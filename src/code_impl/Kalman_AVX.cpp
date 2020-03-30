/*
 * Copyright 2020 Xinyue Lu
 *
 * Kalman filter implementation, AVX SIMD code.
 *
 */

#include "code_impl_AVX.h"

// Profiling Infomation
// Hardware: Ryzen 3600
// Software: AVSMeter64
// Source: 1280x720 1740 frames YUV420P8


// Kalman Profiling
// 11,109 ms -> 4,153 ms -> 3,478ms
template <bool pattern>
void Kalman_AVX(fftwf_complex *outcur, fftwf_complex *outLast, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, outLast, outcur, 0, 0};
  loop_wrapper_AVX(dummy, outLast, sfp,
    [&](LambdaFunctionParams lfp) {
      const __m256 m_one = _mm256_set1_ps(1.0f);

      __m256 sigma;

      __m256 cur = _mm256_load_ps((const float*)(dummy[2]));
      __m256 prev = _mm256_load_ps((const float*)(dummy[1]));
      __m256 m_covar = _mm256_load_ps((const float*)lfp.covar);
      __m256 m_covarProcess = _mm256_load_ps((const float*)lfp.covarProcess);

      if (pattern) {
        // Prevent bad blocks, maybe incorrect -- by XL
        sigma = _mm256_max_ps(lfp.m_pattern2d, lfp.epsilon);
      }
      else
        sigma = lfp.m_sigmaSquaredNoiseNormed2D;

      __m256 motion = (cur - prev) * (cur - prev);
      __m256 motion_threshold = sigma * _mm256_set1_ps(sfp.kratio2);

      __m256 is_motion = _mm256_cmp_ps(motion, motion_threshold, _CMP_GT_OQ);
      is_motion = _mm256_or_ps(_mm256_movehdup_ps(is_motion), _mm256_moveldup_ps(is_motion));
      int motion_mask = _mm256_movemask_ps(is_motion);

      if (motion_mask == 255) {
        _mm256_store_ps((float*)lfp.covar, sigma);
        _mm256_store_ps((float*)lfp.covarProcess, sigma);
        _mm256_store_ps((float*)outLast, cur);
        return;
      }

      __m256 sum = m_covar + m_covarProcess;
      __m256 gain = sum / (sum + sigma);

      __m256 out_covarProcess = gain * gain * sigma;
      __m256 out_covar = (m_one - gain) * sum;
      __m256 out_value = gain * cur + (m_one - gain) * prev;
      if (motion_mask > 0) {
        out_covar = _mm256_blendv_ps(out_covar, sigma, is_motion);
        out_covarProcess = _mm256_blendv_ps(out_covarProcess, sigma, is_motion);
        out_value = _mm256_blendv_ps(out_value, cur, is_motion);
      }

      _mm256_store_ps((float*)lfp.covar, out_covar);
      _mm256_store_ps((float*)lfp.covarProcess, out_covarProcess);
      _mm256_store_ps((float*)outLast, out_value);
    }
  );
}

template void Kalman_AVX<true>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
template void Kalman_AVX<false>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
