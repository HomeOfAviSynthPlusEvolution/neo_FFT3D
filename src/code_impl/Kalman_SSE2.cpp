/*
 * Copyright 2020 Xinyue Lu
 *
 * Kalman filter implementation, SSE SIMD code.
 *
 */

#include "code_impl_SSE2.h"

// Profiling Infomation
// Hardware: Ryzen 3600
// Software: AVSMeter64
// Source: 1280x720 1740 frames YUV420P8


// Kalman Profiling
// 11,109 ms -> 4,153 ms
template <bool pattern>
void Kalman_SSE2(fftwf_complex *outcur, fftwf_complex *outLast, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, outLast, outcur, 0, 0};
  loop_wrapper_SSE2(dummy, outLast, sfp,
    [&](LambdaFunctionParams lfp) {
      const __m128 m_one = _mm_set1_ps(1.0f);

      __m128 sigma;

      __m128 cur = _mm_load_ps((const float*)(dummy[2]));
      __m128 prev = _mm_load_ps((const float*)(dummy[1]));
      __m128 m_covar = _mm_load_ps((const float*)lfp.covar);
      __m128 m_covarProcess = _mm_load_ps((const float*)lfp.covarProcess);

      if constexpr (pattern) {
        // Prevent bad blocks, maybe incorrect -- by XL
        sigma = _mm_max_ps(lfp.m_pattern2d, lfp.epsilon);
      }
      else
        sigma = lfp.m_sigmaSquaredNoiseNormed2D;

      __m128 motion = (cur - prev) * (cur - prev);
      __m128 motion_threshold = sigma * _mm_set1_ps(sfp.kratio2);

      __m128 is_motion = _mm_cmpgt_ps(motion, motion_threshold);
      int motion_mask = _mm_movemask_ps(is_motion);
      motion_mask = (motion_mask & 3 ? 1 : 0) | (motion_mask & 12 ? 2 : 0);

      if (motion_mask == 3) {
        _mm_store_ps((float*)lfp.covar, sigma);
        _mm_store_ps((float*)lfp.covarProcess, sigma);
        _mm_store_ps((float*)outLast, cur);
        return;
      }

      __m128 sum = m_covar + m_covarProcess;
      __m128 gain = sum / (sum + sigma);

      __m128 out_covarProcess = gain * gain * sigma;
      __m128 out_covar = (m_one - gain) * sum;
      __m128 out_value = gain * cur + (m_one - gain) * prev;
      _mm_store_ps((float*)lfp.covar, out_covar);
      _mm_store_ps((float*)lfp.covarProcess, out_covarProcess);
      _mm_store_ps((float*)outLast, out_value);
      if (motion_mask == 0)
        return;

      fftwf_complex sigma_array[2];
      _mm_store_ps((float*)sigma_array, sigma);
      
      if (motion_mask == 2) {
        // NOT using slow _mm_maskmoveu_si128
        // Reset high element
        lfp.covar[1][0] = sigma_array[1][0];
        lfp.covar[1][1] = sigma_array[1][1];
        lfp.covarProcess[1][0] = sigma_array[1][0];
        lfp.covarProcess[1][1] = sigma_array[1][1];
        outLast[1][0] = dummy[2][1][0];
        outLast[1][1] = dummy[2][1][1];
      }
      else {
        // NOT using slow _mm_maskmoveu_si128
        // Reset low element
        lfp.covar[0][0] = sigma_array[0][0];
        lfp.covar[0][1] = sigma_array[0][1];
        lfp.covarProcess[0][0] = sigma_array[0][0];
        lfp.covarProcess[0][1] = sigma_array[0][1];
        outLast[0][0] = dummy[2][0][0];
        outLast[0][1] = dummy[2][0][1];
      }
    }
  );
}

template void Kalman_SSE2<true>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
template void Kalman_SSE2<false>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
