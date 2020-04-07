/*
 * Copyright 2020 Xinyue Lu
 *
 * Sharpener implementation, AVX SIMD code.
 *
 */

#include "code_impl_AVX.h"

// Profiling Infomation
// Hardware: Ryzen 3600
// Software: AVSMeter64
// Source: 1280x720 1740 frames YUV420P8


// Sharpen Profiling
// 13,958 ms -> 3,484 ms -> 1,753ms
template <bool degrid, bool sharpen, bool dehalo>
static inline void Sharpen_AVX_impl(fftwf_complex *out, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, 0, out, 0, 0};
  loop_wrapper_AVX(dummy, out, sfp,
    [&](LambdaFunctionParams lfp) {
      __m256 gridcorrection;

      __m256 cur = _mm256_load_ps((const float*)out);

      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection;
        cur -= gridcorrection;
      }
      __m256 psd = lfp.psd(cur);

      const __m256 m_sigma_min = _mm256_set1_ps(sfp.sigmaSquaredSharpenMinNormed);
      const __m256 m_sigma_max = _mm256_set1_ps(sfp.sigmaSquaredSharpenMaxNormed);
      const __m256 m_sharpen = _mm256_set1_ps(sfp.sharpen);
      const __m256 m_dehalo = _mm256_set1_ps(sfp.dehalo);
      const __m256 m_ht2n = _mm256_set1_ps(sfp.ht2n);
      const __m256 m_one = _mm256_set1_ps(1.0f);

      __m256 s_fact1 = m_sharpen * lfp.m_wsharpen;
      __m256 s_fact2 = psd * m_sigma_max / ((psd + m_sigma_min) * (psd + m_sigma_max));
      __m256 s_fact = m_one + s_fact1 * _mm256_sqrt_ps(s_fact2);

      __m256 d_fact1 = psd + m_ht2n;
      __m256 d_fact = d_fact1 / (d_fact1 + m_dehalo * lfp.m_wdehalo * psd);

      __m256 factor;

      if constexpr (sharpen && !dehalo)
        factor = s_fact;
      else if constexpr (!sharpen && dehalo)
        factor = d_fact;
      else if constexpr (sharpen && dehalo)
        factor = s_fact * d_fact;

      __m256 result = cur * factor;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      _mm256_store_ps((float*)out, result);
    }
  );
}

template <bool degrid>
void Sharpen_AVX(fftwf_complex *out, SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_AVX_impl<degrid, true, false>(out, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_AVX_impl<degrid, false, true>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_AVX_impl<degrid, true, true>(out, sfp);
}

template void Sharpen_AVX<true>(fftwf_complex *, SharedFunctionParams);
template void Sharpen_AVX<false>(fftwf_complex *, SharedFunctionParams);
