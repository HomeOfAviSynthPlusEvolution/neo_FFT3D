/*
 * Copyright 2020 Xinyue Lu
 *
 * Sharpener implementation, SSE SIMD code.
 *
 */

#include "code_impl_SSE2.h"

// Profiling Infomation
// Hardware: Ryzen 3600
// Software: AVSMeter64
// Source: 1280x720 1740 frames YUV420P8


// Sharpen Profiling
// 13,958 ms -> 3,484 ms
template <bool degrid, bool sharpen, bool dehalo>
static inline void Sharpen_SSE2_impl(fftwf_complex *out, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, 0, out, 0, 0};
  loop_wrapper_SSE2(std::execution::par_unseq, dummy, out, sfp,
    [&sfp](LambdaFunctionParams lfp) {
      __m128 gridcorrection;

      __m128 cur = _mm_load_ps((const float*)lfp.in[2]);

      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection;
        cur -= gridcorrection;
      }
      __m128 psd = lfp.psd(cur);

      const __m128 m_sigma_min = _mm_set1_ps(sfp.sigmaSquaredSharpenMinNormed);
      const __m128 m_sigma_max = _mm_set1_ps(sfp.sigmaSquaredSharpenMaxNormed);
      const __m128 m_sharpen = _mm_set1_ps(sfp.sharpen);
      const __m128 m_dehalo = _mm_set1_ps(sfp.dehalo);
      const __m128 m_ht2n = _mm_set1_ps(sfp.ht2n);
      const __m128 m_one = _mm_set1_ps(1.0f);

      __m128 s_fact1 = m_sharpen * lfp.m_wsharpen;
      __m128 s_fact2 = psd * m_sigma_max / ((psd + m_sigma_min) * (psd + m_sigma_max));
      __m128 s_fact = m_one + s_fact1 * _mm_sqrt_ps(s_fact2);

      __m128 d_fact1 = psd + m_ht2n;
      __m128 d_fact = d_fact1 / (d_fact1 + m_dehalo * lfp.m_wdehalo * psd);

      __m128 factor;

      if constexpr (sharpen && !dehalo)
        factor = s_fact;
      else if constexpr (!sharpen && dehalo)
        factor = d_fact;
      else if constexpr (sharpen && dehalo)
        factor = s_fact * d_fact;

      __m128 result = cur * factor;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      _mm_store_ps((float*)lfp.out, result);
    }
  );
}

template <bool degrid>
void Sharpen_SSE2(fftwf_complex *out, SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_SSE2_impl<degrid, true, false>(out, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_SSE2_impl<degrid, false, true>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_SSE2_impl<degrid, true, true>(out, sfp);
}

template void Sharpen_SSE2<true>(fftwf_complex *, SharedFunctionParams);
template void Sharpen_SSE2<false>(fftwf_complex *, SharedFunctionParams);
