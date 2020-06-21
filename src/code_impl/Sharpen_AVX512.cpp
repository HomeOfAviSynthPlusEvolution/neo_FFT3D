/*
 * Copyright 2020 Xinyue Lu
 *
 * Sharpener implementation, AVX512 SIMD code.
 *
 */

#include "code_impl_AVX512.h"

template <bool degrid, bool sharpen, bool dehalo>
static inline void Sharpen_AVX512_impl(fftwf_complex *out, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, 0, out, 0, 0};
  loop_wrapper_AVX512(PAR_POLICY, dummy, out, sfp,
    [&sfp](LambdaFunctionParams lfp) {
      __m512 gridcorrection;

      __m512 cur = _mm512_load_ps((const float*)lfp.in[2]);

      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection;
        cur -= gridcorrection;
      }
      __m512 psd = lfp.psd(cur);

      const __m512 m_sigma_min = _mm512_set1_ps(sfp.sigmaSquaredSharpenMinNormed);
      const __m512 m_sigma_max = _mm512_set1_ps(sfp.sigmaSquaredSharpenMaxNormed);
      const __m512 m_sharpen = _mm512_set1_ps(sfp.sharpen);
      const __m512 m_dehalo = _mm512_set1_ps(sfp.dehalo);
      const __m512 m_ht2n = _mm512_set1_ps(sfp.ht2n);
      const __m512 m_one = _mm512_set1_ps(1.0f);

      __m512 s_fact1 = m_sharpen * lfp.m_wsharpen;
      __m512 s_fact2 = psd * m_sigma_max / ((psd + m_sigma_min) * (psd + m_sigma_max));
      __m512 s_fact = m_one + s_fact1 * _mm512_sqrt_ps(s_fact2);

      __m512 d_fact1 = psd + m_ht2n;
      __m512 d_fact = d_fact1 / (d_fact1 + m_dehalo * lfp.m_wdehalo * psd);

      __m512 factor;

      if constexpr (sharpen && !dehalo)
        factor = s_fact;
      else if constexpr (!sharpen && dehalo)
        factor = d_fact;
      else if constexpr (sharpen && dehalo)
        factor = s_fact * d_fact;

      __m512 result = cur * factor;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      _mm512_store_ps((float*)lfp.out, result);
    }
  );
}

template <bool degrid>
void Sharpen_AVX512(fftwf_complex *out, SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_AVX512_impl<degrid, true, false>(out, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_AVX512_impl<degrid, false, true>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_AVX512_impl<degrid, true, true>(out, sfp);
}

template void Sharpen_AVX512<true>(fftwf_complex *, SharedFunctionParams);
template void Sharpen_AVX512<false>(fftwf_complex *, SharedFunctionParams);
