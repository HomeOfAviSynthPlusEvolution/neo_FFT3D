/*
 * Copyright 2020 Xinyue Lu
 *
 * 2D and 3D filter implementation, SSE SIMD code.
 *
 */

#include "code_impl_SSE2.h"

// Profiling Infomation
// Hardware: Ryzen 3600
// Software: AVSMeter64
// Source: 1280x720 1740 frames YUV420P8


// 2D SSE2 Profiling
// Profiling time: 15,909ms -> 3,517ms
template <bool pattern, bool degrid, bool sharpen, bool dehalo>
static inline void Apply2D_SSE2_impl(fftwf_complex *out, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, 0, out, 0, 0};
  loop_wrapper_SSE2(PAR_POLICY, dummy, out, sfp,
    [&sfp](LambdaFunctionParams lfp) {
      (void)sfp;
      __m128 gridcorrection;

      __m128 cur = _mm_load_ps((const float*)lfp.in[2]);

      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection;
        cur -= gridcorrection;
      }
      __m128 psd = lfp.psd(cur);
      __m128 sigma = pattern ? lfp.m_pattern2d : lfp.m_sigmaSquaredNoiseNormed;
      __m128 factor = (psd - sigma) / psd;
      factor = _mm_max_ps(factor, lfp.m_lowlimit); // limited Wiener filter

      if constexpr (!pattern) {
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

        if (sharpen && !dehalo)
          factor *= s_fact;
        else if (!sharpen && dehalo)
          factor *= d_fact;
        else if (sharpen && dehalo)
          factor *= s_fact * d_fact;
      }

      __m128 result = cur * factor;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      _mm_store_ps((float*)lfp.out, result);
    }
  );
}

template <bool pattern, bool degrid>
void Apply2D_SSE2(fftwf_complex *out, SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    Apply2D_SSE2_impl<pattern, degrid, true, false>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Apply2D_SSE2_impl<pattern, degrid, true, false>(out, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Apply2D_SSE2_impl<pattern, degrid, false, true>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Apply2D_SSE2_impl<pattern, degrid, true, true>(out, sfp);
}

// 3D2 SSE2 Profiling
// Profiling time: 13,830ms -> 3,480ms
template <bool pattern, bool degrid>
void Apply3D2_SSE2(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  loop_wrapper_SSE2(PAR_POLICY, in, out, sfp,
    [](LambdaFunctionParams lfp) {
      __m128 gridcorrection;
      constexpr float scale = 2.0f;

      __m128 cur = _mm_load_ps((const float*)lfp.in[2]);
      __m128 prev = _mm_load_ps((const float*)lfp.in[1]);

      __m128 f3d0 = cur + prev;
      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection * _mm_set1_ps(scale);
        f3d0 -= gridcorrection;
      }
      __m128 f3d1 = cur - prev;

      lfp.wiener_factor_3d<pattern>(f3d0);
      lfp.wiener_factor_3d<pattern>(f3d1);

      __m128 result = f3d0 + f3d1;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      result *= _mm_set1_ps(1 / scale);
      _mm_store_ps((float*)lfp.out, result);
    }
  );
}

// 3D3 SSE2 Profiling
// Profiling time: 20,865ms -> 8,779ms
template <bool pattern, bool degrid>
void Apply3D3_SSE2(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  loop_wrapper_SSE2(PAR_POLICY, in, out, sfp,
    [](LambdaFunctionParams lfp) {
      __m128 gridcorrection;
      constexpr float scale = 3.0f;
      const __m128 sin120 = _mm_set1_ps(0.86602540378443864676372317075294f);//sqrtf(3.0f)*0.5f;
      const __m128 m_0_5 = _mm_set1_ps(0.5f);

      __m128 cur = _mm_load_ps((const float*)lfp.in[2]);
      __m128 prev = _mm_load_ps((const float*)lfp.in[1]);
      __m128 next = _mm_load_ps((const float*)lfp.in[3]);

      __m128 pn = prev + next;
      __m128 fc = cur + pn;
      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection * _mm_set1_ps(scale);
        fc -= gridcorrection;
      }
      __m128 diff = (prev - next) * sin120;
      __m128 d_ir = _mm_swap_ri(_mm_sign_r(diff));

      __m128 tempcur = cur - pn * m_0_5;
      __m128 fp = tempcur + d_ir;
      __m128 fn = tempcur - d_ir;

      lfp.wiener_factor_3d<pattern>(fc);
      lfp.wiener_factor_3d<pattern>(fp);
      lfp.wiener_factor_3d<pattern>(fn);

      __m128 result = fc + fp + fn;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      result *= _mm_set1_ps(1 / scale);
      _mm_store_ps((float*)lfp.out, result);
    }
  );
}

// 3D4 SSE2 Profiling
// Profiling time: 41,681ms -> 10,441ms
template <bool pattern, bool degrid>
void Apply3D4_SSE2(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  loop_wrapper_SSE2(PAR_POLICY, in, out, sfp,
    [](LambdaFunctionParams lfp) {
      __m128 gridcorrection;
      constexpr float scale = 4.0f;

      __m128 cur = _mm_load_ps((const float*)lfp.in[2]);
      __m128 prev = _mm_load_ps((const float*)lfp.in[1]);
      __m128 next = _mm_load_ps((const float*)lfp.in[3]);
      __m128 prev2 = _mm_load_ps((const float*)lfp.in[0]);

      __m128 p_n = _mm_swap_ri(_mm_sign_r(prev - next));

      __m128 fp2 = (cur + prev2) - (prev + next);
      __m128 fp  = (cur - prev2) + p_n;
      __m128 fc  = (cur + prev2) + (prev + next);
      __m128 fn  = (cur - prev2) - p_n;

      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection * _mm_set1_ps(scale);
        fc -= gridcorrection;
      }

      lfp.wiener_factor_3d<pattern>(fp2);
      lfp.wiener_factor_3d<pattern>(fp);
      lfp.wiener_factor_3d<pattern>(fc);
      lfp.wiener_factor_3d<pattern>(fn);

      __m128 result = (fp2 + fp) + (fc + fn);
      if constexpr (degrid) {
        result += gridcorrection;
      }

      result *= _mm_set1_ps(1 / scale);
      _mm_store_ps((float*)lfp.out, result);
    }
  );
}

// 3D5 SSE2 Profiling
// Profiling time: 85,539ms -> 13,925ms
template <bool pattern, bool degrid>
void Apply3D5_SSE2(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  loop_wrapper_SSE2(PAR_POLICY, in, out, sfp,
    [](LambdaFunctionParams lfp) {
      __m128 gridcorrection;
      constexpr float scale = 5.0f;
      __m128 sin72 = _mm_set1_ps(0.95105651629515357211643933337938f);
      __m128 cos72 = _mm_set1_ps(0.30901699437494742410229341718282f);
      __m128 sin144 = _mm_set1_ps(0.58778525229247312916870595463907f);
      __m128 cos144 = _mm_set1_ps(-0.80901699437494742410229341718282f);

      __m128 cur = _mm_load_ps((const float*)lfp.in[2]);
      __m128 prev = _mm_load_ps((const float*)lfp.in[1]);
      __m128 next = _mm_load_ps((const float*)lfp.in[3]);
      __m128 prev2 = _mm_load_ps((const float*)lfp.in[0]);
      __m128 next2 = _mm_load_ps((const float*)lfp.in[4]);

      __m128 sum2 = (next2 + prev2) * cos72
                  + (prev + next) * cos144
                  + cur;
      __m128 dif2 = _mm_swap_ri(_mm_sign_i(
                    (prev2 - next2) * sin72
                  + (next - prev) * sin144
                  ));

      __m128 fp2 = sum2 + dif2;
      __m128 fn2 = sum2 - dif2;

      __m128 sum1 = (next2 + prev2) * cos144
                  + (prev + next) * cos72
                  + cur;
      __m128 dif1 = _mm_swap_ri(_mm_sign_i(
                    (next2 - prev2) * sin144
                  + (next - prev) * sin72
                  ));

      __m128 fp = sum1 + dif1;
      __m128 fn = sum1 - dif1;

      __m128 fc = (prev2 + prev) + cur + (next + next2);

      if constexpr (degrid) {
        gridcorrection = lfp.m_gridcorrection * _mm_set1_ps(scale);
        fc -= gridcorrection;
      }

      lfp.wiener_factor_3d<pattern>(fp2);
      lfp.wiener_factor_3d<pattern>(fp);
      lfp.wiener_factor_3d<pattern>(fc);
      lfp.wiener_factor_3d<pattern>(fn);
      lfp.wiener_factor_3d<pattern>(fn2);

      __m128 result = (fp2 + fp) + (fc + fn) + fn2;
      if constexpr (degrid) {
        result += gridcorrection;
      }

      result *= _mm_set1_ps(1 / scale);
      _mm_store_ps((float*)lfp.out, result);
    }
  );
}


#define DECLARE(pattern, degrid) \
  template void Apply2D_SSE2<pattern, degrid>(fftwf_complex*, SharedFunctionParams);\
  template void Apply3D2_SSE2<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D3_SSE2<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D4_SSE2<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D5_SSE2<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\

DECLARE(true, true)
DECLARE(false, true)
DECLARE(true, false)
DECLARE(false, false)
#undef DECLARE
