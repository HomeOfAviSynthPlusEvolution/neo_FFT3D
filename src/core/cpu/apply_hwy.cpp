#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/core/cpu/apply_hwy.cpp"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

#include "fft3d_common.h"
#include "functions.h"
#include "core/cpu/core_hwy.h"

HWY_BEFORE_NAMESPACE();
namespace neo_fft3d::cpu {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <bool pattern, bool degrid, bool sharpen, bool dehalo>
void Apply2D_Hwy_Impl(float* out_complex_ptr, float* gridsample_ptr, float* pattern2d_ptr, float* wsharpen_ptr, float* wdehalo_ptr, SharedFunctionParams sfp, int size, float gridfraction) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);

  const auto zero = hn::Zero(d);
  const auto eps = hn::Set(d, 1.0e-15f);
  const auto one = hn::Set(d, 1.0f);
  const auto lowlimit = hn::Set(d, (sfp.beta - 1.0f) / sfp.beta);
  const auto sigma_sq = hn::Set(d, sfp.sigmaSquaredNoiseNormed);
  const auto grid_frac = hn::Set(d, gridfraction);

  const auto s_factor_coef = hn::Set(d, sfp.sharpen);
  const auto s_coef_max = hn::Set(d, sfp.sigmaSquaredSharpenMaxNormed);
  const auto s_coef_min = hn::Set(d, sfp.sigmaSquaredSharpenMinNormed);
  const auto d_factor_coef = hn::Set(d, sfp.dehalo);
  const auto d_coef_ht = hn::Set(d, sfp.ht2n);

  for (int i = 0; i < size; i += N) {
    auto cr = zero;
    auto ci = zero;
    hn::LoadInterleaved2(d, out_complex_ptr + 2 * i, cr, ci);

    if constexpr (degrid) {
      auto gs_r = zero;
      auto gs_i = zero;
      hn::LoadInterleaved2(d, gridsample_ptr + 2 * i, gs_r, gs_i);
      auto gc_r = hn::Mul(grid_frac, gs_r);
      auto gc_i = hn::Mul(grid_frac, gs_i);
      cr = hn::Sub(cr, gc_r);
      ci = hn::Sub(ci, gc_i);
    }

    auto psd = hn::Add(hn::Add(hn::Mul(cr, cr), hn::Mul(ci, ci)), eps);

    auto noise = sigma_sq;
    if constexpr (pattern) {
      noise = hn::LoadU(d, pattern2d_ptr + i);
    }

    auto factor = hn::Div(hn::Sub(psd, noise), psd);
    factor = hn::Max(factor, lowlimit);

    if constexpr (!pattern) {
      auto s_fact = one;
      if constexpr (sharpen) {
        auto num = hn::Mul(psd, s_coef_max);
        auto den = hn::Mul(hn::Add(psd, s_coef_min), hn::Add(psd, s_coef_max));
        auto s_val = hn::Sqrt(hn::Div(num, den));
        auto w_sharp = hn::LoadU(d, wsharpen_ptr + i);
        s_fact = hn::Add(one, hn::Mul(hn::Mul(s_factor_coef, w_sharp), s_val));
      }

      auto d_fact = one;
      if constexpr (dehalo) {
        auto num = hn::Add(psd, d_coef_ht);
        auto w_dehalo = hn::LoadU(d, wdehalo_ptr + i);
        auto den = hn::Add(num, hn::Mul(hn::Mul(d_factor_coef, w_dehalo), psd));
        d_fact = hn::Div(num, den);
      }

      if constexpr (sharpen && !dehalo) {
        factor = hn::Mul(factor, s_fact);
      } else if constexpr (!sharpen && dehalo) {
        factor = hn::Mul(factor, d_fact);
      } else if constexpr (sharpen && dehalo) {
        factor = hn::Mul(factor, hn::Mul(s_fact, d_fact));
      }
    }

    auto final_r = hn::Mul(cr, factor);
    auto final_i = hn::Mul(ci, factor);

    if constexpr (degrid) {
      auto gs_r = zero;
      auto gs_i = zero;
      hn::LoadInterleaved2(d, gridsample_ptr + 2 * i, gs_r, gs_i);
      auto gc_r = hn::Mul(grid_frac, gs_r);
      auto gc_i = hn::Mul(grid_frac, gs_i);
      final_r = hn::Add(final_r, gc_r);
      final_i = hn::Add(final_i, gc_i);
    }

    hn::StoreInterleaved2(final_r, final_i, d, out_complex_ptr + 2 * i);
  }
}

template <bool pattern, bool degrid>
void Apply2D_Hwy_Wrap(fftwf_complex* out, SharedFunctionParams sfp) {
  const int size = sfp.outpitch;
  for (int block = 0; block < sfp.howmanyblocks; block++) {
    fftwf_complex* out_block = out + block * sfp.outpitch * sfp.bh;
    float* out_ptr = reinterpret_cast<float*>(out_block);
    float* gs_ptr = reinterpret_cast<float*>(sfp.gridsample);
    const float gridfraction = degrid ? sfp.degrid * out_block[0][0] / sfp.gridsample[0][0] : 0.0f;

    for (int h = 0; h < sfp.bh; h++) {
      float* out_row = out_ptr + h * sfp.outpitch * 2;
      float* gs_row = gs_ptr + h * sfp.outpitch * 2;
      float* pat_row = sfp.pattern2d + h * sfp.outpitch;
      float* ws_row = sfp.wsharpen + h * sfp.outpitch;
      float* wd_row = sfp.wdehalo + h * sfp.outpitch;

      if (sfp.sharpen == 0.0f && sfp.dehalo == 0.0f) {
        Apply2D_Hwy_Impl<pattern, degrid, false, false>(out_row, gs_row, pat_row, ws_row, wd_row, sfp, size, gridfraction);
      } else if (sfp.sharpen != 0.0f && sfp.dehalo == 0.0f) {
        Apply2D_Hwy_Impl<pattern, degrid, true, false>(out_row, gs_row, pat_row, ws_row, wd_row, sfp, size, gridfraction);
      } else if (sfp.sharpen == 0.0f && sfp.dehalo != 0.0f) {
        Apply2D_Hwy_Impl<pattern, degrid, false, true>(out_row, gs_row, pat_row, ws_row, wd_row, sfp, size, gridfraction);
      } else {
        Apply2D_Hwy_Impl<pattern, degrid, true, true>(out_row, gs_row, pat_row, ws_row, wd_row, sfp, size, gridfraction);
      }
    }
  }
}

// Declare four concrete wrappers to avoid comma-in-macro preprocessor issues with HWY_EXPORT
void Apply2D_Hwy_Wrap_tt(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<true, true>(out, sfp); }
void Apply2D_Hwy_Wrap_tf(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<true, false>(out, sfp); }
void Apply2D_Hwy_Wrap_ft(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<false, true>(out, sfp); }
void Apply2D_Hwy_Wrap_ff(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<false, false>(out, sfp); }

} // namespace HWY_NAMESPACE
} // namespace neo_fft3d::cpu
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace neo_fft3d::cpu {

HWY_EXPORT(Apply2D_Hwy_Wrap_tt);
HWY_EXPORT(Apply2D_Hwy_Wrap_tf);
HWY_EXPORT(Apply2D_Hwy_Wrap_ft);
HWY_EXPORT(Apply2D_Hwy_Wrap_ff);

void Apply2D_Hwy(fftwf_complex* out, SharedFunctionParams sfp) {
  if (sfp.pfactor != 0.0f) {
    if (sfp.degrid != 0.0f) {
      HWY_DYNAMIC_POINTER(Apply2D_Hwy_Wrap_tt)(out, sfp);
    } else {
      HWY_DYNAMIC_POINTER(Apply2D_Hwy_Wrap_tf)(out, sfp);
    }
  } else {
    if (sfp.degrid != 0.0f) {
      HWY_DYNAMIC_POINTER(Apply2D_Hwy_Wrap_ft)(out, sfp);
    } else {
      HWY_DYNAMIC_POINTER(Apply2D_Hwy_Wrap_ff)(out, sfp);
    }
  }
}

} // namespace neo_fft3d::cpu
#endif
