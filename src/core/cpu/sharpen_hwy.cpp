#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/core/cpu/sharpen_hwy.cpp"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

#include "fft3d_common.h"
#include "core/cpu/core_hwy.h"
#include <dualsynth/mdspan.hpp>

#include <cstddef>

HWY_BEFORE_NAMESPACE();
namespace neo_fft3d::cpu::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE std::ptrdiff_t interleaved_float_offset(int i) {
  return static_cast<std::ptrdiff_t>(i) * 2;
}

HWY_INLINE std::ptrdiff_t complex_block_offset(const SharedFunctionParams& sfp, int block) {
  return static_cast<std::ptrdiff_t>(block) *
         static_cast<std::ptrdiff_t>(sfp.outpitch) *
         static_cast<std::ptrdiff_t>(sfp.bh);
}

HWY_INLINE std::ptrdiff_t complex_float_stride_bytes(int outpitch) {
  return static_cast<std::ptrdiff_t>(outpitch) *
         2 *
         static_cast<std::ptrdiff_t>(sizeof(float));
}

template <bool degrid, bool sharpen, bool dehalo>
void Sharpen_Hwy_Impl(float* out_complex_ptr, float* gridsample_ptr, float* wsharpen_ptr, float* wdehalo_ptr, SharedFunctionParams sfp, int size, float gridfraction) {
  const hn::ScalableTag<float> d;
  const int N = static_cast<int>(hn::Lanes(d));

  const auto zero = hn::Zero(d);
  const auto one = hn::Set(d, 1.0F);
  const auto grid_frac = hn::Set(d, gridfraction);

  const auto s_factor_coef = hn::Set(d, sfp.sharpen);
  const auto s_coef_max = hn::Set(d, sfp.sigmaSquaredSharpenMaxNormed);
  const auto s_coef_min = hn::Set(d, sfp.sigmaSquaredSharpenMinNormed);
  const auto d_factor_coef = hn::Set(d, sfp.dehalo);
  const auto d_coef_ht = hn::Set(d, sfp.ht2n);

  for (int i = 0; i < size; i += N) {
    const auto offset = interleaved_float_offset(i);
    auto cr = zero;
    auto ci = zero;
    hn::LoadInterleaved2(d, out_complex_ptr + offset, cr, ci);

    if constexpr (degrid) {
      auto gs_r = zero;
      auto gs_i = zero;
      hn::LoadInterleaved2(d, gridsample_ptr + offset, gs_r, gs_i);
      auto gc_r = hn::Mul(grid_frac, gs_r);
      auto gc_i = hn::Mul(grid_frac, gs_i);
      cr = hn::Sub(cr, gc_r);
      ci = hn::Sub(ci, gc_i);
    }

    auto psd = hn::Add(hn::Add(hn::Mul(cr, cr), hn::Mul(ci, ci)), hn::Set(d, 1.0e-15F));

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

    auto factor = one;
    if constexpr (sharpen && !dehalo) {
      factor = s_fact;
    } else if constexpr (!sharpen && dehalo) {
      factor = d_fact;
    } else if constexpr (sharpen && dehalo) {
      factor = hn::Mul(s_fact, d_fact);
    }

    auto final_r = hn::Mul(cr, factor);
    auto final_i = hn::Mul(ci, factor);

    if constexpr (degrid) {
      auto gs_r = zero;
      auto gs_i = zero;
      hn::LoadInterleaved2(d, gridsample_ptr + offset, gs_r, gs_i);
      auto gc_r = hn::Mul(grid_frac, gs_r);
      auto gc_i = hn::Mul(grid_frac, gs_i);
      final_r = hn::Add(final_r, gc_r);
      final_i = hn::Add(final_i, gc_i);
    }

    hn::StoreInterleaved2(final_r, final_i, d, out_complex_ptr + offset);
  }
}

template <bool degrid>
void Sharpen_Hwy_Wrap(fftwf_complex* out, SharedFunctionParams sfp) {
  const int size = sfp.outpitch;
  for (int block = 0; block < sfp.howmanyblocks; block++) {
    const auto block_offset = complex_block_offset(sfp, block);
    fftwf_complex* out_block = out + block_offset;
    const fftwf_complex* gridsample = sfp.gridsample.fftw_data();
    const float gridfraction = degrid ? sfp.degrid * out_block[0][0] / gridsample[0][0] : 0.0F;

    auto out_view = ds::make_plane_view(
      reinterpret_cast<float*>(out_block),
      sfp.outpitch * 2,
      sfp.bh,
      complex_float_stride_bytes(sfp.outpitch)
    );
    auto gs_view = sfp.gridsample.block_float_view(0);
    auto ws_view = sfp.wsharpen;
    auto wd_view = sfp.wdehalo;

    for (int h = 0; h < sfp.bh; h++) {
      if (sfp.sharpen == 0.0F && sfp.dehalo == 0.0F) {
        return;
      }
      if (sfp.sharpen != 0.0F && sfp.dehalo == 0.0F) {
        Sharpen_Hwy_Impl<degrid, true, false>(&out_view[h, 0], &gs_view[h, 0], &ws_view[h, 0], &wd_view[h, 0], sfp, size, gridfraction);
      } else if (sfp.sharpen == 0.0F && sfp.dehalo != 0.0F) {
        Sharpen_Hwy_Impl<degrid, false, true>(&out_view[h, 0], &gs_view[h, 0], &ws_view[h, 0], &wd_view[h, 0], sfp, size, gridfraction);
      } else {
        Sharpen_Hwy_Impl<degrid, true, true>(&out_view[h, 0], &gs_view[h, 0], &ws_view[h, 0], &wd_view[h, 0], sfp, size, gridfraction);
      }
    }
  }
}

// Wrapper proxies to avoid macro comma parser issues
void Sharpen_Hwy_Wrap_t(fftwf_complex* out, SharedFunctionParams sfp) { Sharpen_Hwy_Wrap<true>(out, sfp); }
void Sharpen_Hwy_Wrap_f(fftwf_complex* out, SharedFunctionParams sfp) { Sharpen_Hwy_Wrap<false>(out, sfp); }

} // namespace neo_fft3d::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace neo_fft3d::cpu {

HWY_EXPORT(Sharpen_Hwy_Wrap_t);
HWY_EXPORT(Sharpen_Hwy_Wrap_f);

void Sharpen_Hwy(fftwf_complex* out, SharedFunctionParams sfp) {
  if (sfp.degrid != 0.0F) {
    HWY_DYNAMIC_POINTER(Sharpen_Hwy_Wrap_t)(out, sfp);
  } else {
    HWY_DYNAMIC_POINTER(Sharpen_Hwy_Wrap_f)(out, sfp);
  }
}

} // namespace neo_fft3d::cpu
#endif
