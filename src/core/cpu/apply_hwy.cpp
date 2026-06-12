#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/core/cpu/apply_hwy.cpp"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

#include "fft3d_common.h"
#include "core/cpu/core_hwy.h"
#include <dualsynth/mdspan.hpp>

#include <cstddef>

HWY_BEFORE_NAMESPACE();
namespace neo_fft3d::cpu {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

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

template <bool pattern, bool degrid, bool sharpen, bool dehalo>
void Apply2D_Hwy_Impl(
    ds::PlaneView2D<float> out_view,
    ds::PlaneView2D<float> gs_view,
    ds::PlaneView2D<float> pat_view,
    ds::PlaneView2D<float> ws_view,
    ds::PlaneView2D<float> wd_view,
    int h,
    SharedFunctionParams sfp,
    int size,
    float gridfraction
) {
  const hn::ScalableTag<float> d;
  const int N = static_cast<int>(hn::Lanes(d));

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
    hn::LoadInterleaved2(d, &out_view[h, 2 * i], cr, ci);

    if constexpr (degrid) {
      auto gs_r = zero;
      auto gs_i = zero;
      hn::LoadInterleaved2(d, &gs_view[h, 2 * i], gs_r, gs_i);
      auto gc_r = hn::Mul(grid_frac, gs_r);
      auto gc_i = hn::Mul(grid_frac, gs_i);
      cr = hn::Sub(cr, gc_r);
      ci = hn::Sub(ci, gc_i);
    }

    auto psd = hn::Add(hn::Add(hn::Mul(cr, cr), hn::Mul(ci, ci)), eps);

    auto noise = sigma_sq;
    if constexpr (pattern) {
      noise = hn::LoadU(d, &pat_view[h, i]);
    }

    auto factor = hn::Div(hn::Sub(psd, noise), psd);
    factor = hn::Max(factor, lowlimit);

    if constexpr (!pattern) {
      auto s_fact = one;
      if constexpr (sharpen) {
        auto num = hn::Mul(psd, s_coef_max);
        auto den = hn::Mul(hn::Add(psd, s_coef_min), hn::Add(psd, s_coef_max));
        auto s_val = hn::Sqrt(hn::Div(num, den));
        auto w_sharp = hn::LoadU(d, &ws_view[h, i]);
        s_fact = hn::Add(one, hn::Mul(hn::Mul(s_factor_coef, w_sharp), s_val));
      }

      auto d_fact = one;
      if constexpr (dehalo) {
        auto num = hn::Add(psd, d_coef_ht);
        auto w_dehalo = hn::LoadU(d, &wd_view[h, i]);
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
      hn::LoadInterleaved2(d, &gs_view[h, 2 * i], gs_r, gs_i);
      auto gc_r = hn::Mul(grid_frac, gs_r);
      auto gc_i = hn::Mul(grid_frac, gs_i);
      final_r = hn::Add(final_r, gc_r);
      final_i = hn::Add(final_i, gc_i);
    }

    hn::StoreInterleaved2(final_r, final_i, d, &out_view[h, 2 * i]);
  }
}

template <bool pattern, bool degrid>
void Apply2D_Hwy_Wrap(fftwf_complex* out, SharedFunctionParams sfp) {
  const int size = sfp.outpitch;
  for (int block = 0; block < sfp.howmanyblocks; block++) {
    const auto block_offset = complex_block_offset(sfp, block);
    fftwf_complex* out_block = out + block_offset;
    const fftwf_complex* gridsample = sfp.gridsample.fftw_data();
    const float gridfraction = degrid ? sfp.degrid * out_block[0][0] / gridsample[0][0] : 0.0f;

    auto out_view = ds::make_plane_view(
      reinterpret_cast<float*>(out_block),
      sfp.outpitch * 2,
      sfp.bh,
      complex_float_stride_bytes(sfp.outpitch)
    );
    auto gs_view = sfp.gridsample.block_float_view(0);
    auto pat_view = sfp.pattern2d;
    auto ws_view = sfp.wsharpen;
    auto wd_view = sfp.wdehalo;

    for (int h = 0; h < sfp.bh; h++) {
      if (sfp.sharpen == 0.0f && sfp.dehalo == 0.0f) {
        Apply2D_Hwy_Impl<pattern, degrid, false, false>(out_view, gs_view, pat_view, ws_view, wd_view, h, sfp, size, gridfraction);
      } else if (sfp.sharpen != 0.0f && sfp.dehalo == 0.0f) {
        Apply2D_Hwy_Impl<pattern, degrid, true, false>(out_view, gs_view, pat_view, ws_view, wd_view, h, sfp, size, gridfraction);
      } else if (sfp.sharpen == 0.0f && sfp.dehalo != 0.0f) {
        Apply2D_Hwy_Impl<pattern, degrid, false, true>(out_view, gs_view, pat_view, ws_view, wd_view, h, sfp, size, gridfraction);
      } else {
        Apply2D_Hwy_Impl<pattern, degrid, true, true>(out_view, gs_view, pat_view, ws_view, wd_view, h, sfp, size, gridfraction);
      }
    }
  }
}

// Declare four concrete wrappers to avoid comma-in-macro preprocessor issues with HWY_EXPORT
void Apply2D_Hwy_Wrap_tt(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<true, true>(out, sfp); }
void Apply2D_Hwy_Wrap_tf(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<true, false>(out, sfp); }
void Apply2D_Hwy_Wrap_ft(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<false, true>(out, sfp); }
void Apply2D_Hwy_Wrap_ff(fftwf_complex* out, SharedFunctionParams sfp) { Apply2D_Hwy_Wrap<false, false>(out, sfp); }

template <typename D, typename V>
HWY_INLINE void WienerFactor3D_Hwy(const D d, V& dr, V& di, const V& noise, const V& lowlimit) {
  const auto eps = hn::Set(d, 1.0e-15f);
  const auto psd = hn::Add(hn::Add(hn::Mul(dr, dr), hn::Mul(di, di)), eps);
  auto factor = hn::Div(hn::Sub(psd, noise), psd);
  factor = hn::Max(factor, lowlimit);
  dr = hn::Mul(dr, factor);
  di = hn::Mul(di, factor);
}

template <int bt, bool pattern, bool degrid>
void Apply3D_Hwy_Impl(
    ds::PlaneView2D<float> out_view,
    ds::PlaneView2D<float>* in_views,
    ds::PlaneView2D<float> gs_view,
    ds::PlaneView2D<float> pat_view,
    int h,
    SharedFunctionParams sfp,
    int size,
    float gridfraction
) {
  const hn::ScalableTag<float> d;
  const int N = static_cast<int>(hn::Lanes(d));

  const auto zero = hn::Zero(d);
  const auto lowlimit = hn::Set(d, (sfp.beta - 1.0f) / sfp.beta);
  const auto sigma_sq = hn::Set(d, sfp.sigmaSquaredNoiseNormed);
  const auto grid_frac = hn::Set(d, gridfraction);

  for (int i = 0; i < size; i += N) {
    auto noise3d = sigma_sq;
    if constexpr (pattern) {
      noise3d = hn::LoadU(d, &pat_view[h, i]);
    }

    auto gc_r = zero;
    auto gc_i = zero;
    if constexpr (degrid) {
      auto gs_r = zero;
      auto gs_i = zero;
      hn::LoadInterleaved2(d, &gs_view[h, 2 * i], gs_r, gs_i);
      gc_r = hn::Mul(hn::Mul(grid_frac, gs_r), hn::Set(d, (float)bt));
      gc_i = hn::Mul(hn::Mul(grid_frac, gs_i), hn::Set(d, (float)bt));
    }

    if constexpr (bt == 2) {
      auto incur_r = zero, incur_i = zero;
      hn::LoadInterleaved2(d, &in_views[2][h, 2 * i], incur_r, incur_i);

      auto inprev_r = zero, inprev_i = zero;
      hn::LoadInterleaved2(d, &in_views[1][h, 2 * i], inprev_r, inprev_i);

      auto f3d0r = hn::Sub(hn::Add(incur_r, inprev_r), gc_r);
      auto f3d0i = hn::Sub(hn::Add(incur_i, inprev_i), gc_i);
      auto f3d1r = hn::Sub(incur_r, inprev_r);
      auto f3d1i = hn::Sub(incur_i, inprev_i);

      WienerFactor3D_Hwy(d, f3d0r, f3d0i, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, f3d1r, f3d1i, noise3d, lowlimit);

      auto final_r = hn::Mul(hn::Set(d, 0.5f), hn::Add(hn::Add(f3d0r, f3d1r), gc_r));
      auto final_i = hn::Mul(hn::Set(d, 0.5f), hn::Add(hn::Add(f3d0i, f3d1i), gc_i));

      hn::StoreInterleaved2(final_r, final_i, d, &out_view[h, 2 * i]);
    }
    else if constexpr (bt == 3) {
      constexpr float sin120 = 0.86602540378443864676372317075294f;
      constexpr float athird = 1.0f/3.0f;

      auto incur_r = zero, incur_i = zero;
      hn::LoadInterleaved2(d, &in_views[2][h, 2 * i], incur_r, incur_i);

      auto inprev_r = zero, inprev_i = zero;
      hn::LoadInterleaved2(d, &in_views[1][h, 2 * i], inprev_r, inprev_i);

      auto innext_r = zero, innext_i = zero;
      hn::LoadInterleaved2(d, &in_views[3][h, 2 * i], innext_r, innext_i);

      auto pnr = hn::Add(inprev_r, innext_r);
      auto pni = hn::Add(inprev_i, innext_i);

      auto fcr = hn::Sub(hn::Add(incur_r, pnr), gc_r);
      auto fci = hn::Sub(hn::Add(incur_i, pni), gc_i);

      auto half_pnr = hn::Mul(hn::Set(d, 0.5f), pnr);
      auto half_pni = hn::Mul(hn::Set(d, 0.5f), pni);

      auto v_sin120 = hn::Set(d, sin120);
      auto di = hn::Mul(v_sin120, hn::Sub(inprev_i, innext_i));
      auto dr = hn::Mul(v_sin120, hn::Sub(innext_r, inprev_r));

      auto fpr = hn::Add(hn::Sub(incur_r, half_pnr), di);
      auto fnr = hn::Sub(hn::Sub(incur_r, half_pnr), di);
      auto fpi = hn::Add(hn::Sub(incur_i, half_pni), dr);
      auto fni = hn::Sub(hn::Sub(incur_i, half_pni), dr);

      WienerFactor3D_Hwy(d, fcr, fci, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fpr, fpi, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fnr, fni, noise3d, lowlimit);

      auto final_r = hn::Mul(hn::Set(d, athird), hn::Add(hn::Add(hn::Add(fcr, fpr), fnr), gc_r));
      auto final_i = hn::Mul(hn::Set(d, athird), hn::Add(hn::Add(hn::Add(fci, fpi), fni), gc_i));

      hn::StoreInterleaved2(final_r, final_i, d, &out_view[h, 2 * i]);
    }
    else if constexpr (bt == 4) {
      auto incur_r = zero, incur_i = zero;
      hn::LoadInterleaved2(d, &in_views[2][h, 2 * i], incur_r, incur_i);

      auto inprev_r = zero, inprev_i = zero;
      hn::LoadInterleaved2(d, &in_views[1][h, 2 * i], inprev_r, inprev_i);

      auto innext_r = zero, innext_i = zero;
      hn::LoadInterleaved2(d, &in_views[3][h, 2 * i], innext_r, innext_i);

      auto inprev2_r = zero, inprev2_i = zero;
      hn::LoadInterleaved2(d, &in_views[0][h, 2 * i], inprev2_r, inprev2_i);

      auto sum_cur_prev2_r = hn::Add(incur_r, inprev2_r);
      auto sum_cur_prev2_i = hn::Add(incur_i, inprev2_i);
      auto sum_prev_next_r = hn::Add(inprev_r, innext_r);
      auto sum_prev_next_i = hn::Add(inprev_i, innext_i);

      auto diff_cur_prev2_r = hn::Sub(incur_r, inprev2_r);
      auto diff_cur_prev2_i = hn::Sub(incur_i, inprev2_i);
      auto diff_prev_next_r = hn::Sub(inprev_r, innext_r);
      auto diff_prev_next_i = hn::Sub(inprev_i, innext_i);

      auto fp2r = hn::Sub(sum_cur_prev2_r, sum_prev_next_r);
      auto fp2i = hn::Sub(sum_cur_prev2_i, sum_prev_next_i);

      auto fpr = hn::Add(diff_cur_prev2_r, diff_prev_next_i);
      auto fpi = hn::Sub(diff_cur_prev2_i, diff_prev_next_r);

      auto fcr = hn::Sub(hn::Add(sum_cur_prev2_r, sum_prev_next_r), gc_r);
      auto fci = hn::Sub(hn::Add(sum_cur_prev2_i, sum_prev_next_i), gc_i);

      auto fnr = hn::Sub(diff_cur_prev2_r, diff_prev_next_i);
      auto fni = hn::Add(diff_cur_prev2_i, diff_prev_next_r);

      WienerFactor3D_Hwy(d, fp2r, fp2i, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fpr, fpi, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fcr, fci, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fnr, fni, noise3d, lowlimit);

      auto final_r = hn::Mul(hn::Set(d, 0.25f), hn::Add(hn::Add(hn::Add(fp2r, fpr), hn::Add(fcr, fnr)), gc_r));
      auto final_i = hn::Mul(hn::Set(d, 0.25f), hn::Add(hn::Add(hn::Add(fp2i, fpi), hn::Add(fci, fni)), gc_i));

      hn::StoreInterleaved2(final_r, final_i, d, &out_view[h, 2 * i]);
    }
    else if constexpr (bt == 5) {
      constexpr float sin72 = 0.95105651629515357211643933337938f;
      constexpr float cos72 = 0.30901699437494742410229341718282f;
      constexpr float sin144 = 0.58778525229247312916870595463907f;
      constexpr float cos144 = -0.80901699437494742410229341718282f;

      auto incur_r = zero, incur_i = zero;
      hn::LoadInterleaved2(d, &in_views[2][h, 2 * i], incur_r, incur_i);

      auto inprev_r = zero, inprev_i = zero;
      hn::LoadInterleaved2(d, &in_views[1][h, 2 * i], inprev_r, inprev_i);

      auto innext_r = zero, innext_i = zero;
      hn::LoadInterleaved2(d, &in_views[3][h, 2 * i], innext_r, innext_i);

      auto inprev2_r = zero, inprev2_i = zero;
      hn::LoadInterleaved2(d, &in_views[0][h, 2 * i], inprev2_r, inprev2_i);

      auto innext2_r = zero, innext2_i = zero;
      hn::LoadInterleaved2(d, &in_views[4][h, 2 * i], innext2_r, innext2_i);

      const auto v_cos72 = hn::Set(d, cos72);
      const auto v_sin72 = hn::Set(d, sin72);
      const auto v_cos144 = hn::Set(d, cos144);
      const auto v_sin144 = hn::Set(d, sin144);

      auto sum_p2_n2_r = hn::Add(inprev2_r, innext2_r);
      auto sum_p1_n1_r = hn::Add(inprev_r, innext_r);
      auto sum_r = hn::Add(hn::Add(hn::Mul(sum_p2_n2_r, v_cos72), hn::Mul(sum_p1_n1_r, v_cos144)), incur_r);

      auto dif_p2_n2_i = hn::Sub(innext2_i, inprev2_i);
      auto dif_p1_n1_i = hn::Sub(inprev_i, innext_i);
      auto dif_r = hn::Add(hn::Mul(dif_p2_n2_i, v_sin72), hn::Mul(dif_p1_n1_i, v_sin144));

      auto fp2r = hn::Add(sum_r, dif_r);
      auto fn2r = hn::Sub(sum_r, dif_r);

      auto sum_p2_n2_i = hn::Add(inprev2_i, innext2_i);
      auto sum_p1_n1_i = hn::Add(inprev_i, innext_i);
      auto sum_i = hn::Add(hn::Add(hn::Mul(sum_p2_n2_i, v_cos72), hn::Mul(sum_p1_n1_i, v_cos144)), incur_i);

      auto dif_p2_n2_r = hn::Sub(inprev2_r, innext2_r);
      auto dif_p1_n1_r = hn::Sub(innext_r, inprev_r);
      auto dif_i = hn::Add(hn::Mul(dif_p2_n2_r, v_sin72), hn::Mul(dif_p1_n1_r, v_sin144));

      auto fp2i = hn::Add(sum_i, dif_i);
      auto fn2i = hn::Sub(sum_i, dif_i);

      sum_r = hn::Add(hn::Add(hn::Mul(sum_p2_n2_r, v_cos144), hn::Mul(sum_p1_n1_r, v_cos72)), incur_r);
      dif_p2_n2_i = hn::Sub(inprev2_i, innext2_i);
      dif_p1_n1_i = hn::Sub(inprev_i, innext_i);
      dif_r = hn::Add(hn::Mul(dif_p2_n2_i, v_sin144), hn::Mul(dif_p1_n1_i, v_sin72));

      auto fpr = hn::Add(sum_r, dif_r);
      auto fnr = hn::Sub(sum_r, dif_r);

      sum_i = hn::Add(hn::Add(hn::Mul(sum_p2_n2_i, v_cos144), hn::Mul(sum_p1_n1_i, v_cos72)), incur_i);
      dif_p2_n2_r = hn::Sub(innext2_r, inprev2_r);
      dif_p1_n1_r = hn::Sub(innext_r, inprev_r);
      dif_i = hn::Add(hn::Mul(dif_p2_n2_r, v_sin144), hn::Mul(dif_p1_n1_r, v_sin72));

      auto fpi = hn::Add(sum_i, dif_i);
      auto fni = hn::Sub(sum_i, dif_i);

      auto fcr = hn::Sub(hn::Add(hn::Add(incur_r, hn::Add(inprev2_r, inprev_r)), hn::Add(innext_r, innext2_r)), gc_r);
      auto fci = hn::Sub(hn::Add(hn::Add(incur_i, hn::Add(inprev2_i, inprev_i)), hn::Add(innext_i, innext2_i)), gc_i);

      WienerFactor3D_Hwy(d, fp2r, fp2i, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fpr, fpi, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fcr, fci, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fnr, fni, noise3d, lowlimit);
      WienerFactor3D_Hwy(d, fn2r, fn2i, noise3d, lowlimit);

      auto final_r = hn::Mul(hn::Set(d, 0.2f), hn::Add(hn::Add(hn::Add(hn::Add(fp2r, fpr), hn::Add(fcr, fnr)), fn2r), gc_r));
      auto final_i = hn::Mul(hn::Set(d, 0.2f), hn::Add(hn::Add(hn::Add(hn::Add(fp2i, fpi), hn::Add(fci, fni)), fn2i), gc_i));

      hn::StoreInterleaved2(final_r, final_i, d, &out_view[h, 2 * i]);
    }
  }
}

template <int bt, bool pattern, bool degrid>
void Apply3D_Hwy_Wrap(fftwf_complex** in, fftwf_complex* out, SharedFunctionParams sfp) {
  const int size = sfp.outpitch;
  for (int block = 0; block < sfp.howmanyblocks; block++) {
    const auto block_offset = complex_block_offset(sfp, block);
    fftwf_complex* out_block = out + block_offset;
    const fftwf_complex* gridsample = sfp.gridsample.fftw_data();
    const float gridfraction = degrid ? sfp.degrid * (in[2] + block_offset)[0][0] / gridsample[0][0] : 0.0f;

    auto out_view = ds::make_plane_view(
      reinterpret_cast<float*>(out_block),
      sfp.outpitch * 2,
      sfp.bh,
      complex_float_stride_bytes(sfp.outpitch)
    );
    auto gs_view = sfp.gridsample.block_float_view(0);
    auto pat_view = sfp.pattern3d;

    ds::PlaneView2D<float> in_views[5];
    for (int k = 0; k < 5; k++) {
      if (in[k]) {
        fftwf_complex* in_block = in[k] + block_offset;
        in_views[k] = ds::make_plane_view(
          reinterpret_cast<float*>(in_block),
          sfp.outpitch * 2,
          sfp.bh,
          complex_float_stride_bytes(sfp.outpitch)
        );
      } else {
        in_views[k] = ds::PlaneView2D<float>{};
      }
    }

    for (int h = 0; h < sfp.bh; h++) {
      Apply3D_Hwy_Impl<bt, pattern, degrid>(out_view, in_views, gs_view, pat_view, h, sfp, size, gridfraction);
    }
  }
}

#define DEFINE_3D_WRAP(bt, pattern, degrid, suffix) \
  void Apply3D_Hwy_Wrap_##bt##_##suffix(fftwf_complex** in, fftwf_complex* out, SharedFunctionParams sfp) { \
    Apply3D_Hwy_Wrap<bt, pattern, degrid>(in, out, sfp); \
  }

DEFINE_3D_WRAP(2, true, true, tt)
DEFINE_3D_WRAP(2, true, false, tf)
DEFINE_3D_WRAP(2, false, true, ft)
DEFINE_3D_WRAP(2, false, false, ff)

DEFINE_3D_WRAP(3, true, true, tt)
DEFINE_3D_WRAP(3, true, false, tf)
DEFINE_3D_WRAP(3, false, true, ft)
DEFINE_3D_WRAP(3, false, false, ff)

DEFINE_3D_WRAP(4, true, true, tt)
DEFINE_3D_WRAP(4, true, false, tf)
DEFINE_3D_WRAP(4, false, true, ft)
DEFINE_3D_WRAP(4, false, false, ff)

DEFINE_3D_WRAP(5, true, true, tt)
DEFINE_3D_WRAP(5, true, false, tf)
DEFINE_3D_WRAP(5, false, true, ft)
DEFINE_3D_WRAP(5, false, false, ff)

#undef DEFINE_3D_WRAP

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

#define EXPORT_3D(bt) \
  HWY_EXPORT(Apply3D_Hwy_Wrap_##bt##_tt); \
  HWY_EXPORT(Apply3D_Hwy_Wrap_##bt##_tf); \
  HWY_EXPORT(Apply3D_Hwy_Wrap_##bt##_ft); \
  HWY_EXPORT(Apply3D_Hwy_Wrap_##bt##_ff);

EXPORT_3D(2)
EXPORT_3D(3)
EXPORT_3D(4)
EXPORT_3D(5)
#undef EXPORT_3D

void Apply3D_Hwy(fftwf_complex** in, fftwf_complex* out, SharedFunctionParams sfp) {
  int bt = 2;
  if (in[4]) bt = 5;
  else if (in[0]) bt = 4;
  else if (in[3]) bt = 3;
  else bt = 2;

  const bool pattern = (sfp.pfactor != 0.0f);
  const bool degrid = (sfp.degrid != 0.0f);

  if (bt == 2) {
    if (pattern) {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_2_tt)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_2_tf)(in, out, sfp);
    } else {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_2_ft)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_2_ff)(in, out, sfp);
    }
  } else if (bt == 3) {
    if (pattern) {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_3_tt)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_3_tf)(in, out, sfp);
    } else {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_3_ft)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_3_ff)(in, out, sfp);
    }
  } else if (bt == 4) {
    if (pattern) {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_4_tt)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_4_tf)(in, out, sfp);
    } else {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_4_ft)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_4_ff)(in, out, sfp);
    }
  } else {
    if (pattern) {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_5_tt)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_5_tf)(in, out, sfp);
    } else {
      if (degrid) HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_5_ft)(in, out, sfp);
      else HWY_DYNAMIC_POINTER(Apply3D_Hwy_Wrap_5_ff)(in, out, sfp);
    }
  }
}

} // namespace neo_fft3d::cpu
#endif
