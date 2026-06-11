#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/core/cpu/kalman_hwy.cpp"

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

template <bool pattern>
void Kalman_Hwy_Impl(float* outcur_ptr, float* outLast_ptr, float* pattern2d_ptr, float* covar_ptr, float* covarProcess_ptr, SharedFunctionParams sfp, int size) {
  const hn::ScalableTag<float> d;
  const size_t N = hn::Lanes(d);

  const auto one = hn::Set(d, 1.0f);
  const auto eps = hn::Set(d, 1e-15f);
  const auto kratio2 = hn::Set(d, sfp.kratio2);

  for (int i = 0; i < size; i += N) {
    auto incur_r = hn::Zero(d);
    auto incur_i = hn::Zero(d);
    hn::LoadInterleaved2(d, outcur_ptr + 2 * i, incur_r, incur_i);

    auto inprev_r = hn::Zero(d);
    auto inprev_i = hn::Zero(d);
    hn::LoadInterleaved2(d, outLast_ptr + 2 * i, inprev_r, inprev_i);

    auto cov_r = hn::Zero(d);
    auto cov_i = hn::Zero(d);
    hn::LoadInterleaved2(d, covar_ptr + 2 * i, cov_r, cov_i);

    auto covP_r = hn::Zero(d);
    auto covP_i = hn::Zero(d);
    hn::LoadInterleaved2(d, covarProcess_ptr + 2 * i, covP_r, covP_i);

    auto sigma = hn::Set(d, sfp.sigmaSquaredNoiseNormed2D);
    if constexpr (pattern) {
      sigma = hn::LoadU(d, pattern2d_ptr + i);
      sigma = hn::Max(sigma, eps);
    }

    auto diff_r = hn::Sub(incur_r, inprev_r);
    auto diff_i = hn::Sub(incur_i, inprev_i);

    auto diff_r_sq = hn::Mul(diff_r, diff_r);
    auto diff_i_sq = hn::Mul(diff_i, diff_i);

    auto threshold = hn::Mul(sigma, kratio2);

    auto mask_motion = hn::Or(hn::Gt(diff_r_sq, threshold), hn::Gt(diff_i_sq, threshold));

    auto sum_r = hn::Add(cov_r, covP_r);
    auto sum_i = hn::Add(cov_i, covP_i);

    auto gain_r = hn::Div(sum_r, hn::Add(sum_r, sigma));
    auto gain_i = hn::Div(sum_i, hn::Add(sum_i, sigma));

    auto next_covP_r = hn::Mul(hn::Mul(gain_r, gain_r), sigma);
    auto next_covP_i = hn::Mul(hn::Mul(gain_i, gain_i), sigma);

    auto next_cov_r = hn::Mul(hn::Sub(one, gain_r), sum_r);
    auto next_cov_i = hn::Mul(hn::Sub(one, gain_i), sum_i);

    auto next_out_r = hn::Add(hn::Mul(gain_r, incur_r), hn::Mul(hn::Sub(one, gain_r), inprev_r));
    auto next_out_i = hn::Add(hn::Mul(gain_i, incur_i), hn::Mul(hn::Sub(one, gain_i), inprev_i));

    auto final_cov_r = hn::IfThenElse(mask_motion, sigma, next_cov_r);
    auto final_cov_i = hn::IfThenElse(mask_motion, sigma, next_cov_i);

    auto final_covP_r = hn::IfThenElse(mask_motion, sigma, next_covP_r);
    auto final_covP_i = hn::IfThenElse(mask_motion, sigma, next_covP_i);

    auto final_out_r = hn::IfThenElse(mask_motion, incur_r, next_out_r);
    auto final_out_i = hn::IfThenElse(mask_motion, incur_i, next_out_i);

    hn::StoreInterleaved2(final_cov_r, final_cov_i, d, covar_ptr + 2 * i);
    hn::StoreInterleaved2(final_covP_r, final_covP_i, d, covarProcess_ptr + 2 * i);
    hn::StoreInterleaved2(final_out_r, final_out_i, d, outLast_ptr + 2 * i);
  }
}

template <bool pattern>
void Kalman_Hwy_Wrap(fftwf_complex* outcur, fftwf_complex* outLast, SharedFunctionParams sfp) {
  const int size = sfp.outpitch;
  for (int block = 0; block < sfp.howmanyblocks; block++) {
    float* outcur_ptr = reinterpret_cast<float*>(outcur + block * sfp.outpitch * sfp.bh);
    float* outLast_ptr = reinterpret_cast<float*>(outLast + block * sfp.outpitch * sfp.bh);
    float* cov_ptr = reinterpret_cast<float*>(sfp.covar + block * sfp.outpitch * sfp.bh);
    float* covP_ptr = reinterpret_cast<float*>(sfp.covarProcess + block * sfp.outpitch * sfp.bh);

    for (int h = 0; h < sfp.bh; h++) {
      float* outcur_row = outcur_ptr + h * sfp.outpitch * 2;
      float* outLast_row = outLast_ptr + h * sfp.outpitch * 2;
      float* pat_row = sfp.pattern2d + h * sfp.outpitch;
      float* cov_row = cov_ptr + h * sfp.outpitch * 2;
      float* covP_row = covP_ptr + h * sfp.outpitch * 2;

      Kalman_Hwy_Impl<pattern>(outcur_row, outLast_row, pat_row, cov_row, covP_row, sfp, size);
    }
  }
}

void Kalman_Hwy_Wrap_t(fftwf_complex* outcur, fftwf_complex* outLast, SharedFunctionParams sfp) { Kalman_Hwy_Wrap<true>(outcur, outLast, sfp); }
void Kalman_Hwy_Wrap_f(fftwf_complex* outcur, fftwf_complex* outLast, SharedFunctionParams sfp) { Kalman_Hwy_Wrap<false>(outcur, outLast, sfp); }

} // namespace HWY_NAMESPACE
} // namespace neo_fft3d::cpu
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace neo_fft3d::cpu {

HWY_EXPORT(Kalman_Hwy_Wrap_t);
HWY_EXPORT(Kalman_Hwy_Wrap_f);

void Kalman_Hwy(fftwf_complex* curr, fftwf_complex* prev, SharedFunctionParams sfp) {
  if (sfp.pfactor != 0.0f) {
    HWY_DYNAMIC_POINTER(Kalman_Hwy_Wrap_t)(curr, prev, sfp);
  } else {
    HWY_DYNAMIC_POINTER(Kalman_Hwy_Wrap_f)(curr, prev, sfp);
  }
}

} // namespace neo_fft3d::cpu
#endif
