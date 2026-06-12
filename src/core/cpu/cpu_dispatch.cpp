#include "core/cpu/cpu_dispatch.hpp"

#include "core/cpu/core_hwy.h"

namespace neo_fft3d::cpu {

void CpuDispatch::configure(float degrid, float pfactor, int bt, int opt) {
  Apply2DProc apply2d_c {};
  Apply3DProc apply3d2_c {};
  Apply3DProc apply3d3_c {};
  Apply3DProc apply3d4_c {};
  Apply3DProc apply3d5_c {};

  if (degrid != 0 && pfactor == 0) {
    apply2d_c = ::Apply2D_C<false, true>;
    apply3d2_c = ::Apply3D2_C<false, true>;
    apply3d3_c = ::Apply3D3_C<false, true>;
    apply3d4_c = ::Apply3D4_C<false, true>;
    apply3d5_c = ::Apply3D5_C<false, true>;
  }
  else if (degrid == 0 && pfactor == 0) {
    apply2d_c = ::Apply2D_C<false, false>;
    apply3d2_c = ::Apply3D2_C<false, false>;
    apply3d3_c = ::Apply3D3_C<false, false>;
    apply3d4_c = ::Apply3D4_C<false, false>;
    apply3d5_c = ::Apply3D5_C<false, false>;
  }
  else if (degrid != 0 && pfactor != 0) {
    apply2d_c = ::Apply2D_C<true, true>;
    apply3d2_c = ::Apply3D2_C<true, true>;
    apply3d3_c = ::Apply3D3_C<true, true>;
    apply3d4_c = ::Apply3D4_C<true, true>;
    apply3d5_c = ::Apply3D5_C<true, true>;
  }
  else {
    apply2d_c = ::Apply2D_C<true, false>;
    apply3d2_c = ::Apply3D2_C<true, false>;
    apply3d3_c = ::Apply3D3_C<true, false>;
    apply3d4_c = ::Apply3D4_C<true, false>;
    apply3d5_c = ::Apply3D5_C<true, false>;
  }

  SharpenProc sharpen_c = degrid != 0 ? ::Sharpen_C<true> : ::Sharpen_C<false>;
  KalmanProc kalman_c = pfactor != 0 ? ::Kalman_C<true> : ::Kalman_C<false>;

  Apply3DProc apply3d_c {};
  switch (bt) {
  case 2:
    apply3d_c = apply3d2_c;
    break;
  case 3:
    apply3d_c = apply3d3_c;
    break;
  case 4:
    apply3d_c = apply3d4_c;
    break;
  case 5:
    apply3d_c = apply3d5_c;
    break;
  default:
    apply3d_c = apply3d2_c;
    break;
  }

  apply2d_ = apply2d_c;
  apply3d_ = apply3d_c;
  sharpen_ = sharpen_c;
  kalman_ = kalman_c;

  if (opt != 1) {
    apply2d_ = Apply2D_Hwy;
    apply3d_ = Apply3D_Hwy;
    sharpen_ = Sharpen_Hwy;
    kalman_ = Kalman_Hwy;
  }
}

void CpuDispatch::Apply2D(ComplexBlockView out, SharedFunctionParams sfp) const {
  apply2d_(out.fftw_data(), sfp);
}

void CpuDispatch::Apply3D(const TemporalComplexBlockViews& in, ComplexBlockView out, SharedFunctionParams sfp) const {
  fftwf_complex* raw_in[5] {};
  for (int slot = 0; slot < 5; ++slot) {
    raw_in[slot] = in[slot].fftw_data();
  }
  apply3d_(raw_in, out.fftw_data(), sfp);
}

void CpuDispatch::Sharpen(ComplexBlockView out, SharedFunctionParams sfp) const {
  sharpen_(out.fftw_data(), sfp);
}

void CpuDispatch::Kalman(ComplexBlockView curr, ComplexBlockView prev, SharedFunctionParams sfp) const {
  kalman_(curr.fftw_data(), prev.fftw_data(), sfp);
}

} // namespace neo_fft3d::cpu
