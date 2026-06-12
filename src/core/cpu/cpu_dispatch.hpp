#pragma once

#include "code_impl/code_impl.h"
#include "common/fft3d_views.hpp"

namespace neo_fft3d::cpu {

class CpuDispatch {
public:
  void configure(int cpu_flags, float degrid, float pfactor, int bt, int opt);

  void Apply2D(ComplexBlockView out, SharedFunctionParams sfp) const;
  void Apply3D(const TemporalComplexBlockViews& in, ComplexBlockView out, SharedFunctionParams sfp) const;
  void Sharpen(ComplexBlockView out, SharedFunctionParams sfp) const;
  void Kalman(ComplexBlockView curr, ComplexBlockView prev, SharedFunctionParams sfp) const;

private:
  using Apply2DProc = void (*)(fftwf_complex*, SharedFunctionParams);
  using Apply3DProc = void (*)(fftwf_complex**, fftwf_complex*, SharedFunctionParams);
  using SharpenProc = void (*)(fftwf_complex*, SharedFunctionParams);
  using KalmanProc = void (*)(fftwf_complex*, fftwf_complex*, SharedFunctionParams);

  Apply2DProc apply2d_ {nullptr};
  Apply3DProc apply3d_ {nullptr};
  SharpenProc sharpen_ {nullptr};
  KalmanProc kalman_ {nullptr};
};

} // namespace neo_fft3d::cpu
