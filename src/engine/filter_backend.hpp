#pragma once
#ifndef NEO_FFT3D_ENGINE_FILTER_BACKEND_HPP
#define NEO_FFT3D_ENGINE_FILTER_BACKEND_HPP

#include "code_impl/code_impl.h"
#include "engine/engine_params.hpp"
#include "fft/fft_backend.hpp"

#include <memory>

namespace neo_fft3d::engine {

class FilterBackend {
public:
  virtual ~FilterBackend() = default;

  virtual const char* Name() const noexcept = 0;

  virtual std::unique_ptr<fft::FFTPlan> CreatePlan(
    int bh,
    int bw,
    int outpitch,
    fft::Direction dir,
    int max_batch,
    fft::PlanOptions options,
    fft::PlanBuffers buffers
  ) = 0;

  virtual void ConfigureKernels(int cpu_flags, const EngineParams& params) = 0;

  virtual void Apply2D(ComplexBlockView out, SharedFunctionParams sfp) const = 0;
  virtual void Apply3D(const TemporalComplexBlockViews& in, ComplexBlockView out, SharedFunctionParams sfp) const = 0;
  virtual void Sharpen(ComplexBlockView out, SharedFunctionParams sfp) const = 0;
  virtual void Kalman(ComplexBlockView curr, ComplexBlockView prev, SharedFunctionParams sfp) const = 0;
};

std::unique_ptr<FilterBackend> CreateCpuFilterBackend(std::shared_ptr<fft::FFTBackend> fft_backend);

} // namespace neo_fft3d::engine

#endif
