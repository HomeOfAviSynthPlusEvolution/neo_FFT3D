#pragma once

#include "code_impl/code_impl.h"
#include "engine/engine_params.hpp"

#include <memory>

namespace neo_fft3d::engine {

class FilterBackend {
public:
  virtual ~FilterBackend() = default;

  [[nodiscard]] virtual const char* Name() const noexcept = 0;

  virtual void Configure(const EngineParams& params) = 0;

  virtual void Apply2D(ComplexBlockView out, SharedFunctionParams sfp) const = 0;
  virtual void Apply3D(const TemporalComplexBlockViews& in, ComplexBlockView out, SharedFunctionParams sfp) const = 0;
  virtual void Sharpen(ComplexBlockView out, SharedFunctionParams sfp) const = 0;
  virtual void Kalman(ComplexBlockView curr, ComplexBlockView prev, SharedFunctionParams sfp) const = 0;
};

std::unique_ptr<FilterBackend> CreateCpuFilterBackend();

} // namespace neo_fft3d::engine
