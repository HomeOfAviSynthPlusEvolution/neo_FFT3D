#include "engine/filter_backend.hpp"

#include "core/cpu/cpu_dispatch.hpp"

namespace neo_fft3d::engine {

namespace {

class CpuFilterBackend final : public FilterBackend {
public:
  [[nodiscard]] const char* Name() const noexcept override {
    return "cpu";
  }

  void Configure(const EngineParams& params) override {
    dispatch_.Configure(cpu::CpuDispatchConfig{
      .degrid = params.degrid,
      .pfactor = params.pfactor,
      .bt = params.bt,
      .opt = params.opt
    });
  }

  void Apply2D(ComplexBlockView out, SharedFunctionParams sfp) const override {
    dispatch_.Apply2D(out, sfp);
  }

  void Apply3D(const TemporalComplexBlockViews& in, ComplexBlockView out, SharedFunctionParams sfp) const override {
    dispatch_.Apply3D(in, out, sfp);
  }

  void Sharpen(ComplexBlockView out, SharedFunctionParams sfp) const override {
    dispatch_.Sharpen(out, sfp);
  }

  void Kalman(ComplexBlockView curr, ComplexBlockView prev, SharedFunctionParams sfp) const override {
    dispatch_.Kalman(curr, prev, sfp);
  }

private:
  cpu::CpuDispatch dispatch_;
};

} // namespace

std::unique_ptr<FilterBackend> CreateCpuFilterBackend() {
  return std::make_unique<CpuFilterBackend>();
}

} // namespace neo_fft3d::engine
