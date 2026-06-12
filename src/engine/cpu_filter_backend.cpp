#include "engine/filter_backend.hpp"

#include "core/cpu/cpu_dispatch.hpp"

#include <stdexcept>
#include <utility>

namespace neo_fft3d::engine {

namespace {

class CpuFilterBackend final : public FilterBackend {
public:
  explicit CpuFilterBackend(std::shared_ptr<fft::FFTBackend> fft_backend) :
    fft_backend_(std::move(fft_backend)) {
    if (!fft_backend_) {
      throw std::invalid_argument("neo_fft3d: CPU filter backend requires an FFT backend");
    }
  }

  [[nodiscard]] const char* Name() const noexcept override {
    return "cpu";
  }

  [[nodiscard]] std::unique_ptr<fft::FFTPlan> CreatePlan(
    int bh,
    int bw,
    int outpitch,
    fft::Direction dir,
    int max_batch,
    fft::PlanOptions options,
    fft::PlanBuffers buffers
  ) override {
    return fft_backend_->CreatePlan(bh, bw, outpitch, dir, max_batch, options, buffers);
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
  std::shared_ptr<fft::FFTBackend> fft_backend_;
  cpu::CpuDispatch dispatch_;
};

} // namespace

std::unique_ptr<FilterBackend> CreateCpuFilterBackend(std::shared_ptr<fft::FFTBackend> fft_backend) {
  return std::make_unique<CpuFilterBackend>(std::move(fft_backend));
}

} // namespace neo_fft3d::engine
