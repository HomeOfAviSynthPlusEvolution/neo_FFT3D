#include "fft/fft_backend.hpp"
#include "fft/fftw_runtime.hpp"

#include <stdexcept>
#include <utility>

namespace neo_fft3d::fft {
namespace {

class FftwPlan final : public FFTPlan {
public:
  FftwPlan(std::shared_ptr<const FftwRuntime> api, fftwf_plan plan, Direction dir, int batch) noexcept
    : api_(std::move(api)), plan_(plan), dir_(dir), batch_(batch) {}

  ~FftwPlan() override {
    if (plan_ != nullptr) {
      api_->fftwf_destroy_plan(plan_);
    }
  }

  void Execute(float* real_in, std::complex<float>* complex_out, int count) override {
    Validate(Direction::r2c, count);
    api_->fftwf_execute_dft_r2c(plan_, real_in, reinterpret_cast<fftwf_complex*>(complex_out));
  }

  void Execute(std::complex<float>* complex_in, float* real_out, int count) override {
    Validate(Direction::c2r, count);
    api_->fftwf_execute_dft_c2r(plan_, reinterpret_cast<fftwf_complex*>(complex_in), real_out);
  }

private:
  void Validate(Direction dir, int count) const {
    if (dir != dir_) {
      throw std::runtime_error("fftw: plan direction mismatch");
    }
    if (count != batch_) {
      throw std::runtime_error("fftw: plan batch count mismatch");
    }
  }

  std::shared_ptr<const FftwRuntime> api_;
  fftwf_plan plan_ {nullptr};
  Direction dir_;
  int batch_;
};

class FftwBackend final : public FFTBackend {
public:
  ~FftwBackend() override {
    Unload();
  }

  [[nodiscard]] const char* Name() const override { return "fftw"; }

  bool Load() override {
    try {
      api_ = FftwRuntime::Load();
      return Loaded();
    } catch (...) {
      api_.reset();
      return false;
    }
  }

  void Unload() noexcept override {
    api_.reset();
  }

  [[nodiscard]] bool Loaded() const noexcept override {
    return api_ != nullptr && api_->loaded();
  }

  [[nodiscard]] bool HasThreading() const noexcept override {
    return api_ != nullptr && api_->has_threading();
  }

  void SetThreadCount(int nthreads) override {
    if (HasThreading()) {
      api_->fftwf_init_threads();
      api_->fftwf_plan_with_nthreads(nthreads);
    }
  }

  [[nodiscard]] std::unique_ptr<FFTPlan> CreatePlan(
    int bh,
    int bw,
    int outpitch,
    Direction dir,
    int max_batch,
    PlanOptions options,
    PlanBuffers buffers
  ) override {
    if (!Loaded()) {
      throw std::runtime_error("fftw: backend not loaded");
    }
    if (buffers.real == nullptr || buffers.spectrum == nullptr) {
      throw std::runtime_error("fftw: plan creation buffers are required");
    }

    int n[2] = { bh, bw };
    const unsigned flags = options.measure ? FFTW_MEASURE : FFTW_ESTIMATE;
    fftwf_plan plan = nullptr;

    if (dir == Direction::r2c) {
      int inembed[2] = { bh, bw };
      int onembed[2] = { bh, outpitch };
      plan = api_->fftwf_plan_many_dft_r2c(
        2, n, max_batch,
        buffers.real, inembed, 1, bh * bw,
        reinterpret_cast<fftwf_complex*>(buffers.spectrum), onembed, 1, bh * outpitch,
        flags
      );
    } else {
      int inembed[2] = { bh, outpitch };
      int onembed[2] = { bh, bw };
      plan = api_->fftwf_plan_many_dft_c2r(
        2, n, max_batch,
        reinterpret_cast<fftwf_complex*>(buffers.spectrum), inembed, 1, bh * outpitch,
        buffers.real, onembed, 1, bh * bw,
        flags
      );
    }

    if (plan == nullptr) {
      throw std::runtime_error("fftw: failed to create plan");
    }

    return std::make_unique<FftwPlan>(api_, plan, dir, max_batch);
  }

private:
  std::shared_ptr<const FftwRuntime> api_;
};

} // namespace

std::unique_ptr<FFTBackend> CreateFFTWBackend() {
  return std::make_unique<FftwBackend>();
}

} // namespace neo_fft3d::fft
