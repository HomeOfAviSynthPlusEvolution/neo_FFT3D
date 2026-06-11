#include "fft/fft_backend.hpp"
#include "fftwlite.h"

#include <cstring>
#include <stdexcept>

namespace neo_fft3d::fft {
namespace {

class FftwPlan final : public FFTPlan {
public:
  FftwPlan(FFTFunctionPointers& api, fftwf_plan plan, int bh, int bw) noexcept
    : api_(api), plan_(plan), bh_(bh), bw_(bw) {}

  ~FftwPlan() override {
    if (plan_) {
      api_.fftwf_destroy_plan(plan_);
    }
  }

  void Execute(float* real_in, std::complex<float>* complex_out, int count) override {
    api_.fftwf_execute_dft_r2c(plan_, real_in, reinterpret_cast<fftwf_complex*>(complex_out));
  }

  void Execute(std::complex<float>* complex_in, float* real_out, int count) override {
    api_.fftwf_execute_dft_c2r(plan_, reinterpret_cast<fftwf_complex*>(complex_in), real_out);
  }

private:
  FFTFunctionPointers& api_;
  fftwf_plan plan_ {nullptr};
  int bh_;
  int bw_;
};

class FftwBackend final : public FFTBackend {
public:
  FftwBackend() {
    api_ = std::make_unique<FFTFunctionPointers>();
  }

  ~FftwBackend() override {
    Unload();
  }

  const char* Name() const override { return "fftw"; }

  bool Load() override {
    try {
      api_->load();
      return Loaded();
    } catch (...) {
      return false;
    }
  }

  void Unload() noexcept override {
    if (Loaded()) {
      api_->free();
    }
  }

  bool Loaded() const noexcept override {
    return api_->library != nullptr;
  }

  bool HasThreading() const noexcept override {
    return api_->has_threading();
  }

  void SetThreadCount(int nthreads) override {
    if (HasThreading()) {
      api_->fftwf_init_threads();
      api_->fftwf_plan_with_nthreads(nthreads);
    }
  }

  std::unique_ptr<FFTPlan> CreatePlan(int bh, int bw, Direction dir, int max_batch) override {
    if (!Loaded()) {
      throw std::runtime_error("fftw: backend not loaded");
    }

    int n[2] = { bh, bw };
    fftwf_plan plan = nullptr;

    if (dir == Direction::r2c) {
      int inembed[2] = { bh, bw };
      int onembed[2] = { bh, bw / 2 + 1 };
      plan = api_->fftwf_plan_many_dft_r2c(
        2, n, max_batch,
        nullptr, inembed, 1, bh * bw,
        nullptr, onembed, 1, bh * (bw / 2 + 1),
        FFTW_ESTIMATE
      );
    } else {
      int inembed[2] = { bh, bw / 2 + 1 };
      int onembed[2] = { bh, bw };
      plan = api_->fftwf_plan_many_dft_c2r(
        2, n, max_batch,
        nullptr, inembed, 1, bh * (bw / 2 + 1),
        nullptr, onembed, 1, bh * bw,
        FFTW_ESTIMATE
      );
    }

    if (!plan) {
      throw std::runtime_error("fftw: failed to create plan");
    }

    return std::make_unique<FftwPlan>(*api_, plan, bh, bw);
  }

private:
  std::unique_ptr<FFTFunctionPointers> api_;
};

} // namespace

std::unique_ptr<FFTBackend> CreateFFTWBackend() {
  return std::make_unique<FftwBackend>();
}

} // namespace neo_fft3d::fft
