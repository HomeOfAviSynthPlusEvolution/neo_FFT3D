#include "fft/fft_backend.hpp"

#include <stdexcept>

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace neo_fft3d::fft {
namespace {

class PocketFFTPlan final : public FFTPlan {
public:
  PocketFFTPlan(int bh, int bw, int outpitch, Direction dir, int batch)
    : bh_(bh), bw_(bw), outpitch_(outpitch), dir_(dir), batch_(batch) {
    shape_ = { static_cast<std::size_t>(bh), static_cast<std::size_t>(bw) };
    axes_ = { 0, 1 };

    stride_in_r_ = { static_cast<std::ptrdiff_t>(bw * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float)) };
    stride_out_c_ = { static_cast<std::ptrdiff_t>(outpitch * sizeof(std::complex<float>)), static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) };

    stride_in_c_ = stride_out_c_;
    stride_out_r_ = stride_in_r_;
  }

  void Execute(float* real_in, std::complex<float>* complex_out, int count) override {
    Validate(Direction::r2c, count);
    const std::size_t r_size = static_cast<std::size_t>(bh_) * static_cast<std::size_t>(bw_);
    const std::size_t c_size = static_cast<std::size_t>(bh_) * static_cast<std::size_t>(outpitch_);
    for (int i = 0; i < count; ++i) {
      float* p_in = real_in + i * r_size;
      std::complex<float>* p_out = complex_out + i * c_size;
      pocketfft::r2c(shape_, stride_in_r_, stride_out_c_, axes_, true, p_in, p_out, 1.0f);
    }
  }

  void Execute(std::complex<float>* complex_in, float* real_out, int count) override {
    Validate(Direction::c2r, count);
    const std::size_t r_size = static_cast<std::size_t>(bh_) * static_cast<std::size_t>(bw_);
    const std::size_t c_size = static_cast<std::size_t>(bh_) * static_cast<std::size_t>(outpitch_);
    for (int i = 0; i < count; ++i) {
      std::complex<float>* p_in = complex_in + i * c_size;
      float* p_out = real_out + i * r_size;
      pocketfft::c2r(shape_, stride_in_c_, stride_out_r_, axes_, false, p_in, p_out, 1.0f);
    }
  }

private:
  void Validate(Direction dir, int count) const {
    if (dir != dir_) {
      throw std::runtime_error("pocketfft: plan direction mismatch");
    }
    if (count != batch_) {
      throw std::runtime_error("pocketfft: plan batch count mismatch");
    }
  }

  int bh_;
  int bw_;
  int outpitch_;
  Direction dir_;
  int batch_;
  pocketfft::shape_t shape_;
  pocketfft::shape_t axes_;
  pocketfft::stride_t stride_in_r_;
  pocketfft::stride_t stride_out_c_;
  pocketfft::stride_t stride_in_c_;
  pocketfft::stride_t stride_out_r_;
};

class PocketFFTBackend final : public FFTBackend {
public:
  PocketFFTBackend() = default;
  const char* Name() const override { return "pocketfft"; }
  bool Load() override { return true; }
  void Unload() noexcept override {}
  bool Loaded() const noexcept override { return true; }
  bool HasThreading() const noexcept override { return false; }
  void SetThreadCount(int) override {}

  std::unique_ptr<FFTPlan> CreatePlan(
    int bh,
    int bw,
    int outpitch,
    Direction dir,
    int max_batch,
    PlanOptions,
    PlanBuffers
  ) override {
    return std::make_unique<PocketFFTPlan>(bh, bw, outpitch, dir, max_batch);
  }
};

} // namespace

std::unique_ptr<FFTBackend> CreatePocketFFTBackend() {
  return std::make_unique<PocketFFTBackend>();
}

} // namespace neo_fft3d::fft
