#include "fft/fft_backend.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace neo_fft3d::fft {
namespace {

class PocketFFTPlan final : public FFTPlan {
public:
  PocketFFTPlan(int bh, int bw, Direction dir)
    : bh_(bh), bw_(bw), dir_(dir) {
    shape_ = { static_cast<std::size_t>(bh), static_cast<std::size_t>(bw) };
    axes_ = { 0, 1 };
    
    stride_in_r_ = { static_cast<std::ptrdiff_t>(bw * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float)) };
    stride_out_c_ = { static_cast<std::ptrdiff_t>((bw / 2 + 1) * sizeof(std::complex<float>)), static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) };
    
    stride_in_c_ = stride_out_c_;
    stride_out_r_ = stride_in_r_;
  }

  void Execute(float* real_in, std::complex<float>* complex_out, int count) override {
    std::size_t r_size = bh_ * bw_;
    std::size_t c_size = bh_ * (bw_ / 2 + 1);
    for (int i = 0; i < count; ++i) {
      float* p_in = real_in + i * r_size;
      std::complex<float>* p_out = complex_out + i * c_size;
      pocketfft::r2c(shape_, stride_in_r_, stride_out_c_, axes_, true, p_in, p_out, 1.0f);
    }
  }

  void Execute(std::complex<float>* complex_in, float* real_out, int count) override {
    std::size_t r_size = bh_ * bw_;
    std::size_t c_size = bh_ * (bw_ / 2 + 1);
    for (int i = 0; i < count; ++i) {
      std::complex<float>* p_in = complex_in + i * c_size;
      float* p_out = real_out + i * r_size;
      pocketfft::c2r(shape_, stride_in_c_, stride_out_r_, axes_, false, p_in, p_out, 1.0f);
    }
  }

private:
  int bh_;
  int bw_;
  Direction dir_;
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

  std::unique_ptr<FFTPlan> CreatePlan(int bh, int bw, Direction dir, int max_batch) override {
    return std::make_unique<PocketFFTPlan>(bh, bw, dir);
  }
};

} // namespace

std::unique_ptr<FFTBackend> CreatePocketFFTBackend() {
  return std::make_unique<PocketFFTBackend>();
}

} // namespace neo_fft3d::fft
