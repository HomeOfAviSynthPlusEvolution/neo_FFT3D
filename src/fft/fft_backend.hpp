#pragma once
#include <complex>
#include <memory>
#include <vector>

namespace neo_fft3d::fft {

using Real = float;
using Complex = std::complex<float>;

enum class Direction {
  r2c,
  c2r,
};

struct FFTPlan {
  virtual ~FFTPlan() = default;
  virtual void Execute(float* real_in, std::complex<float>* complex_out, int count) = 0;
  virtual void Execute(std::complex<float>* complex_in, float* real_out, int count) = 0;
};

class FFTBackend {
public:
  virtual ~FFTBackend() = default;
  virtual const char* Name() const = 0;
  virtual bool Load() = 0;
  virtual void Unload() noexcept = 0;
  virtual bool Loaded() const noexcept = 0;
  virtual bool HasThreading() const noexcept = 0;
  virtual void SetThreadCount(int nthreads) = 0;
  virtual std::unique_ptr<FFTPlan> CreatePlan(int bh, int bw, int outpitch, Direction dir, int max_batch) = 0;
};

std::unique_ptr<FFTBackend> CreateFFTWBackend();
std::unique_ptr<FFTBackend> CreatePocketFFTBackend();

} // namespace neo_fft3d::fft
