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

struct PlanOptions {
  bool measure {false};
};

// FFTW_MEASURE needs real work buffers during planning; non-FFTW backends can ignore them.
struct PlanBuffers {
  Real* real {nullptr};
  Complex* spectrum {nullptr};
};

struct FFTPlan {
  virtual ~FFTPlan() = default;
  virtual void Execute(float* real_in, std::complex<float>* complex_out, int count) = 0;
  virtual void Execute(std::complex<float>* complex_in, float* real_out, int count) = 0;
};

class FFTBackend {
public:
  virtual ~FFTBackend() = default;
  [[nodiscard]] virtual const char* Name() const = 0;
  virtual bool Load() = 0;
  virtual void Unload() noexcept = 0;
  [[nodiscard]] virtual bool Loaded() const noexcept = 0;
  [[nodiscard]] virtual bool HasThreading() const noexcept = 0;
  virtual void SetThreadCount(int nthreads) = 0;
  [[nodiscard]] virtual std::unique_ptr<FFTPlan> CreatePlan(
    int bh,
    int bw,
    int outpitch,
    Direction dir,
    int max_batch,
    PlanOptions options,
    PlanBuffers buffers
  ) = 0;
};

std::unique_ptr<FFTBackend> CreateFFTWBackend();
std::unique_ptr<FFTBackend> CreatePocketFFTBackend();

} // namespace neo_fft3d::fft
