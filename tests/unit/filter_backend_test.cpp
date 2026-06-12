#include "engine/filter_backend.hpp"
#include "fft/fft_backend.hpp"

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <memory>
#include <string_view>

namespace {

class RecordingPlan final : public neo_fft3d::fft::FFTPlan {
public:
  void Execute(float*, std::complex<float>*, int) override {}
  void Execute(std::complex<float>*, float*, int) override {}
};

class RecordingFFTBackend final : public neo_fft3d::fft::FFTBackend {
public:
  const char* Name() const override { return "recording"; }
  bool Load() override { return true; }
  void Unload() noexcept override {}
  bool Loaded() const noexcept override { return true; }
  bool HasThreading() const noexcept override { return false; }
  void SetThreadCount(int) override {}

  std::unique_ptr<neo_fft3d::fft::FFTPlan> CreatePlan(
    int bh,
    int bw,
    int outpitch,
    neo_fft3d::fft::Direction dir,
    int max_batch,
    neo_fft3d::fft::PlanOptions options,
    neo_fft3d::fft::PlanBuffers buffers
  ) override {
    create_plan_called = true;
    recorded_bh = bh;
    recorded_bw = bw;
    recorded_outpitch = outpitch;
    recorded_dir = dir;
    recorded_max_batch = max_batch;
    recorded_measure = options.measure;
    recorded_real = buffers.real;
    recorded_spectrum = buffers.spectrum;
    return std::make_unique<RecordingPlan>();
  }

  bool create_plan_called {false};
  int recorded_bh {0};
  int recorded_bw {0};
  int recorded_outpitch {0};
  neo_fft3d::fft::Direction recorded_dir {neo_fft3d::fft::Direction::r2c};
  int recorded_max_batch {0};
  bool recorded_measure {false};
  float* recorded_real {nullptr};
  std::complex<float>* recorded_spectrum {nullptr};
};

} // namespace

TEST_CASE("CpuFilterBackend forwards FFT plan creation through the backend boundary", "[filter_backend]") {
  auto fft_backend = std::make_shared<RecordingFFTBackend>();
  auto backend = neo_fft3d::engine::CreateCpuFilterBackend(fft_backend);

  REQUIRE(std::string_view(backend->Name()) == "cpu");

  float real_buffer[64] {};
  std::complex<float> spectrum_buffer[64] {};

  auto plan = backend->CreatePlan(
    8,
    16,
    12,
    neo_fft3d::fft::Direction::r2c,
    3,
    neo_fft3d::fft::PlanOptions{true},
    neo_fft3d::fft::PlanBuffers{real_buffer, spectrum_buffer}
  );

  REQUIRE(plan != nullptr);
  REQUIRE(fft_backend->create_plan_called);
  REQUIRE(fft_backend->recorded_bh == 8);
  REQUIRE(fft_backend->recorded_bw == 16);
  REQUIRE(fft_backend->recorded_outpitch == 12);
  REQUIRE(fft_backend->recorded_dir == neo_fft3d::fft::Direction::r2c);
  REQUIRE(fft_backend->recorded_max_batch == 3);
  REQUIRE(fft_backend->recorded_measure);
  REQUIRE(fft_backend->recorded_real == real_buffer);
  REQUIRE(fft_backend->recorded_spectrum == spectrum_buffer);
}
