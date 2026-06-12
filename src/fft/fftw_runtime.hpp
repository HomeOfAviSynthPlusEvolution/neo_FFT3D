#pragma once

#include "fft/dynamic_library.hpp"
#include "fft/fftw_abi.hpp"

#include <cstddef>
#include <memory>

namespace neo_fft3d::fft {

class FftwRuntime {
public:
  using MallocProc = void* (*)(std::size_t n);
  using FreeProc = void (*)(void* ptr);
  using PlanManyDftR2CProc = fftwf_plan (*)(int rank, const int* n, int howmany, float* in, const int* inembed,
                                            int istride, int idist, fftwf_complex* out, const int* onembed,
                                            int ostride, int odist, unsigned flags);
  using PlanManyDftC2RProc = fftwf_plan (*)(int rank, const int* n, int howmany, fftwf_complex* out,
                                            const int* inembed, int istride, int idist, float* in,
                                            const int* onembed, int ostride, int odist, unsigned flags);
  using DestroyPlanProc = void (*)(fftwf_plan plan);
  using ExecuteDftR2CProc = void (*)(fftwf_plan plan, float* realdata, fftwf_complex* fftsrc);
  using ExecuteDftC2RProc = void (*)(fftwf_plan plan, fftwf_complex* fftsrc, float* realdata);
  using InitThreadsProc = int (*)();
  using PlanWithNThreadsProc = void (*)(int nthreads);

  static std::shared_ptr<const FftwRuntime> Load();

  [[nodiscard]] bool loaded() const noexcept;
  [[nodiscard]] bool has_threading() const noexcept;

  MallocProc fftwf_malloc {nullptr};
  FreeProc fftwf_free {nullptr};
  PlanManyDftR2CProc fftwf_plan_many_dft_r2c {nullptr};
  PlanManyDftC2RProc fftwf_plan_many_dft_c2r {nullptr};
  DestroyPlanProc fftwf_destroy_plan {nullptr};
  ExecuteDftR2CProc fftwf_execute_dft_r2c {nullptr};
  ExecuteDftC2RProc fftwf_execute_dft_c2r {nullptr};
  InitThreadsProc fftwf_init_threads {nullptr};
  PlanWithNThreadsProc fftwf_plan_with_nthreads {nullptr};

private:
  explicit FftwRuntime(DynamicLibrary library);

  void load_symbols();

  DynamicLibrary library_;
};

} // namespace neo_fft3d::fft
