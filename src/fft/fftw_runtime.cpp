#include "fft/fftw_runtime.hpp"

#include <utility>

namespace neo_fft3d::fft {

std::shared_ptr<const FftwRuntime> FftwRuntime::Load() {
  return std::shared_ptr<const FftwRuntime>(new FftwRuntime(DynamicLibrary::OpenFftw()));
}

FftwRuntime::FftwRuntime(DynamicLibrary library) : library_(std::move(library)) {
  load_symbols();
}

bool FftwRuntime::loaded() const noexcept {
  return library_.loaded();
}

bool FftwRuntime::has_threading() const noexcept {
  return loaded() && (fftwf_init_threads != nullptr) && (fftwf_plan_with_nthreads != nullptr);
}

void FftwRuntime::load_symbols() {
  fftwf_malloc = library_.required_symbol<MallocProc>("fftwf_malloc");
  fftwf_free = library_.required_symbol<FreeProc>("fftwf_free");
  fftwf_plan_many_dft_r2c = library_.required_symbol<PlanManyDftR2CProc>("fftwf_plan_many_dft_r2c");
  fftwf_plan_many_dft_c2r = library_.required_symbol<PlanManyDftC2RProc>("fftwf_plan_many_dft_c2r");
  fftwf_destroy_plan = library_.required_symbol<DestroyPlanProc>("fftwf_destroy_plan");
  fftwf_execute_dft_r2c = library_.required_symbol<ExecuteDftR2CProc>("fftwf_execute_dft_r2c");
  fftwf_execute_dft_c2r = library_.required_symbol<ExecuteDftC2RProc>("fftwf_execute_dft_c2r");
  fftwf_init_threads = library_.optional_symbol<InitThreadsProc>("fftwf_init_threads");
  fftwf_plan_with_nthreads = library_.optional_symbol<PlanWithNThreadsProc>("fftwf_plan_with_nthreads");
}

} // namespace neo_fft3d::fft
