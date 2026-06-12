#include "fft/dynamic_library.hpp"

#include <utility>

#if _WIN32
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace neo_fft3d::fft {

DynamicLibrary::DynamicLibrary(void* handle) noexcept : handle_(handle) {
}

DynamicLibrary::~DynamicLibrary() {
  close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

DynamicLibrary DynamicLibrary::OpenFftw() {
#if _WIN32
  HMODULE handle = LoadLibraryW(L"libfftw3f-3");
  if (handle == nullptr) {
    handle = LoadLibraryW(L"fftw3");
  }
  if (handle == nullptr) {
    throw std::runtime_error("libfftw3f-3.dll or fftw3.dll not found. Please put in PATH or use LoadDll() plugin.");
  }
  return DynamicLibrary{handle};
#else
  #ifdef __MACH__
  constexpr const char* library_name = "libfftw3f_threads.dylib";
  constexpr const char* versioned_library_name = "libfftw3f_threads.3.dylib";
  constexpr const char* not_found_message =
      "libfftw3f_threads.dylib or libfftw3f_threads.3.dylib not found. Please install libfftw3.";
  #else
  constexpr const char* library_name = "libfftw3f_threads.so";
  constexpr const char* versioned_library_name = "libfftw3f_threads.so.3";
  constexpr const char* not_found_message =
      "libfftw3f_threads.so or libfftw3f_threads.so.3 not found. Please install libfftw3-single3 (deb) or fftw-devel (rpm) package.";
  #endif

  void* handle = dlopen(library_name, RTLD_NOW);
  if (handle == nullptr) {
    handle = dlopen(versioned_library_name, RTLD_NOW);
  }
  if (handle == nullptr) {
    throw std::runtime_error(not_found_message);
  }
  return DynamicLibrary{handle};
#endif
}

bool DynamicLibrary::loaded() const noexcept {
  return handle_ != nullptr;
}

void* DynamicLibrary::symbol_address(const char* name) const noexcept {
  if (handle_ == nullptr) {
    return nullptr;
  }
#if _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

void DynamicLibrary::close() noexcept {
  if (handle_ == nullptr) {
    return;
  }
#if _WIN32
  FreeLibrary(static_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
  handle_ = nullptr;
}

} // namespace neo_fft3d::fft
