#pragma once

#include <stdexcept>
#include <string>

namespace neo_fft3d::fft {

class DynamicLibrary {
public:
  DynamicLibrary() noexcept = default;
  ~DynamicLibrary();

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  static DynamicLibrary OpenFftw();

  [[nodiscard]] bool loaded() const noexcept;
  [[nodiscard]] void* symbol_address(const char* name) const noexcept;

  template <typename Function>
  [[nodiscard]] Function required_symbol(const char* name) const {
    void* address = symbol_address(name);
    if (address == nullptr) {
      throw std::runtime_error(std::string("Library function is missing: ") + name);
    }
    return reinterpret_cast<Function>(address);
  }

  template <typename Function>
  [[nodiscard]] Function optional_symbol(const char* name) const noexcept {
    return reinterpret_cast<Function>(symbol_address(name));
  }

private:
  explicit DynamicLibrary(void* handle) noexcept;

  void close() noexcept;

  void* handle_ {nullptr};
};

} // namespace neo_fft3d::fft
