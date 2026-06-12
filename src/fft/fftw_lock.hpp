#pragma once

#include <mutex>

namespace neo_fft3d::fft {

class GlobalLockGuard {
public:
  GlobalLockGuard() : lock_(mutex()) {}

  GlobalLockGuard(const GlobalLockGuard&) = delete;
  GlobalLockGuard& operator=(const GlobalLockGuard&) = delete;

private:
  static std::mutex& mutex() {
    static std::mutex global_mutex;
    return global_mutex;
  }

  std::unique_lock<std::mutex> lock_;
};

} // namespace neo_fft3d::fft
