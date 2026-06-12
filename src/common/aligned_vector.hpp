#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#ifndef FRAME_ALIGN
#define FRAME_ALIGN 64
#endif

using byte = std::uint8_t;

namespace neo_fft3d {

inline void* aligned_malloc_bytes(std::size_t size, std::size_t alignment) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

inline void aligned_free_bytes(void* ptr) noexcept {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

} // namespace neo_fft3d

template <typename T, std::size_t Alignment = FRAME_ALIGN>
class AlignedAllocator {
public:
  using value_type = T;

  static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0, "Alignment must be a positive power of 2");

  AlignedAllocator() noexcept = default;

  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  T* allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }
    if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
      throw std::bad_alloc();
    }

    const std::size_t size_bytes = n * sizeof(T);
    void* ptr = neo_fft3d::aligned_malloc_bytes(size_bytes, Alignment);
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept {
    neo_fft3d::aligned_free_bytes(p);
  }

  bool operator==(const AlignedAllocator&) const noexcept { return true; }
  bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, FRAME_ALIGN>>;
