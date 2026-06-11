#pragma once

#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <mutex>

#ifdef _WIN32
  #include <malloc.h>
#else
  #include <cstdlib>
  #define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)
  #define _aligned_free(ptr) free(ptr)
#endif

class IScriptEnvironment;

namespace ds {
class VideoFrameProvider;
struct VideoFrameView;
struct MutableVideoFrameView;
}

struct DSFormat {
  bool IsFamilyYUV {false};
  bool IsFamilyRGB {false};
  bool IsFamilyGray {false};
  int Planes {1};
  int BytesPerSample {1};
  int BitsPerSample {8};
  int SSW {0};
  int SSH {0};
};

struct DSVideoInfo {
  int Width {0};
  int Height {0};
  int Frames {0};
  int FPSNum {0};
  int FPSDen {0};
  DSFormat Format;
};

struct DSFrame {
  int FrameWidth {0}, FrameHeight {0};
  const unsigned char** SrcPointers {nullptr};
  int* StrideBytes {nullptr};
  unsigned char** DstPointers {nullptr};
  DSFormat Format;

  bool is_heap_allocated {false};
  unsigned char* heap_buffer {nullptr};

  DSFrame() {}

  DSFrame(const ds::VideoFrameView& vsView);
  DSFrame(const ds::MutableVideoFrameView& vsView);

  DSFrame Create(bool copy) {
    DSFrame new_frame;
    new_frame.Format = Format;
    new_frame.FrameWidth = FrameWidth;
    new_frame.FrameHeight = FrameHeight;
    new_frame.is_heap_allocated = true;

    new_frame.SrcPointers = new const unsigned char*[Format.Planes];
    new_frame.DstPointers = new unsigned char*[Format.Planes];
    new_frame.StrideBytes = new int[Format.Planes];

    std::size_t total_size = 0;
    std::vector<std::size_t> plane_offsets;
    for (int i = 0; i < Format.Planes; i++) {
      int height = FrameHeight;
      if (Format.IsFamilyYUV && i > 0 && i < 3) {
        height >>= Format.SSH;
      }
      plane_offsets.push_back(total_size);
      total_size += StrideBytes[i] * height;
    }

    new_frame.heap_buffer = static_cast<unsigned char*>(_aligned_malloc(total_size, 64));
    std::memset(new_frame.heap_buffer, 0, total_size);

    for (int i = 0; i < Format.Planes; i++) {
      new_frame.StrideBytes[i] = StrideBytes[i];
      new_frame.DstPointers[i] = new_frame.heap_buffer + plane_offsets[i];
      new_frame.SrcPointers[i] = new_frame.DstPointers[i];

      if (copy && SrcPointers[i]) {
        int height = FrameHeight;
        if (Format.IsFamilyYUV && i > 0 && i < 3) {
          height >>= Format.SSH;
        }
        std::memcpy(new_frame.DstPointers[i], SrcPointers[i], StrideBytes[i] * height);
      }
    }

    return new_frame;
  }

  ~DSFrame() {
    cleanup();
  }

  void cleanup() {
    if (SrcPointers) { delete[] SrcPointers; SrcPointers = nullptr; }
    if (DstPointers) { delete[] DstPointers; DstPointers = nullptr; }
    if (StrideBytes) { delete[] StrideBytes; StrideBytes = nullptr; }
    if (is_heap_allocated && heap_buffer) {
      _aligned_free(heap_buffer);
      heap_buffer = nullptr;
    }
  }

  DSFrame(const DSFrame& old) {
    std::memcpy(this, &old, sizeof(DSFrame));
    if (old.SrcPointers) {
      SrcPointers = new const unsigned char*[Format.Planes];
      std::copy_n(old.SrcPointers, Format.Planes, SrcPointers);
    }
    if (old.DstPointers) {
      DstPointers = new unsigned char*[Format.Planes];
      std::copy_n(old.DstPointers, Format.Planes, DstPointers);
    }
    if (old.StrideBytes) {
      StrideBytes = new int[Format.Planes];
      std::copy_n(old.StrideBytes, Format.Planes, StrideBytes);
    }
    if (old.is_heap_allocated && old.heap_buffer) {
      std::size_t total_size = 0;
      for (int i = 0; i < Format.Planes; i++) {
        int height = FrameHeight;
        if (Format.IsFamilyYUV && i > 0 && i < 3) height >>= Format.SSH;
        total_size += StrideBytes[i] * height;
      }
      heap_buffer = static_cast<unsigned char*>(_aligned_malloc(total_size, 64));
      std::memcpy(heap_buffer, old.heap_buffer, total_size);
      for (int i = 0; i < Format.Planes; i++) {
        std::size_t offset = old.DstPointers[i] - old.heap_buffer;
        DstPointers[i] = heap_buffer + offset;
        SrcPointers[i] = DstPointers[i];
      }
    }
  }

  DSFrame& operator=(const DSFrame& old) {
    if (this == &old) return *this;
    cleanup();
    std::memcpy(this, &old, sizeof(DSFrame));
    if (old.SrcPointers) {
      SrcPointers = new const unsigned char*[Format.Planes];
      std::copy_n(old.SrcPointers, Format.Planes, SrcPointers);
    }
    if (old.DstPointers) {
      DstPointers = new unsigned char*[Format.Planes];
      std::copy_n(old.DstPointers, Format.Planes, DstPointers);
    }
    if (old.StrideBytes) {
      StrideBytes = new int[Format.Planes];
      std::copy_n(old.StrideBytes, Format.Planes, StrideBytes);
    }
    if (old.is_heap_allocated && old.heap_buffer) {
      std::size_t total_size = 0;
      for (int i = 0; i < Format.Planes; i++) {
        int height = FrameHeight;
        if (Format.IsFamilyYUV && i > 0 && i < 3) height >>= Format.SSH;
        total_size += StrideBytes[i] * height;
      }
      heap_buffer = static_cast<unsigned char*>(_aligned_malloc(total_size, 64));
      std::memcpy(heap_buffer, old.heap_buffer, total_size);
      for (int i = 0; i < Format.Planes; i++) {
        std::size_t offset = old.DstPointers[i] - old.heap_buffer;
        DstPointers[i] = heap_buffer + offset;
        SrcPointers[i] = DstPointers[i];
      }
    }
    return *this;
  }

  DSFrame(DSFrame&& old) noexcept {
    std::memcpy(this, &old, sizeof(DSFrame));
    old.SrcPointers = nullptr;
    old.DstPointers = nullptr;
    old.StrideBytes = nullptr;
    old.heap_buffer = nullptr;
    old.is_heap_allocated = false;
  }

  DSFrame& operator=(DSFrame&& old) noexcept {
    if (this == &old) return *this;
    cleanup();
    std::memcpy(this, &old, sizeof(DSFrame));
    old.SrcPointers = nullptr;
    old.DstPointers = nullptr;
    old.StrideBytes = nullptr;
    old.heap_buffer = nullptr;
    old.is_heap_allocated = false;
    return *this;
  }
};

using FetchFrameBridgeFn = DSFrame (*)(void* Opaque, int frame_num);

struct FetchFrameFunctor {
  FetchFrameBridgeFn Fn {nullptr};
  void* Opaque {nullptr};

  DSFrame operator()(int frame_num) const {
    if (Fn) return Fn(Opaque, frame_num);
    return DSFrame();
  }
};

inline std::mutex& GetFFTWMutex() {
  static std::mutex m;
  return m;
}

class GlobalLockGuard {
  void* env;
  const char* name;
  bool acquired;
  bool is_legacy;

public:
  GlobalLockGuard(void* _env, const char* _name, bool use_avs_lock)
    : env(_env), name(_name), acquired(false), is_legacy(false)
  {
    if (!name) return;
    if (std::strcmp(name, "fftw") == 0) {
      GetFFTWMutex().lock();
      acquired = true;
      is_legacy = true;
    }
  }

  ~GlobalLockGuard() {
    if (acquired) {
      if (is_legacy) {
        GetFFTWMutex().unlock();
      }
    }
  }

  GlobalLockGuard(const GlobalLockGuard&) = delete;
  GlobalLockGuard& operator=(const GlobalLockGuard&) = delete;
};
