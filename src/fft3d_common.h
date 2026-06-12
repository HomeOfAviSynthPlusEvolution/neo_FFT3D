#pragma once

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <memory>
#include <new>

#ifdef HAS_EXECUTION
  #include <execution>
#endif

#ifndef __cpp_lib_execution
  #undef ENABLE_PAR
#endif

#ifdef ENABLE_PAR
  #define PAR_POLICY std::execution::par
  #define SEQ_POLICY std::execution::seq
#else
  #define PAR_POLICY nullptr
  #define SEQ_POLICY nullptr
#endif

#ifndef FRAME_ALIGN
  #define FRAME_ALIGN 64
#endif

// Ensure fftwlite.h is included BEFORE we use fftwf_complex
#include "fftwlite.h"
#include "dualsynth_compat.hpp"

using byte = std::uint8_t;

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
        if (n == 0) return nullptr;
        if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
            throw std::bad_alloc();

        std::size_t size_bytes = n * sizeof(T);
        if (size_bytes > (std::numeric_limits<std::size_t>::max)() - Alignment + 1)
            throw std::bad_alloc();

        std::size_t aligned_size = (size_bytes + Alignment - 1) & ~(Alignment - 1);

        void* ptr = _aligned_malloc(aligned_size, Alignment);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        _aligned_free(p);
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, FRAME_ALIGN>>;

static_assert(sizeof(std::complex<float>) == sizeof(fftwf_complex) &&
              alignof(std::complex<float>) == alignof(fftwf_complex),
              "std::complex<float> and fftwf_complex must have the identical ABI size and alignment.");

inline fftwf_complex* as_fftw(std::complex<float>* ptr) {
    return reinterpret_cast<fftwf_complex*>(ptr);
}
inline fftwf_complex* as_fftw(const std::shared_ptr<AlignedVector<std::complex<float>>>& sp) {
    return sp ? reinterpret_cast<fftwf_complex*>(sp->data()) : nullptr;
}
inline std::complex<float>* as_complex(fftwf_complex* ptr) {
    return reinterpret_cast<std::complex<float>*>(ptr);
}

#ifndef _WIN32
  #define wsprintf sprintf
#endif

#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

int GetCPUFlags();

struct EngineParams {
  float sigma; // noise level (std deviation) for high frequncies ***
  float beta; // relative noise margin for Wiener filter
  int bw, bh; // block width / height
  int bt; // block size  along time (mumber of frames), =0 for Kalman, >0 for Wiener
  int ow, oh; // overlap width / height
  float kratio; // threshold to sigma ratio for Kalman filter
  float sharpen; // sharpen factor (0 to 1 and above)
  float scutoff; // sharpen cufoff frequency (relative to max) - v1.7
  float svr; // sharpen vertical ratio (0 to 1 and above) - v.1.0
  float smin; // minimum limit for sharpen (prevent noise amplifying) - v.1.1  ***
  float smax; // maximum limit for sharpen (prevent oversharping) - v.1.1      ***
  bool measure; // fft optimal method
  bool interlaced;
  int wintype; // window type
  int pframe; // noise pattern frame number
  int px, py; // noise pattern window x / y position
  bool pshow; // show noise pattern
  float pcutoff; // pattern cutoff frequency (relative to max)
  float pfactor; // noise pattern denoise strength
  float sigma2; // noise level for middle frequencies           ***
  float sigma3; // noise level for low frequencies              ***
  float sigma4; // noise level for lowest (zero) frequencies    ***
  float degrid; // decrease grid
  float dehalo; // remove halo strength - v.1.9
  float hr; // halo radius - v1.9
  float ht; // halo threshold - v1.9
  int l, t, r, b; // cropping
  int opt;

  DSVideoInfo vi;
  IScriptEnvironment* avs_env {nullptr};
  bool has_at_least_v12 {false};
  bool IsChroma;

  int framewidth; // in pixels, not bytes
  int frameheight;
  int framepitch; // in pixels, not bytes
  int framepitch_dst; // in pixels, not bytes
};


struct IOParams {
  int nox, noy;

  // analysis
  AlignedVector<float> wanxl, wanxr, wanyl, wanyr;
  // synthesis
  AlignedVector<float> wsynxl, wsynxr, wsynyl, wsynyr;
};
