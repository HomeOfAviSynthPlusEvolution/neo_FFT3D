#pragma once
#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "fft3d_common.h"
#include "code_impl/code_impl.h"
#include <avs/cpuid.h>

struct FilterFunctionPointers {
  typedef void (*Apply3D_PROC)(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  // C
    void (*Apply2D_C_Dispatch)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D_C_Dispatch;
    Apply3D_PROC Apply3D2_C_Dispatch;
    Apply3D_PROC Apply3D3_C_Dispatch;
    Apply3D_PROC Apply3D4_C_Dispatch;
    Apply3D_PROC Apply3D5_C_Dispatch;
    void (*Sharpen_C_Dispatch)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman_C_Dispatch)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  // SSE2
    void (*Apply2D_SSE2_Dispatch)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D_SSE2_Dispatch;
    Apply3D_PROC Apply3D2_SSE2_Dispatch;
    Apply3D_PROC Apply3D3_SSE2_Dispatch;
    Apply3D_PROC Apply3D4_SSE2_Dispatch;
    Apply3D_PROC Apply3D5_SSE2_Dispatch;
    void (*Sharpen_SSE2_Dispatch)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman_SSE2_Dispatch)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  // AVX
    void (*Apply2D_AVX_Dispatch)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D_AVX_Dispatch;
    Apply3D_PROC Apply3D2_AVX_Dispatch;
    Apply3D_PROC Apply3D3_AVX_Dispatch;
    Apply3D_PROC Apply3D4_AVX_Dispatch;
    Apply3D_PROC Apply3D5_AVX_Dispatch;
    void (*Sharpen_AVX_Dispatch)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman_AVX_Dispatch)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  // AVX512
    void (*Apply2D_AVX512_Dispatch)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D_AVX512_Dispatch;
    Apply3D_PROC Apply3D2_AVX512_Dispatch;
    Apply3D_PROC Apply3D3_AVX512_Dispatch;
    Apply3D_PROC Apply3D4_AVX512_Dispatch;
    Apply3D_PROC Apply3D5_AVX512_Dispatch;
    void (*Sharpen_AVX512_Dispatch)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman_AVX512_Dispatch)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  // Dispatcher -> [C / SSE2 / AVX]
    void (*Apply2D)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D;
    void (*Sharpen)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  void set_ffp(int CPUFlags, float degrid, float pfactor, int bt, int opt)
  {
    // opt
    // 1: C
    // 2: SSE
    // 3: AVX
    // 4: AVX512
    // 0: auto-detect without AVX512
    // -1: auto-detect with AVX512
    if (degrid != 0 && pfactor == 0) {
      // Default Dispatcher
      Apply2D_C_Dispatch = Apply2D_C<false, true>;;
      Apply3D2_C_Dispatch = Apply3D2_C<false, true>;
      Apply3D3_C_Dispatch = Apply3D3_C<false, true>;
      Apply3D4_C_Dispatch = Apply3D4_C<false, true>;
      Apply3D5_C_Dispatch = Apply3D5_C<false, true>;

      Apply2D_SSE2_Dispatch = Apply2D_SSE2<false, true>;
      Apply3D2_SSE2_Dispatch = Apply3D2_SSE2<false, true>;
      Apply3D3_SSE2_Dispatch = Apply3D3_SSE2<false, true>;
      Apply3D4_SSE2_Dispatch = Apply3D4_SSE2<false, true>;
      Apply3D5_SSE2_Dispatch = Apply3D5_SSE2<false, true>;

      Apply2D_AVX_Dispatch = Apply2D_AVX<false, true>;
      Apply3D2_AVX_Dispatch = Apply3D2_AVX<false, true>;
      Apply3D3_AVX_Dispatch = Apply3D3_AVX<false, true>;
      Apply3D4_AVX_Dispatch = Apply3D4_AVX<false, true>;
      Apply3D5_AVX_Dispatch = Apply3D5_AVX<false, true>;

      Apply2D_AVX512_Dispatch = Apply2D_AVX512<false, true>;
      Apply3D2_AVX512_Dispatch = Apply3D2_AVX512<false, true>;
      Apply3D3_AVX512_Dispatch = Apply3D3_AVX512<false, true>;
      Apply3D4_AVX512_Dispatch = Apply3D4_AVX512<false, true>;
      Apply3D5_AVX512_Dispatch = Apply3D5_AVX512<false, true>;
    }
    else if (degrid == 0 && pfactor == 0) {
      Apply2D_C_Dispatch = Apply2D_C<false, false>;;
      Apply3D2_C_Dispatch = Apply3D2_C<false, false>;
      Apply3D3_C_Dispatch = Apply3D3_C<false, false>;
      Apply3D4_C_Dispatch = Apply3D4_C<false, false>;
      Apply3D5_C_Dispatch = Apply3D5_C<false, false>;

      Apply2D_SSE2_Dispatch = Apply2D_SSE2<false, false>;
      Apply3D2_SSE2_Dispatch = Apply3D2_SSE2<false, false>;
      Apply3D3_SSE2_Dispatch = Apply3D3_SSE2<false, false>;
      Apply3D4_SSE2_Dispatch = Apply3D4_SSE2<false, false>;
      Apply3D5_SSE2_Dispatch = Apply3D5_SSE2<false, false>;

      Apply2D_AVX_Dispatch = Apply2D_AVX<false, false>;
      Apply3D2_AVX_Dispatch = Apply3D2_AVX<false, false>;
      Apply3D3_AVX_Dispatch = Apply3D3_AVX<false, false>;
      Apply3D4_AVX_Dispatch = Apply3D4_AVX<false, false>;
      Apply3D5_AVX_Dispatch = Apply3D5_AVX<false, false>;

      Apply2D_AVX512_Dispatch = Apply2D_AVX512<false, false>;
      Apply3D2_AVX512_Dispatch = Apply3D2_AVX512<false, false>;
      Apply3D3_AVX512_Dispatch = Apply3D3_AVX512<false, false>;
      Apply3D4_AVX512_Dispatch = Apply3D4_AVX512<false, false>;
      Apply3D5_AVX512_Dispatch = Apply3D5_AVX512<false, false>;
    }
    else if (degrid != 0 && pfactor != 0) {
      Apply2D_C_Dispatch = Apply2D_C<true, true>;;
      Apply3D2_C_Dispatch = Apply3D2_C<true, true>;
      Apply3D3_C_Dispatch = Apply3D3_C<true, true>;
      Apply3D4_C_Dispatch = Apply3D4_C<true, true>;
      Apply3D5_C_Dispatch = Apply3D5_C<true, true>;

      Apply2D_SSE2_Dispatch = Apply2D_SSE2<true, true>;
      Apply3D2_SSE2_Dispatch = Apply3D2_SSE2<true, true>;
      Apply3D3_SSE2_Dispatch = Apply3D3_SSE2<true, true>;
      Apply3D4_SSE2_Dispatch = Apply3D4_SSE2<true, true>;
      Apply3D5_SSE2_Dispatch = Apply3D5_SSE2<true, true>;

      Apply2D_AVX_Dispatch = Apply2D_AVX<true, true>;
      Apply3D2_AVX_Dispatch = Apply3D2_AVX<true, true>;
      Apply3D3_AVX_Dispatch = Apply3D3_AVX<true, true>;
      Apply3D4_AVX_Dispatch = Apply3D4_AVX<true, true>;
      Apply3D5_AVX_Dispatch = Apply3D5_AVX<true, true>;

      Apply2D_AVX512_Dispatch = Apply2D_AVX512<true, true>;
      Apply3D2_AVX512_Dispatch = Apply3D2_AVX512<true, true>;
      Apply3D3_AVX512_Dispatch = Apply3D3_AVX512<true, true>;
      Apply3D4_AVX512_Dispatch = Apply3D4_AVX512<true, true>;
      Apply3D5_AVX512_Dispatch = Apply3D5_AVX512<true, true>;
    }
    else if (degrid == 0 && pfactor != 0) {
      Apply2D_C_Dispatch = Apply2D_C<true, false>;;
      Apply3D2_C_Dispatch = Apply3D2_C<true, false>;
      Apply3D3_C_Dispatch = Apply3D3_C<true, false>;
      Apply3D4_C_Dispatch = Apply3D4_C<true, false>;
      Apply3D5_C_Dispatch = Apply3D5_C<true, false>;

      Apply2D_SSE2_Dispatch = Apply2D_SSE2<true, false>;
      Apply3D2_SSE2_Dispatch = Apply3D2_SSE2<true, false>;
      Apply3D3_SSE2_Dispatch = Apply3D3_SSE2<true, false>;
      Apply3D4_SSE2_Dispatch = Apply3D4_SSE2<true, false>;
      Apply3D5_SSE2_Dispatch = Apply3D5_SSE2<true, false>;

      Apply2D_AVX_Dispatch = Apply2D_AVX<true, false>;
      Apply3D2_AVX_Dispatch = Apply3D2_AVX<true, false>;
      Apply3D3_AVX_Dispatch = Apply3D3_AVX<true, false>;
      Apply3D4_AVX_Dispatch = Apply3D4_AVX<true, false>;
      Apply3D5_AVX_Dispatch = Apply3D5_AVX<true, false>;

      Apply2D_AVX512_Dispatch = Apply2D_AVX512<true, false>;
      Apply3D2_AVX512_Dispatch = Apply3D2_AVX512<true, false>;
      Apply3D3_AVX512_Dispatch = Apply3D3_AVX512<true, false>;
      Apply3D4_AVX512_Dispatch = Apply3D4_AVX512<true, false>;
      Apply3D5_AVX512_Dispatch = Apply3D5_AVX512<true, false>;
    }

    if (degrid != 0) {
      Sharpen_C_Dispatch = Sharpen_C<true>;
      Sharpen_SSE2_Dispatch = Sharpen_SSE2<true>;
      Sharpen_AVX_Dispatch = Sharpen_AVX<true>;
      Sharpen_AVX512_Dispatch = Sharpen_AVX512<true>;
    }
    else {
      Sharpen_C_Dispatch = Sharpen_C<false>;
      Sharpen_SSE2_Dispatch = Sharpen_SSE2<false>;
      Sharpen_AVX_Dispatch = Sharpen_AVX<false>;
      Sharpen_AVX512_Dispatch = Sharpen_AVX512<false>;
    }

    if (pfactor != 0) {
      Kalman_C_Dispatch = Kalman_C<true>;
      Kalman_SSE2_Dispatch = Kalman_SSE2<true>;
      Kalman_AVX_Dispatch = Kalman_AVX<true>;
      Kalman_AVX512_Dispatch = Kalman_AVX512<true>;
    }
    else {
      Kalman_C_Dispatch = Kalman_C<false>;
      Kalman_SSE2_Dispatch = Kalman_SSE2<false>;
      Kalman_AVX_Dispatch = Kalman_AVX<false>;
      Kalman_AVX512_Dispatch = Kalman_AVX512<false>;
    }

    switch(bt) {
      case 2:
        Apply3D_C_Dispatch = Apply3D2_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D2_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D2_AVX_Dispatch;
        Apply3D_AVX512_Dispatch = Apply3D2_AVX512_Dispatch;
        break;
      case 3:
        Apply3D_C_Dispatch = Apply3D3_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D3_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D3_AVX_Dispatch;
        Apply3D_AVX512_Dispatch = Apply3D3_AVX512_Dispatch;
        break;
      case 4:
        Apply3D_C_Dispatch = Apply3D4_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D4_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D4_AVX_Dispatch;
        Apply3D_AVX512_Dispatch = Apply3D4_AVX512_Dispatch;
        break;
      case 5:
        Apply3D_C_Dispatch = Apply3D5_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D5_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D5_AVX_Dispatch;
        Apply3D_AVX512_Dispatch = Apply3D5_AVX512_Dispatch;
        break;
    }

    Apply2D = Apply2D_C_Dispatch;
    Apply3D = Apply3D_C_Dispatch;
    Sharpen = Sharpen_C_Dispatch;
    Kalman = Kalman_C_Dispatch;

    // We actually only used SSE code.
    // Let's try SSE and if it breaks on pure SSE we'll change it to SSE2.
    if ((CPUFlags & CPUF_SSE) && (opt <= 0 || opt > 1)) {
      Apply2D = Apply2D_SSE2_Dispatch;
      Apply3D = Apply3D_SSE2_Dispatch;
      Sharpen = Sharpen_SSE2_Dispatch;
      Kalman = Kalman_SSE2_Dispatch;
    }
    if ((CPUFlags & CPUF_AVX) && (opt <= 0 || opt > 2)) {
      Apply2D = Apply2D_AVX_Dispatch;
      Apply3D = Apply3D_AVX_Dispatch;
      Sharpen = Sharpen_AVX_Dispatch;
      Kalman = Kalman_AVX_Dispatch;
    }
    if ((CPUFlags & CPUF_AVX512F) && (opt < 0 || opt > 3)) {
      Apply2D = Apply2D_AVX512_Dispatch;
      Apply3D = Apply3D_AVX512_Dispatch;
      Sharpen = Sharpen_AVX512_Dispatch;
      Kalman = Kalman_AVX512_Dispatch;
    }
  }
};

#endif
