#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "common.h"
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

  // SSE2
    void (*Apply2D_AVX_Dispatch)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D_AVX_Dispatch;
    Apply3D_PROC Apply3D2_AVX_Dispatch;
    Apply3D_PROC Apply3D3_AVX_Dispatch;
    Apply3D_PROC Apply3D4_AVX_Dispatch;
    Apply3D_PROC Apply3D5_AVX_Dispatch;
    void (*Sharpen_AVX_Dispatch)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman_AVX_Dispatch)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  // Dispatcher -> [C / SSE2 / AVX]
    void (*Apply2D)(fftwf_complex *, SharedFunctionParams);
    Apply3D_PROC Apply3D;
    void (*Sharpen)(fftwf_complex *, SharedFunctionParams);
    void (*Kalman)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  void set_ffp(int CPUFlags, float degrid, float pfactor, int bt)
  {
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
    }

    if (degrid != 0) {
      Sharpen_C_Dispatch = Sharpen_C<true>;
      Sharpen_SSE2_Dispatch = Sharpen_SSE2<true>;
      Sharpen_AVX_Dispatch = Sharpen_AVX<true>;
    }
    else {
      Sharpen_C_Dispatch = Sharpen_C<false>;
      Sharpen_SSE2_Dispatch = Sharpen_SSE2<false>;
      Sharpen_AVX_Dispatch = Sharpen_AVX<false>;
    }

    if (pfactor != 0) {
      Kalman_C_Dispatch = Kalman_C<true>;
      Kalman_SSE2_Dispatch = Kalman_SSE2<true>;
      Kalman_AVX_Dispatch = Kalman_AVX<true>;
    }
    else {
      Kalman_C_Dispatch = Kalman_C<false>;
      Kalman_SSE2_Dispatch = Kalman_SSE2<false>;
      Kalman_AVX_Dispatch = Kalman_AVX<false>;
    }

    switch(bt) {
      case 2:
        Apply3D_C_Dispatch = Apply3D2_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D2_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D2_AVX_Dispatch;
        break;
      case 3:
        Apply3D_C_Dispatch = Apply3D3_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D3_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D3_AVX_Dispatch;
        break;
      case 4:
        Apply3D_C_Dispatch = Apply3D4_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D4_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D4_AVX_Dispatch;
        break;
      case 5:
        Apply3D_C_Dispatch = Apply3D5_C_Dispatch;
        Apply3D_SSE2_Dispatch = Apply3D5_SSE2_Dispatch;
        Apply3D_AVX_Dispatch = Apply3D5_AVX_Dispatch;
        break;
    }

    Apply2D = Apply2D_C_Dispatch;
    Apply3D = Apply3D_C_Dispatch;
    Sharpen = Sharpen_C_Dispatch;
    Kalman = Kalman_C_Dispatch;

    // We actually only used SSE code.
    // Let's try SSE and if it breaks on pure SSE we'll change it to SSE2.
    if (CPUFlags & CPUF_SSE) {
      Apply2D = Apply2D_SSE2_Dispatch;
      Apply3D = Apply3D_SSE2_Dispatch;
      Sharpen = Sharpen_SSE2_Dispatch;
      Kalman = Kalman_SSE2_Dispatch;
    }
    if (CPUFlags & CPUF_AVX) {
      Apply2D = Apply2D_AVX_Dispatch;
      Apply3D = Apply3D_AVX_Dispatch;
      Sharpen = Sharpen_AVX_Dispatch;
      Kalman = Kalman_AVX_Dispatch;
    }
  }
};

#endif
