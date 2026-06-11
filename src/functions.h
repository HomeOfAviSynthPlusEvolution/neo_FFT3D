#pragma once
#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "fft3d_common.h"
#include "code_impl/code_impl.h"
#include "core/cpu/core_hwy.h"

struct FilterFunctionPointers {
  typedef void (*Apply3D_PROC)(fftwf_complex **, fftwf_complex *, SharedFunctionParams);

  void (*Apply2D_C_Dispatch)(fftwf_complex *, SharedFunctionParams);
  Apply3D_PROC Apply3D_C_Dispatch;
  Apply3D_PROC Apply3D2_C_Dispatch;
  Apply3D_PROC Apply3D3_C_Dispatch;
  Apply3D_PROC Apply3D4_C_Dispatch;
  Apply3D_PROC Apply3D5_C_Dispatch;
  void (*Sharpen_C_Dispatch)(fftwf_complex *, SharedFunctionParams);
  void (*Kalman_C_Dispatch)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  void (*Apply2D)(fftwf_complex *, SharedFunctionParams);
  Apply3D_PROC Apply3D;
  void (*Sharpen)(fftwf_complex *, SharedFunctionParams);
  void (*Kalman)(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

  void set_ffp(int CPUFlags, float degrid, float pfactor, int bt, int opt)
  {
    (void)CPUFlags;

    if (degrid != 0 && pfactor == 0) {
      Apply2D_C_Dispatch = Apply2D_C<false, true>;
      Apply3D2_C_Dispatch = Apply3D2_C<false, true>;
      Apply3D3_C_Dispatch = Apply3D3_C<false, true>;
      Apply3D4_C_Dispatch = Apply3D4_C<false, true>;
      Apply3D5_C_Dispatch = Apply3D5_C<false, true>;
    }
    else if (degrid == 0 && pfactor == 0) {
      Apply2D_C_Dispatch = Apply2D_C<false, false>;
      Apply3D2_C_Dispatch = Apply3D2_C<false, false>;
      Apply3D3_C_Dispatch = Apply3D3_C<false, false>;
      Apply3D4_C_Dispatch = Apply3D4_C<false, false>;
      Apply3D5_C_Dispatch = Apply3D5_C<false, false>;
    }
    else if (degrid != 0 && pfactor != 0) {
      Apply2D_C_Dispatch = Apply2D_C<true, true>;
      Apply3D2_C_Dispatch = Apply3D2_C<true, true>;
      Apply3D3_C_Dispatch = Apply3D3_C<true, true>;
      Apply3D4_C_Dispatch = Apply3D4_C<true, true>;
      Apply3D5_C_Dispatch = Apply3D5_C<true, true>;
    }
    else if (degrid == 0 && pfactor != 0) {
      Apply2D_C_Dispatch = Apply2D_C<true, false>;
      Apply3D2_C_Dispatch = Apply3D2_C<true, false>;
      Apply3D3_C_Dispatch = Apply3D3_C<true, false>;
      Apply3D4_C_Dispatch = Apply3D4_C<true, false>;
      Apply3D5_C_Dispatch = Apply3D5_C<true, false>;
    }

    if (degrid != 0) {
      Sharpen_C_Dispatch = Sharpen_C<true>;
    }
    else {
      Sharpen_C_Dispatch = Sharpen_C<false>;
    }

    if (pfactor != 0) {
      Kalman_C_Dispatch = Kalman_C<true>;
    }
    else {
      Kalman_C_Dispatch = Kalman_C<false>;
    }

    switch(bt) {
      case 2:
        Apply3D_C_Dispatch = Apply3D2_C_Dispatch;
        break;
      case 3:
        Apply3D_C_Dispatch = Apply3D3_C_Dispatch;
        break;
      case 4:
        Apply3D_C_Dispatch = Apply3D4_C_Dispatch;
        break;
      case 5:
        Apply3D_C_Dispatch = Apply3D5_C_Dispatch;
        break;
      default:
        Apply3D_C_Dispatch = Apply3D2_C_Dispatch;
        break;
    }

    Apply2D = Apply2D_C_Dispatch;
    Apply3D = Apply3D_C_Dispatch;
    Sharpen = Sharpen_C_Dispatch;
    Kalman = Kalman_C_Dispatch;

    if (opt != 1) {
      Apply2D = neo_fft3d::cpu::Apply2D_Hwy;
      Apply3D = neo_fft3d::cpu::Apply3D_Hwy;
      Sharpen = neo_fft3d::cpu::Sharpen_Hwy;
      Kalman = neo_fft3d::cpu::Kalman_Hwy;
    }
  }
};

#endif
