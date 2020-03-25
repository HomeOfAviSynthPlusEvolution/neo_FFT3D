#ifndef __FFT3DFILTER_H__
#define __FFT3DFILTER_H__

#include "common.h"
#include <avisynth.h>
#include "info.h"
#include <emmintrin.h>
#include <mmintrin.h> // _mm_empty
#include <algorithm>
#include <atomic>
#include "code_impl/code_impl.h"

struct FilterFunctionPointers {
  // C
    void (*Apply2D_C_Dispatch)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*Apply3D2_C_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*Apply3D3_C_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*Apply3D4_C_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*Apply3D5_C_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);
    void (*Sharpen_C_Dispatch)(
      fftwf_complex *outcur,
      SharedFunctionParams sfp);

  // SSE2
    void (*Apply2D_SSE2_Dispatch)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*Apply3D2_SSE2_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*Apply3D3_SSE2_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*Apply3D4_SSE2_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*Apply3D5_SSE2_Dispatch)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);
    void (*Sharpen_SSE2_Dispatch)(
      fftwf_complex *outcur,
      SharedFunctionParams sfp);

  // Dispatcher -> [C / SSE2 / AVX]
    void (*Apply2D)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*Apply3D2)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*Apply3D3)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*Apply3D4)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*Apply3D5)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);
    void (*Sharpen)(
      fftwf_complex *outcur,
      SharedFunctionParams sfp);

  // Kalman
    void (*ApplyKalman)
        (fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
    void (*ApplyKalmanPattern)
        (fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float *covarNoiseNormed, float kratio2);


  void set_ffp(int CPUFlags, float degrid, float pfactor)
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
    }

    if (degrid != 0) {
      Sharpen_C_Dispatch = Sharpen_C<true>;
      Sharpen_SSE2_Dispatch = Sharpen_SSE2<true>;
    }
    else {
      Sharpen_C_Dispatch = Sharpen_C<false>;
      Sharpen_SSE2_Dispatch = Sharpen_SSE2<false>;
    }

    ApplyKalman = ApplyKalman_C;
    ApplyKalmanPattern = ApplyKalmanPattern_C;

    Apply2D = Apply2D_C_Dispatch;
    Apply3D2 = Apply3D2_C_Dispatch;
    Apply3D3 = Apply3D3_C_Dispatch;
    Apply3D4 = Apply3D4_C_Dispatch;
    Apply3D5 = Apply3D5_C_Dispatch;
    Sharpen = Sharpen_C_Dispatch;

    if (CPUFlags & CPUF_SSE2) {
      Apply2D = Apply2D_SSE2_Dispatch;
      Apply3D2 = Apply3D2_SSE2_Dispatch;
      Apply3D3 = Apply3D3_SSE2_Dispatch;
      Apply3D4 = Apply3D4_SSE2_Dispatch;
      Apply3D5 = Apply3D5_SSE2_Dispatch;
      Sharpen = Sharpen_SSE2_Dispatch;
    }
  }
};

#endif
