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
  // Wiener
    void (*ApplyWiener2D)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D2)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D3)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D4)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D5)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);

  // Wiener Degrid
    void (*ApplyWiener2D_degrid)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D2_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D3_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D4_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyWiener3D5_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);

  // Pattern
    void (*ApplyPattern2D)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D2)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D3)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D4)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D5)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);

  // Pattern Degrid
    void (*ApplyPattern2D_degrid)(
        fftwf_complex *outcur,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D2_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D3_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D4_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        SharedFunctionParams sfp);
    void (*ApplyPattern3D5_degrid)(
        fftwf_complex *outcur,
        fftwf_complex *outprev2,
        fftwf_complex *outprev,
        fftwf_complex *outnext,
        fftwf_complex *outnext2,
        SharedFunctionParams sfp);

  // Dispatcher
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

  // Kalman
    void (*ApplyKalman)
        (fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
    void (*ApplyKalmanPattern)
        (fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float *covarNoiseNormed, float kratio2);

  // Sharpen
    void (*Sharpen)(
      fftwf_complex *outcur,
      SharedFunctionParams sfp);
    void (*Sharpen_degrid)(
      fftwf_complex *outcur,
      SharedFunctionParams sfp);
  // Sharpen Dispatcher
    void (*Sharpen_dispatcher)(
      fftwf_complex *outcur,
      SharedFunctionParams sfp);

  void set_ffp(int CPUFlags)
  {
    ApplyWiener2D = ApplyWiener2D_C;
    ApplyWiener3D2 = ApplyWiener3D2_C;
    ApplyWiener3D3 = ApplyWiener3D3_C;
    ApplyWiener3D4 = ApplyWiener3D4_C;
    ApplyWiener3D5 = ApplyWiener3D5_C;
    ApplyWiener2D_degrid = ApplyWiener2D_degrid_C;
    ApplyWiener3D2_degrid = ApplyWiener3D2_degrid_C;
    ApplyWiener3D3_degrid = ApplyWiener3D3_degrid_C;
    ApplyWiener3D4_degrid = ApplyWiener3D4_degrid_C;
    ApplyWiener3D5_degrid = ApplyWiener3D5_degrid_C;
    ApplyWiener3D3_degrid = ApplyWiener3D3_degrid_C;
    ApplyPattern2D = ApplyPattern2D_C<false>;
    ApplyPattern3D2 = ApplyPattern3D2_C<false>;
    ApplyPattern3D3 = ApplyPattern3D3_C<false>;
    ApplyPattern3D4 = ApplyPattern3D4_C<false>;
    ApplyPattern3D5 = ApplyPattern3D5_C<false>;
    ApplyPattern2D_degrid = ApplyPattern2D_C<true>;
    ApplyPattern3D2_degrid = ApplyPattern3D2_C<true>;
    ApplyPattern3D3_degrid = ApplyPattern3D3_C<true>;
    ApplyPattern3D4_degrid = ApplyPattern3D4_C<true>;
    ApplyPattern3D5_degrid = ApplyPattern3D5_C<true>;
    ApplyKalman = ApplyKalman_C;
    ApplyKalmanPattern = ApplyKalmanPattern_C;
    Sharpen = Sharpen_C;
    Sharpen_degrid = Sharpen_degrid_C;

    #ifndef X86_64
      if (CPUFlags & CPUF_SSE) {
        ApplyWiener3D3 = ApplyWiener3D3_SSE;
        ApplyWiener3D3_degrid = ApplyWiener3D3_degrid_SSE_simd;
        ApplyWiener3D4_degrid = ApplyWiener3D4_degrid_SSE;
        ApplyPattern3D2 = ApplyPattern3D2_SSE;
        ApplyPattern3D3 = ApplyPattern3D3_SSE;
        ApplyPattern3D3_degrid = ApplyPattern3D3_degrid_SSE;
        ApplyPattern3D4_degrid = ApplyPattern3D4_degrid_SSE;
        Sharpen = Sharpen_SSE;
      }
    #endif

    if (CPUFlags & CPUF_SSE2) { // 170302 simd, SSE2
      ApplyWiener3D2 = ApplyWiener3D2_SSE2;
      ApplyWiener3D3_degrid = ApplyWiener3D3_degrid_SSE2;
      ApplyKalman = ApplyKalman_SSE2_simd;
      Sharpen_degrid = Sharpen_degrid_SSE2;
    }
  }

  void set_ffp2(float degrid, float pfactor)
  {
    if (degrid != 0 && pfactor == 0) {
      // Default Dispatcher
      Apply2D = ApplyWiener2D_degrid;
      Apply3D2 = ApplyWiener3D2_degrid;
      Apply3D3 = ApplyWiener3D3_degrid;
      Apply3D4 = ApplyWiener3D4_degrid;
      Apply3D5 = ApplyWiener3D5_degrid;
      Apply3D3 = ApplyWiener3D3_degrid;
    }
    else if (degrid == 0 && pfactor == 0) {
      Apply2D = ApplyWiener2D;
      Apply3D2 = ApplyWiener3D2;
      Apply3D3 = ApplyWiener3D3;
      Apply3D4 = ApplyWiener3D4;
      Apply3D5 = ApplyWiener3D5;
      Apply3D3 = ApplyWiener3D3;
    }
    else if (degrid != 0 && pfactor != 0) {
      Apply2D = ApplyPattern2D_degrid;
      Apply3D2 = ApplyPattern3D2_degrid;
      Apply3D3 = ApplyPattern3D3_degrid;
      Apply3D4 = ApplyPattern3D4_degrid;
      Apply3D5 = ApplyPattern3D5_degrid;
      Apply3D3 = ApplyPattern3D3_degrid;
    }
    else if (degrid == 0 && pfactor != 0) {
      Apply2D = ApplyPattern2D;
      Apply3D2 = ApplyPattern3D2;
      Apply3D3 = ApplyPattern3D3;
      Apply3D4 = ApplyPattern3D4;
      Apply3D5 = ApplyPattern3D5;
      Apply3D3 = ApplyPattern3D3;
    }

    if (degrid != 0) {
      Sharpen_dispatcher = Sharpen_degrid;
    }
    else {
      Sharpen_dispatcher = Sharpen;
    }
  }
};

#endif
