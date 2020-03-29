#ifndef __CODE_IMPL_H__
#define __CODE_IMPL_H__

#include "common.h"

struct SharedFunctionParams {
    int outwidth;
    int outpitch;
    int bh;
    int howmanyblocks;
    float sigmaSquaredNoiseNormed;
    float pfactor;
    float *pattern2d;
    float *pattern3d;
    float beta;
    float degrid;
    fftwf_complex *gridsample;
    float sharpen;
    float sigmaSquaredSharpenMinNormed;
    float sigmaSquaredSharpenMaxNormed;
    float *wsharpen;
    float dehalo;
    float *wdehalo;
    float ht2n;
    fftwf_complex *covar;
    fftwf_complex *covarProcess;
    float sigmaSquaredNoiseNormed2D;
    float kratio2;
};

// C
  template <bool pattern, bool degrid> void Apply2D_C(fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D2_C(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D3_C(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D4_C(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D5_C(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool degrid> void Sharpen_C(fftwf_complex *, SharedFunctionParams);
  template <bool pattern> void Kalman_C(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

// SSE2
  template <bool pattern, bool degrid> void Apply2D_SSE2(fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D2_SSE2(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D3_SSE2(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D4_SSE2(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool pattern, bool degrid> void Apply3D5_SSE2(fftwf_complex **, fftwf_complex *, SharedFunctionParams);
  template <bool degrid> void Sharpen_SSE2(fftwf_complex *, SharedFunctionParams);
  template <bool pattern> void Kalman_SSE2(fftwf_complex *, fftwf_complex *, SharedFunctionParams);

#endif
