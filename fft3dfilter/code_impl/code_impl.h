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
};

// Wiener
  void ApplyWiener2D_C(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);
  void ApplyWiener3D2_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    SharedFunctionParams sfp);
  void ApplyWiener3D3_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyWiener3D4_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyWiener3D5_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    fftwf_complex *outnext2,
    SharedFunctionParams sfp);

  void ApplyWiener3D2_SSE2(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    SharedFunctionParams sfp);

// Wiener Degrid
  void ApplyWiener2D_degrid_C(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);
  void ApplyWiener3D2_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    SharedFunctionParams sfp);
  void ApplyWiener3D3_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyWiener3D4_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyWiener3D5_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    fftwf_complex *outnext2,
    SharedFunctionParams sfp);

  void ApplyWiener3D3_degrid_SSE2(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);

// Pattern
  void ApplyPattern2D_C(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);
  void ApplyPattern3D2_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    SharedFunctionParams sfp);
  void ApplyPattern3D3_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyPattern3D4_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyPattern3D5_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    fftwf_complex *outnext2,
    SharedFunctionParams sfp);

// Pattern Degrid
  void ApplyPattern2D_degrid_C(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);
  void ApplyPattern3D2_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    SharedFunctionParams sfp);
  void ApplyPattern3D3_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyPattern3D4_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    SharedFunctionParams sfp);
  void ApplyPattern3D5_degrid_C(
    fftwf_complex *outcur,
    fftwf_complex *outprev2,
    fftwf_complex *outprev,
    fftwf_complex *outnext,
    fftwf_complex *outnext2,
    SharedFunctionParams sfp);

// Sharpen
  void Sharpen_C(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);
  void Sharpen_degrid_C(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);

  void Sharpen_degrid_SSE2(
    fftwf_complex *outcur,
    SharedFunctionParams sfp);

// declarations of filtering functions:
void ApplyKalman_SSE2_simd(fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
// SSE
void ApplyPattern3D2_SSE(fftwf_complex *out, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float * pattern3d, float beta);
void ApplyWiener3D3_SSE(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D3_SSE(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta);
void Sharpen_SSE(fftwf_complex *out, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n);
// C
void ApplyKalmanPattern_C(fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float *covarNoiseNormed, float kratio2);
void ApplyKalman_C(fftwf_complex *out, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
// degrid_SSE
void ApplyPattern3D3_degrid_SSE(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample);
void ApplyWiener3D4_degrid_SSE(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D4_degrid_SSE(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample);

#endif
