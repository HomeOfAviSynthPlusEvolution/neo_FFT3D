/*
 * Copyright 2020 Xinyue Lu
 *
 * C code wrapper.
 *
 */

#ifndef __CODE_IMPL_C_H__
#define __CODE_IMPL_C_H__

#include "code_impl.h"

struct LambdaFunctionParams {
  // Progress
  int block;
  int h;
  int w;
  // Pattern
  float *pattern2d;
  float *pattern3d;
  // Wiener
  float sigmaSquaredNoiseNormed;
  float *wsharpen;
  float *wdehalo;
  // Grid
  fftwf_complex *gridsample;
  float gridfraction = 0.0f;
  // Kalman
  fftwf_complex *covar;
  fftwf_complex *covarProcess;

  float lowlimit;

  template <bool pattern>
  inline void wiener_factor_3d(float &dr, float &di) {
    float _psd = dr * dr + di * di + 1.0e-15f; // power spectrum density 0
    float _wiener_factor = MAX((_psd - (pattern ? pattern3d[w] : sigmaSquaredNoiseNormed) ) / _psd, lowlimit); // limited Wiener filter
    dr *= _wiener_factor;
    di *= _wiener_factor;
  }

};

template<typename ... T, typename Func>
inline void loop_wrapper_C(fftwf_complex** in, fftwf_complex* &out, SharedFunctionParams sfp, Func f) {
  LambdaFunctionParams lfp;

  lfp.lowlimit = (sfp.beta - 1) / sfp.beta;
  lfp.sigmaSquaredNoiseNormed = sfp.sigmaSquaredNoiseNormed;
  // Kalman
  lfp.covar = sfp.covar.fftw_data();
  lfp.covarProcess = sfp.covarProcess.fftw_data();

  for (lfp.block = 0; lfp.block < sfp.howmanyblocks; lfp.block++)
  {
    // Pattern
    lfp.pattern2d = sfp.pattern2d.data_handle();
    lfp.pattern3d = sfp.pattern3d.data_handle();
    // Wiener
    lfp.wsharpen = sfp.wsharpen.data_handle();
    lfp.wdehalo = sfp.wdehalo.data_handle();
    // Grid
    lfp.gridsample = sfp.gridsample.fftw_data();
    lfp.gridfraction = sfp.degrid * in[2][0][0] / lfp.gridsample[0][0];

    for (lfp.h = 0; lfp.h < sfp.bh; lfp.h++)
    {
      for (lfp.w = 0; lfp.w < sfp.outwidth; lfp.w++)
      {
        f(lfp);
      }
      // Data
      if (in[0]) in[0] += sfp.outpitch;
      if (in[1]) in[1] += sfp.outpitch;
      if (in[2]) in[2] += sfp.outpitch;
      if (in[3]) in[3] += sfp.outpitch;
      if (in[4]) in[4] += sfp.outpitch;
      out += sfp.outpitch;
      // Pattern
      lfp.pattern2d += sfp.outpitch;
      lfp.pattern3d += sfp.outpitch;
      // Wiener
      lfp.wsharpen += sfp.outpitch;
      lfp.wdehalo += sfp.outpitch;
      // Grid
      lfp.gridsample += sfp.outpitch;
      // Kalman
      lfp.covar += sfp.outpitch;
      lfp.covarProcess += sfp.outpitch;
    }
  }
}

#endif
