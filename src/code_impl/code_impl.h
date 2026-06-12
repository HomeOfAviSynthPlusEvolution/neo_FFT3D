/*
 * Copyright 2020 Xinyue Lu
 *
 * Code implementation section interface.
 * This file exposes all internal functions and parameters to main file.
 *
 */

#ifndef __CODE_IMPL_H__
#define __CODE_IMPL_H__

#include "../fft3d_common.h"
#include "common/fft3d_views.hpp"

#pragma warning(disable:4101)

struct SharedFunctionParams {
    int outwidth;
    int outpitch;
    int bh;
    int howmanyblocks;
    float sigmaSquaredNoiseNormed;
    float pfactor;
    neo_fft3d::FloatPlaneView pattern2d;
    neo_fft3d::FloatPlaneView pattern3d;
    float beta;
    float degrid;
    neo_fft3d::ComplexBlockView gridsample;
    float sharpen;
    float sigmaSquaredSharpenMinNormed;
    float sigmaSquaredSharpenMaxNormed;
    neo_fft3d::FloatPlaneView wsharpen;
    float dehalo;
    neo_fft3d::FloatPlaneView wdehalo;
    float ht2n;
    neo_fft3d::ComplexBlockView covar;
    neo_fft3d::ComplexBlockView covarProcess;
    float sigmaSquaredNoiseNormed2D;
    float kratio2;
};

// C
  template <bool pattern, bool degrid> void Apply2D_C(fftwf_complex * /*out*/, SharedFunctionParams /*sfp*/);
  template <bool pattern, bool degrid> void Apply3D2_C(fftwf_complex ** /*in*/, fftwf_complex * /*out*/, SharedFunctionParams /*sfp*/);
  template <bool pattern, bool degrid> void Apply3D3_C(fftwf_complex ** /*in*/, fftwf_complex * /*out*/, SharedFunctionParams /*sfp*/);
  template <bool pattern, bool degrid> void Apply3D4_C(fftwf_complex ** /*in*/, fftwf_complex * /*out*/, SharedFunctionParams /*sfp*/);
  template <bool pattern, bool degrid> void Apply3D5_C(fftwf_complex ** /*in*/, fftwf_complex * /*out*/, SharedFunctionParams /*sfp*/);
  template <bool degrid> void Sharpen_C(fftwf_complex * /*out*/, SharedFunctionParams /*sfp*/);
  template <bool pattern> void Kalman_C(fftwf_complex * /*outcur*/, fftwf_complex * /*outLast*/, SharedFunctionParams /*sfp*/);

#endif
