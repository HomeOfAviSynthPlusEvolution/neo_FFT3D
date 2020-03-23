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

  float lowlimit;

  template <bool pattern>
  void wiener_factor_3d(float &dr, float &di) {
    float _psd = dr * dr + di * di + 1e-15f; // power spectrum density 0
    float _wiener_factor = MAX((_psd - (pattern ? pattern3d[w] : sigmaSquaredNoiseNormed) ) / _psd, lowlimit); // limited Wiener filter
    dr *= _wiener_factor;
    di *= _wiener_factor;
  }

};

template<typename ... T>
inline void loop_wrapper_C_advance(int pitch, fftwf_complex* &fft_data, T&&... other_fft_data) {
  fft_data += pitch;
  loop_wrapper_C_advance(pitch, other_fft_data...);
}

inline void loop_wrapper_C_advance(int pitch) {}

// Note, MSVC requires parameter pack to be the last to work
template<typename ... T, typename Func>
inline void loop_wrapper_C(Func f, SharedFunctionParams sfp, fftwf_complex* &outcur, T&&... fft_data) {
  LambdaFunctionParams lfp;

  lfp.lowlimit = (sfp.beta - 1) / sfp.beta;
  lfp.sigmaSquaredNoiseNormed = sfp.sigmaSquaredNoiseNormed;
  for (lfp.block = 0; lfp.block < sfp.howmanyblocks; lfp.block++)
  {
    // Pattern
    lfp.pattern2d = sfp.pattern2d;
    lfp.pattern3d = sfp.pattern3d;
    // Wiener
    lfp.wsharpen = sfp.wsharpen;
    lfp.wdehalo = sfp.wdehalo;
    // Grid
    lfp.gridsample = sfp.gridsample;
    lfp.gridfraction = sfp.degrid * outcur[0][0] / lfp.gridsample[0][0];

    for (lfp.h = 0; lfp.h < sfp.bh; lfp.h++)
    {
      for (lfp.w = 0; lfp.w < sfp.outwidth; lfp.w++)
      {
        f(lfp);
      }
      // Data
      outcur += sfp.outpitch;
      loop_wrapper_C_advance(sfp.outpitch, fft_data...);
      // Pattern
      lfp.pattern2d += sfp.outpitch;
      lfp.pattern3d += sfp.outpitch;
      // Wiener
      lfp.wsharpen += sfp.outpitch;
      lfp.wdehalo += sfp.outpitch;
      // Grid
      lfp.gridsample += sfp.outpitch;
    }
  }
}

#endif
