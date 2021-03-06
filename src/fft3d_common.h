#pragma once

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>

#ifdef HAS_EXECUTION
  #include <execution>
#endif

#ifndef __cpp_lib_execution
  #undef ENABLE_PAR
#endif

#ifdef ENABLE_PAR
  #define PAR_POLICY std::execution::par
  #define SEQ_POLICY std::execution::seq
#else
  #define PAR_POLICY nullptr
  #define SEQ_POLICY nullptr
#endif

#include "fftwlite.h"
#include <ds_common.hpp>

typedef unsigned char byte;

#ifndef _WIN32
  #define wsprintf sprintf
  #define _aligned_malloc(a,b) aligned_alloc(b,a)
  #define _aligned_free(a) free(a)
#endif

#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

int GetCPUFlags();

struct EngineParams {
  float sigma; // noise level (std deviation) for high frequncies ***
  float beta; // relative noise margin for Wiener filter
  int bw, bh; // block width / height
  int bt; // block size  along time (mumber of frames), =0 for Kalman, >0 for Wiener
  int ow, oh; // overlap width / height
  float kratio; // threshold to sigma ratio for Kalman filter
  float sharpen; // sharpen factor (0 to 1 and above)
  float scutoff; // sharpen cufoff frequency (relative to max) - v1.7
  float svr; // sharpen vertical ratio (0 to 1 and above) - v.1.0
  float smin; // minimum limit for sharpen (prevent noise amplifying) - v.1.1  ***
  float smax; // maximum limit for sharpen (prevent oversharping) - v.1.1      ***
  bool measure; // fft optimal method
  bool interlaced;
  int wintype; // window type
  int pframe; // noise pattern frame number
  int px, py; // noise pattern window x / y position
  bool pshow; // show noise pattern
  float pcutoff; // pattern cutoff frequency (relative to max)
  float pfactor; // noise pattern denoise strength
  float sigma2; // noise level for middle frequencies           ***
  float sigma3; // noise level for low frequencies              ***
  float sigma4; // noise level for lowest (zero) frequencies    ***
  float degrid; // decrease grid
  float dehalo; // remove halo strength - v.1.9
  float hr; // halo radius - v1.9
  float ht; // halo threshold - v1.9
  int l, t, r, b; // cropping
  int opt;

  DSVideoInfo vi;
  bool IsChroma;

  int framewidth; // in pixels, not bytes
  int frameheight;
  int framepitch; // in pixels, not bytes
  int framepitch_dst; // in pixels, not bytes
};


struct IOParams {
  int nox, noy;

  // analysis
  float *wanxl, *wanxr, *wanyl, *wanyr;
  // synthesis
  float *wsynxl, *wsynxr, *wsynyl, *wsynyr;

  float *wsharpen;
  float *wdehalo;
};
