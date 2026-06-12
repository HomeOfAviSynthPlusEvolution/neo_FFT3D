#pragma once

#include "common/aligned_vector.hpp"

struct EngineFormat {
  bool IsFamilyYUV {false};
  bool IsFamilyRGB {false};
  bool IsFamilyGray {false};
  int Planes {0};
  int BytesPerSample {1};
  int BitsPerSample {8};
  int SSW {0};
  int SSH {0};
};

struct EngineVideoInfo {
  int Width {0};
  int Height {0};
  int Frames {0};
  int FPSNum {0};
  int FPSDen {1};
  EngineFormat Format;
};

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

  EngineVideoInfo vi;
  bool IsChroma {false};

  int framewidth {0}; // in pixels, not bytes
  int frameheight {0};
  int framepitch {0}; // in pixels, not bytes
  int framepitch_dst {0}; // in pixels, not bytes
};

struct IOParams {
  int nox {0};
  int noy {0};

  AlignedVector<float> wanxl;
  AlignedVector<float> wanxr;
  AlignedVector<float> wanyl;
  AlignedVector<float> wanyr;
  AlignedVector<float> wsynxl;
  AlignedVector<float> wsynxr;
  AlignedVector<float> wsynyl;
  AlignedVector<float> wsynyr;
};
