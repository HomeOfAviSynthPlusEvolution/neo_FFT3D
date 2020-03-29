#ifndef __FFT3DENGINE_H__
#define __FFT3DENGINE_H__

#include "functions.h"
#include "helper.h"
#include "info.h"
#include "wrapper/avs_filter.hpp"
#include "cache.hpp"
#include <atomic>

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
  int ncpu; // max number of threads
  int l, t, r, b; // cropping
};

template <typename Interface>
class FFT3DEngine {
  EngineParams* ep;
  Interface* super;
  int plane; // color plane

  // additional parameterss
  float *in;
  fftwf_complex *out, *outprev, *outnext, *outtemp, *outprev2, *outnext2;
  fftwf_complex *outrez, *gridsample; //v1.8
  fftwf_plan plan, planinv, plan1;
  int nox, noy;
  int outwidth;
  int outpitch; //v.1.7

  int outsize;
  int howmanyblocks;

  int ndim[2];
  int inembed[2];
  int onembed[2];

  float *wanxl; // analysis
  float *wanxr;
  float *wanyl;
  float *wanyr;

  float *wsynxl; // synthesis
  float *wsynxr;
  float *wsynyl;
  float *wsynyr;

  float *wsharpen;
  float *wdehalo;

  int nlast;// frame number at last step, PF: multithread warning, used for cacheing when sequential access detected
  int btcurlast;  //v1.7 to prevent multiple Pattern2Dto3D for the same btcurrent. btcurrent can change and may differ from bt for e.g. first/last frame

  fftwf_complex *outLast, *covar, *covarProcess;
  float sigmaSquaredNoiseNormed;
  float sigmaSquaredNoiseNormed2D;
  float sigmaNoiseNormed2D;
  float sigmaMotionNormed;
  float sigmaSquaredSharpenMinNormed;
  float sigmaSquaredSharpenMaxNormed;
  float ht2n; // halo threshold squared normed
  float norm; // normalization factor

  BYTE *coverbuf; //  block buffer covering the frame without remainders (with sufficient width and heigth)
  int coverwidth;
  int coverheight;
  int coverpitch;

  int mirw; // mirror width for padding
  int mirh; // mirror height for padding

  float *mean;

  float *pwin;
  float *pattern2d;
  float *pattern3d;
  bool isPatternSet;
  float psigma;
  char *messagebuf;

  // added in v.0.9 for delayed FFTW3.DLL loading
  struct FFTFunctionPointers fftfp;

  int CPUFlags;

  // avs+
  int pixelsize;
  int bits_per_pixel;

  int cachesize;//v1.8
  cache<fftwf_complex> *fftcache;

  int _instance_id; // debug unique id
  std::atomic<bool> reentrancy_check;

  struct FilterFunctionPointers ffp;

  bool isRGB;

  template<typename pixel_t, int bits_per_pixel, bool chroma>
  void do_InitOverlapPlane(float * inp, const BYTE *srcp, int src_pitch);

  void InitOverlapPlane(float * inp, const BYTE *srcp, int src_pitch, bool chroma);

  template<typename pixel_t, int bits_per_pixel, bool chroma>
  void do_DecodeOverlapPlane(float *in, float norm, BYTE *dstp, int dst_pitch);

  void DecodeOverlapPlane(float *in, float norm, BYTE *dstp, int dst_pitch, bool chroma);

  void load_buffer(int plane, typename Interface::AFrame &src, byte *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, int bits_per_pixel);
  void store_buffer(int plane, const byte *coverbuf, int coverwidth, int coverheight, int coverpitch, typename Interface::AFrame &dst, int mirw, int mirh, bool interlaced, int bits_per_pixel);

public:
  FFT3DEngine(Interface*, EngineParams, int);
  ~FFT3DEngine();

  typename Interface::AFrame GetFrame(int n);

};

#endif
