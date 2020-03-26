#pragma once
#include "common.h"

void GetAndSubtactMean(float *in, int howmanyblocks, int bw, int bh, int ow, int oh, float *wxl, float *wxr, float *wyl, float *wyr, float *mean);
void RestoreMean(float *in, int howmanyblocks, int bw, int bh, float *mean);
void FindPatternBlock(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample);
void SetPattern(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample);
void PutPatternOnly(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py);
void Pattern2Dto3D(float *pattern2d, int bh, int outwidth, int outpitch, float mult, float *pattern3d);
void Copyfft(fftwf_complex *outrez, fftwf_complex *outprev, int outsize);
void SortCache(int *cachewhat, fftwf_complex **cachefft, int cachesize, int cachestart, int cachestartold);
// void CopyFrame(PVideoFrame &src, PVideoFrame &dst, VideoInfo vi, int planeskip);
void fill_complex(fftwf_complex *plane, int outsize, float realvalue, float imgvalue);
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitch, float norm, float *pattern2d);

template<typename pixel_t>
void PlanarPlaneToCoverbuf(const pixel_t *srcp, int src_width, int src_height, int src_pitch, pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced);
template<typename pixel_t>
void CoverbufToPlanarPlane(const pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, pixel_t *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced);

#define DECLARE(pixel_t) \
  template void PlanarPlaneToCoverbuf<pixel_t>(const pixel_t *srcp, int src_width, int src_height, int src_pitch, pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced);\
  template void CoverbufToPlanarPlane<pixel_t>(const pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, pixel_t *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced);\

DECLARE(uint8_t)
DECLARE(uint16_t)
DECLARE(float)
#undef DECLARE
