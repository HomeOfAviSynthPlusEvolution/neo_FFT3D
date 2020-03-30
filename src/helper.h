#ifndef __HELPER_H__
#define __HELPER_H__

#include "common.h"

void FindPatternBlock(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample);
void SetPattern(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample);
void PutPatternOnly(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py);
void Pattern2Dto3D(float *pattern2d, int bh, int outwidth, int outpitch, float mult, float *pattern3d);
// void CopyFrame(PVideoFrame &src, PVideoFrame &dst, VideoInfo vi, int planeskip);
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitch, float norm, float *pattern2d);

#endif
