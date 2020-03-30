/*
 * Copyright 2004-2007 A.G.Balakhnin aka Fizick
 * Copyright 2015 martin53
 * Copyright 2017-2019 Ferenc Pinter aka printerf
 * Copyright 2020 Xinyue Lu
 *
 * Some helper functions.
 *
 */

#include "helper.h"

//-------------------------------------------------------------------
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitch, float norm, float *pattern2d)
{
  // it is not fast, but called only in constructor
  float sigmacur;
  float ft2 = sqrt(0.5f) / 2; // frequency for sigma2
  float ft3 = sqrt(0.5f) / 4; // frequency for sigma3
  for (int h = 0; h < bh; h++)
  {
    for (int w = 0; w < outwidth; w++)
    {
      float fy = (bh - 2.0f*abs(h - bh / 2)) / bh; // normalized to 1
      float fx = (w*1.0f) / outwidth;  // normalized to 1
      float f = sqrt((fx*fx + fy*fy)*0.5f); // normalized to 1
      if (f < ft3)
      { // low frequencies
        sigmacur = sigma4 + (sigma3 - sigma4)*f / ft3;
      }
      else if (f < ft2)
      { // middle frequencies
        sigmacur = sigma3 + (sigma2 - sigma3)*(f - ft3) / (ft2 - ft3);
      }
      else
      {// high frequencies
        sigmacur = sigma + (sigma2 - sigma)*(1 - f) / (1 - ft2);
      }
      pattern2d[w] = sigmacur*sigmacur / norm;
    }
    pattern2d += outpitch;
  }
}

//-------------------------------------------------------------------------------------------
void FindPatternBlock(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample)
{
  // since v1.7 outwidth must be really an outpitch
  int h;
  int w;
  fftwf_complex *outcur;
  float psd;
  float sigmaSquaredcur;
  float sigmaSquared;
  sigmaSquared = 1e15f;

  for (int by = 2; by < noy - 2; by++)
  {
    for (int bx = 2; bx < nox - 2; bx++)
    {
      outcur = outcur0 + nox*by*bh*outpitch + bx*bh*outpitch;
      sigmaSquaredcur = 0;
      float gcur = degrid*outcur[0][0] / gridsample[0][0]; // grid (windowing) correction factor
      for (h = 0; h < bh; h++)
      {
        for (w = 0; w < outwidth; w++)
        {
          //					psd = outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1];
          float grid0 = gcur*gridsample[w][0];
          float grid1 = gcur*gridsample[w][1];
          float corrected0 = outcur[w][0] - grid0;
          float corrected1 = outcur[w][1] - grid1;
          psd = corrected0*corrected0 + corrected1*corrected1;
          sigmaSquaredcur += psd*pwin[w]; // windowing
        }
        outcur += outpitch;
        pwin += outpitch;
        gridsample += outpitch;
      }
      pwin -= outpitch*bh; // restore
      if (sigmaSquaredcur < sigmaSquared)
      {
        px = bx;
        py = by;
        sigmaSquared = sigmaSquaredcur;
      }
    }
  }
}
//-------------------------------------------------------------------------------------------
void SetPattern(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample)
{
  int h;
  int w;
  outcur += nox*py*bh*outpitch + px*bh*outpitch;
  float psd;
  float sigmaSquared = 0;
  float weight = 0;

  for (h = 0; h < bh; h++)
  {
    for (w = 0; w < outwidth; w++)
    {
      weight += pwin[w];
    }
    pwin += outpitch;
  }
  pwin -= outpitch*bh; // restore

  float gcur = degrid*outcur[0][0] / gridsample[0][0]; // grid (windowing) correction factor

  for (h = 0; h < bh; h++)
  {
    for (w = 0; w < outwidth; w++)
    {
      float grid0 = gcur*gridsample[w][0];
      float grid1 = gcur*gridsample[w][1];
      float corrected0 = outcur[w][0] - grid0;
      float corrected1 = outcur[w][1] - grid1;
      psd = corrected0*corrected0 + corrected1*corrected1;
      //			psd = outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1];
      pattern2d[w] = psd*pwin[w]; // windowing
      sigmaSquared += pattern2d[w]; // sum
    }
    outcur += outpitch;
    pattern2d += outpitch;
    pwin += outpitch;
    gridsample += outpitch;
  }
  psigma = sqrt(sigmaSquared / (weight*bh*outwidth)); // mean std deviation (sigma)
}
//-------------------------------------------------------------------------------------------
void PutPatternOnly(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py)
{
  int h, w;
  int block;
  int pblock = py*nox + px;
  int blocks = nox*noy;

  for (block = 0; block < pblock; block++)
  {
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < outwidth; w++)
      {
        outcur[w][0] = 0;
        outcur[w][1] = 0;
      }
      outcur += outpitch;
    }
  }

  outcur += bh*outpitch;

  for (block = pblock + 1; block < blocks; block++)
  {
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < outwidth; w++)
      {
        outcur[w][0] = 0;
        outcur[w][1] = 0;
      }
      outcur += outpitch;
    }
  }

}
//-------------------------------------------------------------------------------------------
void Pattern2Dto3D(float *pattern2d, int bh, int outwidth, int outpitch, float mult, float *pattern3d)
{
  // slow, but executed once only per clip
  int size = bh*outpitch;
  for (int i = 0; i < size; i++)
  { // get 3D pattern
    pattern3d[i] = pattern2d[i] * mult;
  }
}

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
// void CopyFrame(PVideoFrame &src, PVideoFrame &dst, VideoInfo vi, int planeskip, IScriptEnvironment* env)
// {
//   const BYTE * srcp;
//   BYTE * dstp;
//   int src_height, src_width, src_pitch;
//   int dst_height, dst_width, dst_pitch;
//   int planarNum, plane;

//   // greyscale is planar as well
//   for (plane = 0; plane < vi.NumComponents(); plane++)
//   {
//     if (plane != planeskip)
//     {
//       int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
//       int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
//       int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
//       planarNum = planes[plane];

//       srcp = src->GetReadPtr(planarNum);
//       src_height = src->GetHeight(planarNum);
//       src_width = src->GetRowSize(planarNum);
//       src_pitch = src->GetPitch(planarNum);
//       dstp = dst->GetWritePtr(planarNum);
//       dst_height = dst->GetHeight(planarNum);
//       dst_width = dst->GetRowSize(planarNum);
//       dst_pitch = dst->GetPitch(planarNum);
//       env->BitBlt(dstp, dst_pitch, srcp, src_pitch, dst_width, dst_height); // copy one plane
//     }
//   }
// }

