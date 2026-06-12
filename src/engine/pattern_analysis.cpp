/*
 * Copyright 2004-2007 A.G.Balakhnin aka Fizick
 * Copyright 2015 martin53
 * Copyright 2017-2019 Ferenc Pinter aka printerf
 * Copyright 2020 Xinyue Lu
 *
 * Some helper functions.
 *
 */

#include "engine/pattern_analysis.hpp"

#include "fft3d_common.h"

#include <cmath>
#include <cstddef>

namespace neo_fft3d {

//-------------------------------------------------------------------
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int outwidth, float norm, FloatPlaneView pattern2d_view)
{
  // it is not fast, but called only in constructor
  float sigmacur;
  float ft2 = std::sqrt(0.5F) / 2.0F; // frequency for sigma2
  float ft3 = std::sqrt(0.5F) / 4.0F; // frequency for sigma3
  const int bh = static_cast<int>(pattern2d_view.extent(0));
  const int outpitch = static_cast<int>(pattern2d_view.mapping().stride(0));
  float* pattern2d = pattern2d_view.data_handle();
  for (int h = 0; h < bh; h++)
  {
    for (int w = 0; w < outwidth; w++)
    {
      float fy = (static_cast<float>(bh) - 2.0F * static_cast<float>(abs(h - (bh / 2)))) / static_cast<float>(bh); // normalized to 1
      float fx = static_cast<float>(w) / static_cast<float>(outwidth);  // normalized to 1
      float f = std::sqrt((fx*fx + fy*fy)*0.5F); // normalized to 1
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
void FindPatternBlock(ComplexBlockView spectrum, int outwidth, int nox, int noy, int &px, int &py, ConstFloatPlaneView pwin_view, float degrid, ComplexBlockView gridsample_view)
{
  // since v1.7 outwidth must be really an outpitch
  int h;
  int w;
  fftwf_complex *outcur;
  const int outpitch = spectrum.outpitch;
  const int bh = spectrum.block_height;
  const float* pwin = pwin_view.data_handle();
  fftwf_complex* gridsample = gridsample_view.fftw_data();
  float psd;
  float sigmaSquaredcur;
  float sigmaSquared;
  sigmaSquared = 1e15F;

  for (int by = 2; by < noy - 2; by++)
  {
    for (int bx = 2; bx < nox - 2; bx++)
    {
      outcur = spectrum.fftw_block_data((nox*by) + bx);
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
      pwin -= static_cast<std::ptrdiff_t>(outpitch) * static_cast<std::ptrdiff_t>(bh); // restore
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
void SetPattern(ComplexBlockView spectrum, int outwidth, int nox, int noy, int px, int py, ConstFloatPlaneView pwin_view, FloatPlaneView pattern2d_view, float &psigma, float degrid, ComplexBlockView gridsample_view)
{
  (void)noy;
  int h;
  int w;
  const int outpitch = spectrum.outpitch;
  const int bh = spectrum.block_height;
  fftwf_complex* outcur = spectrum.fftw_block_data((nox*py) + px);
  const float* pwin = pwin_view.data_handle();
  float* pattern2d = pattern2d_view.data_handle();
  fftwf_complex* gridsample = gridsample_view.fftw_data();
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
  pwin -= static_cast<std::ptrdiff_t>(outpitch) * static_cast<std::ptrdiff_t>(bh); // restore

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
  psigma = std::sqrt(
    sigmaSquared / (weight * static_cast<float>(bh) * static_cast<float>(outwidth))
  ); // mean std deviation (sigma)
}
//-------------------------------------------------------------------------------------------
void PutPatternOnly(ComplexBlockView spectrum, int outwidth, int nox, int noy, int px, int py)
{
  int h;
  int w;
  int block;
  fftwf_complex* outcur = spectrum.fftw_data();
  const int outpitch = spectrum.outpitch;
  const int bh = spectrum.block_height;
  int pblock = (py*nox) + px;
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

  outcur += static_cast<std::ptrdiff_t>(bh) * static_cast<std::ptrdiff_t>(outpitch);

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
void Pattern2Dto3D(ConstFloatPlaneView pattern2d_view, float mult, FloatPlaneView pattern3d_view)
{
  // slow, but executed once only per clip
  const int bh = static_cast<int>(pattern2d_view.extent(0));
  const int outpitch = static_cast<int>(pattern2d_view.mapping().stride(0));
  int size = bh*outpitch;
  const float* pattern2d = pattern2d_view.data_handle();
  float* pattern3d = pattern3d_view.data_handle();
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

} // namespace neo_fft3d
