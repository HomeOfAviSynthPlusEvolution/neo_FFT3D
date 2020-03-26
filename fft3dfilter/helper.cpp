#include "helper.h"

//-------------------------------------------------------------------
void fill_complex(fftwf_complex *plane, int outsize, float realvalue, float imgvalue)
{
  // it is not fast, but called only in constructor
  int w;
  for (w = 0; w < outsize; w++) {
    plane[w][0] = realvalue;
    plane[w][1] = imgvalue;
  }
}
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
void Copyfft(fftwf_complex *outrez, fftwf_complex *outprev, int outsize)
{
  memcpy((byte*)&outrez[0][0], (byte*)&outprev[0][0], outsize * 8); // more fast
}

//-----------------------------------------------------------------------
//
template<typename pixel_t>
void PlanarPlaneToCoverbuf(const pixel_t *srcp, int src_width, int src_height, int src_pitch, pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced)
{
  int h, w;
  int width2 = src_width + src_width + mirw + mirw - 2;
  pixel_t * coverbuf1 = coverbuf + coverpitch*mirh;

  int pixelsize = sizeof(pixel_t);

  if (!interlaced) //progressive
  {
    for (h = mirh; h < src_height + mirh; h++)
    {
      memcpy((byte *)(coverbuf1 + mirw), (const byte *)srcp, src_width*pixelsize);
      for (w = 0; w < mirw; w++)
      {
        coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
      }
      for (w = src_width + mirw; w < coverwidth; w++)
      {
        coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
      }
      coverbuf1 += coverpitch;
      srcp += src_pitch;
    }
  }
  else // interlaced
  {
    for (h = mirh; h < src_height / 2 + mirh; h++) // first field
    {
      memcpy((byte *)(coverbuf1 + mirw), (const byte *)srcp, src_width*pixelsize); // copy line
      for (w = 0; w < mirw; w++)
      {
        coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
      }
      for (w = src_width + mirw; w < coverwidth; w++)
      {
        coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
      }
      coverbuf1 += coverpitch;
      srcp += src_pitch * 2;
    }

    srcp -= src_pitch;
    for (h = src_height / 2 + mirh; h < src_height + mirh; h++) // flip second field
    {
      memcpy((byte *)(coverbuf1 + mirw), (const byte *)srcp, src_width*pixelsize); // copy line
      for (w = 0; w < mirw; w++)
      {
        coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
      }
      for (w = src_width + mirw; w < coverwidth; w++)
      {
        coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
      }
      coverbuf1 += coverpitch;
      srcp -= src_pitch * 2;
    }
  }

  pixel_t * pmirror = coverbuf1 - coverpitch * 2; // pointer to vertical mirror
  for (h = src_height + mirh; h < coverheight; h++)
  {
    memcpy((byte *)coverbuf1, (const byte *)pmirror, coverwidth*pixelsize); // mirror bottom line by line
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
  coverbuf1 = coverbuf;
  pmirror = coverbuf1 + coverpitch*mirh * 2; // pointer to vertical mirror
  for (h = 0; h < mirh; h++)
  {
    memcpy((byte *)coverbuf1, (const byte *)pmirror, coverwidth*pixelsize); // mirror bottom line by line
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
}
//-----------------------------------------------------------------------
//
template<typename pixel_t>
void CoverbufToPlanarPlane(const pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, pixel_t *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced)
{
  int h;

  int pixelsize = sizeof(pixel_t);

  const pixel_t *coverbuf1 = coverbuf + coverpitch*mirh + mirw;
  if (!interlaced) // progressive
  {
    for (h = 0; h < dst_height; h++)
    {
      memcpy((byte *)dstp, (const byte *)coverbuf1, dst_width*pixelsize); // copy pure frame size only
      dstp += dst_pitch;
      coverbuf1 += coverpitch;
    }
  }
  else // interlaced
  {
    for (h = 0; h < dst_height; h += 2)
    {
      memcpy((byte *)dstp, (const byte *)coverbuf1, dst_width*pixelsize); // copy pure frame size only
      dstp += dst_pitch * 2;
      coverbuf1 += coverpitch;
    }
    // second field is flipped
    dstp -= dst_pitch;
    for (h = 0; h < dst_height; h += 2)
    {
      memcpy((byte *)dstp, (const byte *)coverbuf1, dst_width*pixelsize); // copy pure frame size only
      dstp -= dst_pitch * 2;
      coverbuf1 += coverpitch;
    }
  }
}

//-------------------------------------------------------------------------------------------
void GetAndSubtactMean(float *in, int howmanyblocks, int bw, int bh, int ow, int oh, float *wxl, float *wxr, float *wyl, float *wyr, float *mean)
{
  int h, w, block;
  float meanblock;
  float norma;

  // calculate norma
  norma = 0;
  for (h = 0; h < oh; h++)
  {
    for (w = 0; w < ow; w++)
    {
      norma += wxl[w] * wyl[h];
    }
    for (w = ow; w < bw - ow; w++)
    {
      norma += wyl[h];
    }
    for (w = bw - ow; w < bw; w++)
    {
      norma += wxr[w - bw + ow] * wyl[h];
    }
  }
  for (h = oh; h < bh - oh; h++)
  {
    for (w = 0; w < ow; w++)
    {
      norma += wxl[w];
    }
    for (w = ow; w < bw - ow; w++)
    {
      norma += 1;
    }
    for (w = bw - ow; w < bw; w++)
    {
      norma += wxr[w - bw + ow];
    }
  }
  for (h = bh - oh; h < bh; h++)
  {
    for (w = 0; w < ow; w++)
    {
      norma += wxl[w] * wyr[h - bh + oh];
    }
    for (w = ow; w < bw - ow; w++)
    {
      norma += wyr[h - bh + oh];
    }
    for (w = bw - ow; w < bw; w++)
    {
      norma += wxr[w - bw + ow] * wyr[h - bh + oh];
    }
  }


  for (block = 0; block < howmanyblocks; block++)
  {
    meanblock = 0;
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < bw; w++)
      {
        meanblock += in[w];
      }
      in += bw;
    }
    meanblock /= (bw*bh);
    mean[block] = meanblock;

    in -= bw*bh; // restore pointer
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < bw; w++)
      {
        in[w] -= meanblock;
      }
      in += bw;
    }

  }
}
//-------------------------------------------------------------------------------------------
void RestoreMean(float *in, int howmanyblocks, int bw, int bh, float *mean)
{
  int h, w, block;
  float meanblock;

  for (block = 0; block < howmanyblocks; block++)
  {
    meanblock = mean[block] * (bw*bh);

    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < bw; w++)
      {
        in[w] += meanblock;
      }
      in += bw;
    }
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
void SortCache(int *cachewhat, fftwf_complex **cachefft, int cachesize, int cachestart, int cachestartold)
{
  // sort ordered series, put existant ffts to proper places
  int i;
  int ctemp;
  fftwf_complex *ffttemp;

  int offset = cachestart - cachestartold;
  if (offset > 0) // right
  {
    for (i = 0; i < cachesize; i++)
    {
      if ((i + offset) < cachesize)
      {
        //swap
        ctemp = cachewhat[i + offset];
        cachewhat[i + offset] = cachewhat[i];
        cachewhat[i] = ctemp;
        ffttemp = cachefft[i + offset];
        cachefft[i + offset] = cachefft[i];
        cachefft[i] = ffttemp;
      }
    }
  }
  else if (offset < 0)
  {
    for (i = cachesize - 1; i >= 0; i--)
    {
      if ((i + offset) >= 0)
      {
        ctemp = cachewhat[i + offset];
        cachewhat[i + offset] = cachewhat[i];
        cachewhat[i] = ctemp;
        ffttemp = cachefft[i + offset];
        cachefft[i + offset] = cachefft[i];
        cachefft[i] = ffttemp;
      }
    }
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

