#include "buffer.h"

template<typename pixel_t>
static void FrameToCover_impl(const pixel_t *srcp, int src_width, int src_height, int src_pitch, pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced)
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

template<typename pixel_t>
static void CoverToFrame_impl(const pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, pixel_t *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced)
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

template<typename pixel_t, int _bits_per_pixel, bool chroma>
static void CoverToOverlap_impl(EngineParams * ep, IOParams * iop, float *dst_ptr, const byte *src_ptr, int src_width, int src_pitch)
{
  // pitch is pixel_t granularity, can be used directly as scrp+=pitch
  int w, h;
  int ihx, ihy;
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(ep, iop, src_ptr);// + (hrest/2)*src_pitch + wrest/2; // centered
  float ftmp;
  int xoffset = ep->bh*ep->bw - (ep->bw - ep->ow); // skip frames
  int yoffset = ep->bw*iop->nox*ep->bh - ep->bw*(ep->bh - ep->oh); // vertical offset of same block (overlap)

  float *inp = dst_ptr;
  //	char debugbuf[1536];
  //	wsprintf(debugbuf,"FFT3DFilter: InitOverlapPlane");
  //	OutputDebugString(debugbuf);
  typedef typename std::conditional<sizeof(pixel_t) == 4, float, int>::type cast_t;
  // for float: chroma center is also 0.0
  constexpr cast_t planeBase = sizeof(pixel_t) == 4 ? cast_t(chroma ? 0.0f : 0.0f) : cast_t(chroma ? (1 << (_bits_per_pixel - 1)) : 0); // anti warning

  ihy = 0; // first top (big non-overlapped) part
  {
    for (h = 0; h < ep->oh; h++)
    {
      inp = dst_ptr + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(iop->wanxl[w] * iop->wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(iop->wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ep->ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float(iop->wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop->wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *iop->wanxl[w];   // overlapped Copy - next block
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(iop->wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last part (non-overlapped) of line of last block
      {
        inp[w] = float(iop->wanxr[w] * iop->wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->ow;
      srcp += ep->ow;
      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }
    for (h = ep->oh; h < ep->bh - ep->oh; h++)
    {
      inp = dst_ptr + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(iop->wanxl[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ep->ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop->wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *iop->wanxl[w];   // overlapped Copy - next block
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last part (non-overlapped) line of last block
      {
        inp[w] = float(iop->wanxr[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }
  }

  for (ihy = 1; ihy < iop->noy; ihy += 1) // middle vertical
  {
    for (h = 0; h < ep->oh; h++) // top overlapped part
    {
      inp = dst_ptr + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // first half line of first block
      {
        ftmp = float(iop->wanxl[w] * (srcp[w] - planeBase));
        inp[w] = ftmp*iop->wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*iop->wanyl[h];   // y overlapped
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        ftmp = float((srcp[w] - planeBase));
        inp[w] = ftmp*iop->wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*iop->wanyl[h];   // y overlapped
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop->wanxr[w] * iop->wanyr[h];
          inp[w + xoffset] = ftmp *iop->wanxl[w] * iop->wanyr[h];   // x overlapped
          inp[w + yoffset] = ftmp * iop->wanxr[w] * iop->wanyl[h];
          inp[w + xoffset + yoffset] = ftmp *iop->wanxl[w] * iop->wanyl[h];   // x overlapped
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop->wanyr[h];
          inp[w + yoffset] = ftmp * iop->wanyl[h];
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        ftmp = float(iop->wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp*iop->wanyr[h];
        inp[w + yoffset] = ftmp*iop->wanyl[h];
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical nonovelapped part
    for (h = 0; h < ep->bh - ep->oh - ep->oh; h++)
    {
      inp = dst_ptr + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh)*ep->bw + h*ep->bw + yoffset;
      for (w = 0; w < ep->ow; w++)   // first half line of first block
      {
        ftmp = float(iop->wanxl[w] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        ftmp = float((srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop->wanxr[w];
          inp[w + xoffset] = ftmp *iop->wanxl[w];   // x overlapped
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp;
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        ftmp = float(iop->wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp;
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }

  }

  ihy = iop->noy; // last bottom  part
  {
    for (h = 0; h < ep->oh; h++)
    {
      inp = dst_ptr + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // first half line of first block
      {
        ftmp = float(iop->wanxl[w] * iop->wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        ftmp = float(iop->wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half line of block
        {
          float ftmp = float(iop->wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop->wanxr[w];
          inp[w + xoffset] = ftmp *iop->wanxl[w];   // overlapped Copy
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(iop->wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        ftmp = float(iop->wanxr[w] * iop->wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }

  }
}

template<typename pixel_t, int _bits_per_pixel, bool chroma>
static void OverlapToCover_impl(EngineParams * ep, IOParams * iop, float *src_ptr, float norm, byte *dst_ptr, int dst_width, int dst_pitch)
{
  int w, h;
  int ihx, ihy;
  pixel_t *dstp = reinterpret_cast<pixel_t *>(ep, iop, dst_ptr);// + (hrest/2)*dst_pitch + wrest/2; // centered
  float *inp = src_ptr;
  int xoffset = ep->bh*ep->bw - (ep->bw - ep->ow);
  int yoffset = ep->bw*iop->nox*ep->bh - ep->bw*(ep->bh - ep->oh); // vertical offset of same block (overlap)
  typedef typename std::conditional<sizeof(pixel_t) == 4, float, int>::type cast_t;

  constexpr float rounder = sizeof(pixel_t) == 4 ? 0.0f : 0.5f; // v2.6

  // for float: chroma center is also 0.0
  constexpr cast_t planeBase = sizeof(pixel_t) == 4 ? cast_t(chroma ? 0.0f : 0.0f) : cast_t(chroma ? (1 << (_bits_per_pixel-1)) : 0); // anti warning

  constexpr cast_t max_pixel_value = sizeof(pixel_t) == 4 ? (pixel_t)1.0f : (pixel_t)((1 << _bits_per_pixel) - 1);

  ihy = 0; // first top big non-overlapped) part
  {
    for (h = 0; h < ep->bh - ep->oh; h++)
    {
      inp = src_ptr + h*ep->bw;
      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX((cast_t)0, (cast_t)(inp[w] * norm + rounder) + planeBase));   // Copy each byte from float array to dest with windows
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle horizontal half-blocks
      {
        for (w = 0; w < ep->ow; w++)   // half line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * iop->wsynxr[w] + inp[w + xoffset] * iop->wsynxl[w])*norm + rounder) + planeBase));   // overlapped Copy
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // first half line of first block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));   // Copy each byte from float array to dest with windows
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the dest image.
    }
  }

  for (ihy = 1; ihy < iop->noy; ihy += 1) // middle vertical
  {
    for (h = 0; h < ep->oh; h++) // top overlapped part
    {
      inp = src_ptr + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;

      float wsynyrh = iop->wsynyr[h] * norm; // remove from cycle for speed
      float wsynylh = iop->wsynyl[h] * norm;

      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder)+ planeBase));   // y overlapped
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(((inp[w] * iop->wsynxr[w] + inp[w + xoffset] * iop->wsynxl[w])*wsynyrh
            + (inp[w + yoffset] * iop->wsynxr[w] + inp[w + xoffset + yoffset] * iop->wsynxl[w])*wsynylh) + rounder) + planeBase));   // x overlapped
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // double minus - half non-overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder) + planeBase));
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical non-ovelapped part
    for (h = 0; h < (ep->bh - ep->oh - ep->oh); h++)
    {
      inp = src_ptr + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh)*ep->bw + h*ep->bw + yoffset;
      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * iop->wsynxr[w] + inp[w + xoffset] * iop->wsynxl[w])*norm + rounder) + planeBase));   // x overlapped
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half non-overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the source image.
    }

  }

  ihy = iop->noy; // last bottom part
  {
    for (h = 0; h < ep->oh; h++)
    {
      inp = src_ptr + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;
      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < iop->nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * iop->wsynxr[w] + inp[w + xoffset] * iop->wsynxl[w])*norm + rounder) + planeBase));   // overlapped Copy
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the source image.
    }
  }
}

void FrameToCover(EngineParams * ep, int plane, const byte *src_ptr, byte *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh)
{
  auto l = ep->IsChroma ? (ep->l >> ep->ssw) : ep->l;
  auto r = ep->IsChroma ? (ep->r >> ep->ssw) : ep->r;
  auto t = ep->IsChroma ? (ep->t >> ep->ssh) : ep->t;
  auto b = ep->IsChroma ? (ep->b >> ep->ssh) : ep->b;
  auto width = ep->framewidth - l - r;
  auto height = ep->frameheight - t - b;
  auto new_src_ptr = src_ptr + (t * ep->framepitch + l) * ep->byte_per_channel;
  switch (ep->bit_per_channel)
  {
  case 8: FrameToCover_impl<uint8_t>(new_src_ptr, width, height, ep->framepitch, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced); break;
  case 10:
  case 12:
  case 14:
  case 16: FrameToCover_impl<uint16_t>((uint16_t *)new_src_ptr, width, height, ep->framepitch, (uint16_t *)coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced); break;
  case 32: FrameToCover_impl<float>((float *)new_src_ptr, width, height, ep->framepitch, (float *)coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced); break;
  }
}

void CoverToFrame(EngineParams * ep, int plane, const byte *coverbuf, int coverwidth, int coverheight, int coverpitch, byte *dst_ptr, int mirw, int mirh)
{
  auto l = ep->IsChroma ? (ep->l >> ep->ssw) : ep->l;
  auto r = ep->IsChroma ? (ep->r >> ep->ssw) : ep->r;
  auto t = ep->IsChroma ? (ep->t >> ep->ssh) : ep->t;
  auto b = ep->IsChroma ? (ep->b >> ep->ssh) : ep->b;
  auto width = ep->framewidth - l - r;
  auto height = ep->frameheight - t - b;
  auto new_dst_ptr = dst_ptr + (t * ep->framepitch + l) * ep->byte_per_channel;
  switch (ep->bit_per_channel)
  {
  case 8: CoverToFrame_impl<uint8_t>(coverbuf, coverwidth, coverheight, coverpitch, new_dst_ptr, width, height, ep->framepitch, mirw, mirh, ep->interlaced); break;
  case 10:
  case 12:
  case 14:
  case 16: CoverToFrame_impl<uint16_t>((uint16_t *)coverbuf, coverwidth, coverheight, coverpitch, (uint16_t *)new_dst_ptr, width, height, ep->framepitch, mirw, mirh, ep->interlaced); break;
  case 32: CoverToFrame_impl<float>((float *)coverbuf, coverwidth, coverheight, coverpitch, (float *)new_dst_ptr, width, height, ep->framepitch, mirw, mirh, ep->interlaced); break;
  }
}

void CoverToOverlap(EngineParams * ep, IOParams * iop, float *dst_ptr, const byte *src_ptr, int src_width, int src_pitch, bool chroma)
{
  // for float: chroma center is also 0.0
  if (chroma) {
    switch (ep->bit_per_channel) {
    case 8: CoverToOverlap_impl<uint8_t, 8, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 10: CoverToOverlap_impl<uint16_t, 10, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 12: CoverToOverlap_impl<uint16_t, 12, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 14: CoverToOverlap_impl<uint16_t, 14, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 16: CoverToOverlap_impl<uint16_t, 16, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 32: CoverToOverlap_impl<float, 8 /*n/a*/, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    }
  }
  else {
    switch (ep->bit_per_channel) {
    case 8: CoverToOverlap_impl<uint8_t, 8, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 10: CoverToOverlap_impl<uint16_t, 10, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 12: CoverToOverlap_impl<uint16_t, 12, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 14: CoverToOverlap_impl<uint16_t, 14, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 16: CoverToOverlap_impl<uint16_t, 16, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 32: CoverToOverlap_impl<float, 8 /*n/a*/, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    }
  }
}

void OverlapToCover(EngineParams * ep, IOParams * iop, float *src_ptr, float norm, byte *dst_ptr, int dst_width, int dst_pitch, bool chroma)
{
  if (chroma) {
    switch (ep->bit_per_channel) {
    case 8: OverlapToCover_impl<uint8_t, 8, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 10: OverlapToCover_impl<uint16_t, 10, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 12: OverlapToCover_impl<uint16_t, 12, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 14: OverlapToCover_impl<uint16_t, 14, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 16: OverlapToCover_impl<uint16_t, 16, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 32: OverlapToCover_impl<float, 8 /*n/a*/, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    }
  }
  else {
    switch (ep->bit_per_channel) {
    case 8: OverlapToCover_impl<uint8_t, 8, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 10: OverlapToCover_impl<uint16_t, 10, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 12: OverlapToCover_impl<uint16_t, 12, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 14: OverlapToCover_impl<uint16_t, 14, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 16: OverlapToCover_impl<uint16_t, 16, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 32: OverlapToCover_impl<float, 8 /*n/a*/, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    }
  }
}
