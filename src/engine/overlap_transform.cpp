#include "engine/frame_buffer.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#pragma warning (disable: 26451)

namespace {

template <class View>
int view_width_bytes(View view) {
  return static_cast<int>(view.extent(1));
}

template <class View>
int view_height(View view) {
  return static_cast<int>(view.extent(0));
}

template <class View>
int view_stride_bytes(View view) {
  return static_cast<int>(view.mapping().stride(0));
}

int view_width_samples(neo_fft3d::BytePlaneView view, int bytes_per_sample) {
  return view_width_bytes(view) / bytes_per_sample;
}

int view_stride_samples(neo_fft3d::BytePlaneView view, int bytes_per_sample) {
  return view_stride_bytes(view) / bytes_per_sample;
}

int view_width_samples(neo_fft3d::MutableBytePlaneView view, int bytes_per_sample) {
  return view_width_bytes(view) / bytes_per_sample;
}

int view_stride_samples(neo_fft3d::MutableBytePlaneView view, int bytes_per_sample) {
  return view_stride_bytes(view) / bytes_per_sample;
}

std::ptrdiff_t sample_offset(int lhs, int rhs) {
  return static_cast<std::ptrdiff_t>(lhs) * static_cast<std::ptrdiff_t>(rhs);
}

std::ptrdiff_t sample_offset(std::ptrdiff_t lhs, int rhs) {
  return lhs * static_cast<std::ptrdiff_t>(rhs);
}

std::ptrdiff_t sample_offset(int lhs, std::ptrdiff_t rhs) {
  return static_cast<std::ptrdiff_t>(lhs) * rhs;
}

template <typename Pixel, typename Value>
Pixel clamp_pixel(Value value, Value max_value) {
  return static_cast<Pixel>(std::clamp(value, Value{0}, max_value));
}

} // namespace

template<typename pixel_t, int _bits_per_pixel, bool chroma>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void CoverToOverlap_impl(const EngineParams& ep, const IOParams& iop, float *dst_ptr, const byte *src_ptr, int src_width, int src_pitch)
{
  // pitch is pixel_t granularity, can be used directly as scrp+=pitch
  int w;
  int h;
  int ihx;
  int ihy;
  const auto *srcp = reinterpret_cast<const pixel_t *>(src_ptr);// + (hrest/2)*src_pitch + wrest/2; // centered
  float ftmp;
  int xoffset = (ep.bh*ep.bw) - (ep.bw - ep.ow); // skip frames
  int yoffset = (ep.bw*iop.nox*ep.bh) - (ep.bw*(ep.bh - ep.oh)); // vertical offset of same block (overlap)

  float *inp = dst_ptr;
  //	char debugbuf[1536];
  //	wsprintf(debugbuf,"FFT3DFilter: InitOverlapPlane");
  //	OutputDebugString(debugbuf);
  using cast_t = std::conditional_t<sizeof(pixel_t) == 4, float, int>;
  // for float: chroma center is also 0.0
  constexpr cast_t planeBase = sizeof(pixel_t) == 4 ? 0 : cast_t(chroma ? (1 << (_bits_per_pixel - 1)) : 0); // anti warning

  ihy = 0; // first top (big non-overlapped) part
  {
    for (h = 0; h < ep.oh; h++)
    {
      inp = dst_ptr + sample_offset(h, ep.bw);
      for (w = 0; w < ep.ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(iop.wanxl[w] * iop.wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ep.ow; w < ep.bw - ep.ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(iop.wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep.bw - ep.ow;
      srcp += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ep.ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float(iop.wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop.wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *iop.wanxl[w];   // overlapped Copy - next block
        }
        inp += ep.ow;
        inp += xoffset;
        srcp += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(iop.wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep.bw - ep.ow - ep.ow;
        srcp += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last part (non-overlapped) of line of last block
      {
        inp[w] = float(iop.wanxr[w] * iop.wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep.ow;
      srcp += ep.ow;
      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }
    for (h = ep.oh; h < ep.bh - ep.oh; h++)
    {
      inp = dst_ptr + sample_offset(h, ep.bw);
      for (w = 0; w < ep.ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(iop.wanxl[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ep.ow; w < ep.bw - ep.ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep.bw - ep.ow;
      srcp += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ep.ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop.wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *iop.wanxl[w];   // overlapped Copy - next block
        }
        inp += ep.ow;
        inp += xoffset;
        srcp += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep.bw - ep.ow - ep.ow;
        srcp += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last part (non-overlapped) line of last block
      {
        inp[w] = float(iop.wanxr[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep.ow;
      srcp += ep.ow;

      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }
  }

#ifdef ENABLE_PAR
  std::for_each_n(std::execution::par_unseq, reinterpret_cast<char*>(1), iop.noy - 1, [&](char&idx) {
    int ihy = static_cast<int>(reinterpret_cast<intptr_t>(&idx));
#else
  for (ihy = 1; ihy < iop.noy; ihy += 1) {
#endif
    int w;
    int h;
    int ihx;
    float ftmp;
    auto srcp0 = srcp + sample_offset(sample_offset(src_pitch, ihy - 1), ep.bh - ep.oh);
    for (h = 0; h < ep.oh; h++) // top overlapped part
    {
      auto *inp = dst_ptr +
                 sample_offset(ihy - 1, static_cast<std::ptrdiff_t>(yoffset) + sample_offset(ep.bh - ep.oh, ep.bw)) +
                 sample_offset(ep.bh - ep.oh, ep.bw) +
                 sample_offset(h, ep.bw);
      for (w = 0; w < ep.ow; w++)   // first half line of first block
      {
        ftmp = float(iop.wanxl[w] * (srcp0[w] - planeBase));
        inp[w] = ftmp*iop.wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*iop.wanyl[h];   // y overlapped
      }
      for (w = ep.ow; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        ftmp = float((srcp0[w] - planeBase));
        inp[w] = ftmp*iop.wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*iop.wanyl[h];   // y overlapped
      }
      inp += ep.bw - ep.ow;
      srcp0 += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep.ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp0[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop.wanxr[w] * iop.wanyr[h];
          inp[w + xoffset] = ftmp *iop.wanxl[w] * iop.wanyr[h];   // x overlapped
          inp[w + yoffset] = ftmp * iop.wanxr[w] * iop.wanyl[h];
          inp[w + xoffset + yoffset] = ftmp *iop.wanxl[w] * iop.wanyl[h];   // x overlapped
        }
        inp += ep.ow;
        inp += xoffset;
        srcp0 += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp0[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop.wanyr[h];
          inp[w + yoffset] = ftmp * iop.wanyl[h];
        }
        inp += ep.bw - ep.ow - ep.ow;
        srcp0 += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        ftmp = float(iop.wanxr[w] * (srcp0[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp*iop.wanyr[h];
        inp[w + yoffset] = ftmp*iop.wanyl[h];
      }
      inp += ep.ow;
      srcp0 += ep.ow;

      srcp0 += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical nonovelapped part
    for (h = 0; h < ep.bh - ep.oh - ep.oh; h++)
    {
      auto *inp = dst_ptr +
                 sample_offset(ihy - 1, static_cast<std::ptrdiff_t>(yoffset) + sample_offset(ep.bh - ep.oh, ep.bw)) +
                 sample_offset(ep.bh, ep.bw) +
                 sample_offset(h, ep.bw) +
                 yoffset;
      for (w = 0; w < ep.ow; w++)   // first half line of first block
      {
        ftmp = float(iop.wanxl[w] * (srcp0[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ep.ow; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        ftmp = float((srcp0[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep.bw - ep.ow;
      srcp0 += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep.ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp0[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop.wanxr[w];
          inp[w + xoffset] = ftmp *iop.wanxl[w];   // x overlapped
        }
        inp += ep.ow;
        inp += xoffset;
        srcp0 += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp0[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp;
        }
        inp += ep.bw - ep.ow - ep.ow;
        srcp0 += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        ftmp = float(iop.wanxr[w] * (srcp0[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp;
      }
      inp += ep.ow;
      srcp0 += ep.ow;

      srcp0 += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }

  }
#ifdef ENABLE_PAR
  ); // std::for_each
#endif

  srcp += sample_offset(sample_offset(src_pitch, iop.noy - 1), ep.bh - ep.oh);
  ihy = iop.noy; // last bottom  part
  {
    for (h = 0; h < ep.oh; h++)
    {
      inp = dst_ptr +
            sample_offset(ihy - 1, static_cast<std::ptrdiff_t>(yoffset) + sample_offset(ep.bh - ep.oh, ep.bw)) +
            sample_offset(ep.bh - ep.oh, ep.bw) +
            sample_offset(h, ep.bw);
      for (w = 0; w < ep.ow; w++)   // first half line of first block
      {
        ftmp = float(iop.wanxl[w] * iop.wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ep.ow; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        ftmp = float(iop.wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep.bw - ep.ow;
      srcp += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep.ow; w++)   // half line of block
        {
          auto ftmp = float(iop.wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * iop.wanxr[w];
          inp[w + xoffset] = ftmp *iop.wanxl[w];   // overlapped Copy
        }
        inp += ep.ow;
        inp += xoffset;
        srcp += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(iop.wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep.bw - ep.ow - ep.ow;
        srcp += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        ftmp = float(iop.wanxr[w] * iop.wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep.ow;
      srcp += ep.ow;

      srcp += (src_pitch - src_width);  // Add the pitch of one line (in bytes) to the source image.
    }

  }
}

template<typename pixel_t, int _bits_per_pixel, bool chroma>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void OverlapToCover_impl(const EngineParams& ep, const IOParams& iop, float *src_ptr, float norm, byte *dst_ptr, int dst_width, int dst_pitch)
{
  int w;
  int h;
  int ihx;
  int ihy;
  auto *dstp = reinterpret_cast<pixel_t *>(dst_ptr);// + (hrest/2)*dst_pitch + wrest/2; // centered
  float *inp = src_ptr;
  int xoffset = (ep.bh*ep.bw) - (ep.bw - ep.ow);
  int yoffset = (ep.bw*iop.nox*ep.bh) - (ep.bw*(ep.bh - ep.oh)); // vertical offset of same block (overlap)
  using cast_t = std::conditional_t<sizeof(pixel_t) == 4, float, int>;

  constexpr float rounder = sizeof(pixel_t) == 4 ? 0.0F : 0.5F; // v2.6

  // for float: chroma center is also 0.0
  constexpr cast_t planeBase = sizeof(pixel_t) == 4 ? 0 : cast_t(chroma ? (1 << (_bits_per_pixel-1)) : 0); // anti warning

  constexpr cast_t max_pixel_value = sizeof(pixel_t) == 4 ? (pixel_t)1 : (pixel_t)((1 << _bits_per_pixel) - 1);

  ihy = 0; // first top big non-overlapped) part
  {
    for (h = 0; h < ep.bh - ep.oh; h++)
    {
      inp = src_ptr + sample_offset(h, ep.bw);
      for (w = 0; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * norm) + rounder + planeBase), max_pixel_value);   // Copy each byte from float array to dest with windows
      }
      inp += ep.bw - ep.ow;
      dstp += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle horizontal half-blocks
      {
        for (w = 0; w < ep.ow; w++)   // half line of block
        {
          dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w] * iop.wsynxr[w] + inp[w + xoffset] * iop.wsynxl[w]) * norm) + rounder + planeBase), max_pixel_value);   // overlapped Copy
        }
        inp += xoffset + ep.ow;
        dstp += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // first half line of first block
        {
          dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * norm) + rounder + planeBase), max_pixel_value);   // Copy each byte from float array to dest with windows
        }
        inp += ep.bw - ep.ow - ep.ow;
        dstp += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * norm) + rounder + planeBase), max_pixel_value);
      }
      inp += ep.ow;
      dstp += ep.ow;

      dstp += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the dest image.
    }
  }

#ifdef ENABLE_PAR
  std::for_each_n(std::execution::par_unseq, reinterpret_cast<char*>(1), iop.noy - 1, [&](char&idx) {
    int ihy = static_cast<int>(reinterpret_cast<intptr_t>(&idx));
#else
  for (ihy = 1; ihy < iop.noy; ihy += 1) {
#endif
    int w;
    int h;
    int ihx;
    auto dstp0 = dstp + sample_offset(sample_offset(dst_pitch, ihy - 1), ep.bh - ep.oh);
    for (h = 0; h < ep.oh; h++) // top overlapped part
    {
      auto *inp = src_ptr +
                 sample_offset(ihy - 1, static_cast<std::ptrdiff_t>(yoffset) + sample_offset(ep.bh - ep.oh, ep.bw)) +
                 sample_offset(ep.bh - ep.oh, ep.bw) +
                 sample_offset(h, ep.bw);

      float wsynyrh = iop.wsynyr[h] * norm; // remove from cycle for speed
      float wsynylh = iop.wsynyl[h] * norm;

      for (w = 0; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder + planeBase), max_pixel_value);   // y overlapped
      }
      inp += ep.bw - ep.ow;
      dstp0 += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep.ow; w++)   // half overlapped line of block
        {
          dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w] * iop.wsynxr[w] + inp[w + xoffset] * iop.wsynxl[w])*wsynyrh
            + (inp[w + yoffset] * iop.wsynxr[w] + inp[w + xoffset + yoffset] * iop.wsynxl[w])*wsynylh) + rounder + planeBase), max_pixel_value);   // x overlapped
        }
        inp += xoffset + ep.ow;
        dstp0 += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // double minus - half non-overlapped line of block
        {
          dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder + planeBase), max_pixel_value);
        }
        inp += ep.bw - ep.ow - ep.ow;
        dstp0 += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder + planeBase), max_pixel_value);
      }
      inp += ep.ow;
      dstp0 += ep.ow;

      dstp0 += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical non-ovelapped part
    for (h = 0; h < (ep.bh - ep.oh - ep.oh); h++)
    {
      auto *inp = src_ptr +
                 sample_offset(ihy - 1, static_cast<std::ptrdiff_t>(yoffset) + sample_offset(ep.bh - ep.oh, ep.bw)) +
                 sample_offset(ep.bh, ep.bw) +
                 sample_offset(h, ep.bw) +
                 yoffset;
      for (w = 0; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w]) * norm) + rounder + planeBase), max_pixel_value);
      }
      inp += ep.bw - ep.ow;
      dstp0 += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep.ow; w++)   // half overlapped line of block
        {
          dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w] * iop.wsynxr[w] + inp[w + xoffset] * iop.wsynxl[w]) * norm) + rounder + planeBase), max_pixel_value);   // x overlapped
        }
        inp += xoffset + ep.ow;
        dstp0 += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // half non-overlapped line of block
        {
          dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w]) * norm) + rounder + planeBase), max_pixel_value);
        }
        inp += ep.bw - ep.ow - ep.ow;
        dstp0 += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        dstp0[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w]) * norm) + rounder + planeBase), max_pixel_value);
      }
      inp += ep.ow;
      dstp0 += ep.ow;

      dstp0 += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the source image.
    }

  }
#ifdef ENABLE_PAR
  ); // std::for_each
#endif

  dstp += sample_offset(sample_offset(dst_pitch, iop.noy - 1), ep.bh - ep.oh);
  ihy = iop.noy; // last bottom part
  {
    for (h = 0; h < ep.oh; h++)
    {
      inp = src_ptr +
            sample_offset(ihy - 1, static_cast<std::ptrdiff_t>(yoffset) + sample_offset(ep.bh - ep.oh, ep.bw)) +
            sample_offset(ep.bh - ep.oh, ep.bw) +
            sample_offset(h, ep.bw);
      for (w = 0; w < ep.bw - ep.ow; w++)   // first half line of first block
      {
        dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * norm) + rounder + planeBase), max_pixel_value);
      }
      inp += ep.bw - ep.ow;
      dstp += ep.bw - ep.ow;
      for (ihx = 1; ihx < iop.nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep.ow; w++)   // half line of block
        {
          dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w] * iop.wsynxr[w] + inp[w + xoffset] * iop.wsynxl[w]) * norm) + rounder + planeBase), max_pixel_value);   // overlapped Copy
        }
        inp += xoffset + ep.ow;
        dstp += ep.ow;
        for (w = 0; w < ep.bw - ep.ow - ep.ow; w++)   // half line of block
        {
          dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>(((inp[w]) * norm) + rounder + planeBase), max_pixel_value);
        }
        inp += ep.bw - ep.ow - ep.ow;
        dstp += ep.bw - ep.ow - ep.ow;
      }
      for (w = 0; w < ep.ow; w++)   // last half line of last block
      {
        dstp[w] = clamp_pixel<pixel_t>(static_cast<cast_t>((inp[w] * norm) + rounder + planeBase), max_pixel_value);
      }
      inp += ep.ow;
      dstp += ep.ow;

      dstp += (dst_pitch - dst_width);  // Add the pitch of one line (in bytes) to the source image.
    }
  }
}

void CoverToOverlap(const EngineParams& ep, const IOParams& iop, neo_fft3d::FloatSpan dst, neo_fft3d::BytePlaneView src, bool chroma)
{
  const int bytes_per_sample = ep.vi.Format.BytesPerSample;
  const int src_width = view_width_samples(src, bytes_per_sample);
  const int src_pitch = view_stride_samples(src, bytes_per_sample);
  const byte* src_ptr = src.data_handle();
  float* dst_ptr = dst.data();

  // for float: chroma center is also 0.0
  if (chroma) {
    switch (ep.vi.Format.BitsPerSample) {
    case 8: CoverToOverlap_impl<std::uint8_t, 8, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 10: CoverToOverlap_impl<std::uint16_t, 10, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 12: CoverToOverlap_impl<std::uint16_t, 12, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 14: CoverToOverlap_impl<std::uint16_t, 14, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 16: CoverToOverlap_impl<std::uint16_t, 16, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 32: CoverToOverlap_impl<float, 8 /*n/a*/, true>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    default: break;
    }
  }
  else {
    switch (ep.vi.Format.BitsPerSample) {
    case 8: CoverToOverlap_impl<std::uint8_t, 8, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 10: CoverToOverlap_impl<std::uint16_t, 10, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 12: CoverToOverlap_impl<std::uint16_t, 12, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 14: CoverToOverlap_impl<std::uint16_t, 14, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 16: CoverToOverlap_impl<std::uint16_t, 16, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    case 32: CoverToOverlap_impl<float, 8 /*n/a*/, false>(ep, iop, dst_ptr, src_ptr, src_width, src_pitch); break;
    default: break;
    }
  }
}

void OverlapToCover(const EngineParams& ep, const IOParams& iop, neo_fft3d::FloatSpan src, float norm, neo_fft3d::MutableBytePlaneView dst, bool chroma)
{
  const int bytes_per_sample = ep.vi.Format.BytesPerSample;
  const int dst_width = view_width_samples(dst, bytes_per_sample);
  const int dst_pitch = view_stride_samples(dst, bytes_per_sample);
  float* src_ptr = src.data();
  byte* dst_ptr = dst.data_handle();

  if (chroma) {
    switch (ep.vi.Format.BitsPerSample) {
    case 8: OverlapToCover_impl<std::uint8_t, 8, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 10: OverlapToCover_impl<std::uint16_t, 10, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 12: OverlapToCover_impl<std::uint16_t, 12, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 14: OverlapToCover_impl<std::uint16_t, 14, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 16: OverlapToCover_impl<std::uint16_t, 16, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 32: OverlapToCover_impl<float, 8 /*n/a*/, true>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    default: break;
    }
  }
  else {
    switch (ep.vi.Format.BitsPerSample) {
    case 8: OverlapToCover_impl<std::uint8_t, 8, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 10: OverlapToCover_impl<std::uint16_t, 10, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 12: OverlapToCover_impl<std::uint16_t, 12, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 14: OverlapToCover_impl<std::uint16_t, 14, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 16: OverlapToCover_impl<std::uint16_t, 16, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    case 32: OverlapToCover_impl<float, 8 /*n/a*/, false>(ep, iop, src_ptr, norm, dst_ptr, dst_width, dst_pitch); break;
    default: break;
    }
  }
}
