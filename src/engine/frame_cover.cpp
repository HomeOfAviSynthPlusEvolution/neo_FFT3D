#include "engine/frame_buffer.hpp"
#include <cstddef>
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

std::size_t bytes_for_samples(int samples, int bytes_per_sample) {
  return static_cast<std::size_t>(samples) * static_cast<std::size_t>(bytes_per_sample);
}

std::ptrdiff_t byte_offset_for_sample(int row, int pitch, int column, int bytes_per_sample) {
  return (static_cast<std::ptrdiff_t>(row) * static_cast<std::ptrdiff_t>(pitch) +
          static_cast<std::ptrdiff_t>(column)) *
         static_cast<std::ptrdiff_t>(bytes_per_sample);
}

} // namespace

template<typename pixel_t>
static void FrameToCover_impl(const pixel_t *srcp, int src_width, int src_height, int src_pitch, pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced)
{
  int h, w;
  int width2 = src_width + src_width + mirw + mirw - 2;
  pixel_t * coverbuf1 = coverbuf + coverpitch*mirh;

  constexpr int pixelsize = static_cast<int>(sizeof(pixel_t));
  const auto src_line_bytes = bytes_for_samples(src_width, pixelsize);
  const auto cover_line_bytes = bytes_for_samples(coverwidth, pixelsize);

  if (!interlaced) //progressive
  {
    for (h = mirh; h < src_height + mirh; h++)
    {
      memcpy((byte *)(coverbuf1 + mirw), (const byte *)srcp, src_line_bytes);
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
      memcpy((byte *)(coverbuf1 + mirw), (const byte *)srcp, src_line_bytes); // copy line
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
      memcpy((byte *)(coverbuf1 + mirw), (const byte *)srcp, src_line_bytes); // copy line
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
    memcpy((byte *)coverbuf1, (const byte *)pmirror, cover_line_bytes); // mirror bottom line by line
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
  coverbuf1 = coverbuf;
  pmirror = coverbuf1 + coverpitch*mirh * 2; // pointer to vertical mirror
  for (h = 0; h < mirh; h++)
  {
    memcpy((byte *)coverbuf1, (const byte *)pmirror, cover_line_bytes); // mirror bottom line by line
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
}

template<typename pixel_t>
static void CoverToFrame_impl(const pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, pixel_t *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced)
{
  int h;

  constexpr int pixelsize = static_cast<int>(sizeof(pixel_t));
  const auto dst_line_bytes = bytes_for_samples(dst_width, pixelsize);

  const pixel_t *coverbuf1 = coverbuf + coverpitch*mirh + mirw;
  if (!interlaced) // progressive
  {
    for (h = 0; h < dst_height; h++)
    {
      memcpy((byte *)dstp, (const byte *)coverbuf1, dst_line_bytes); // copy pure frame size only
      dstp += dst_pitch;
      coverbuf1 += coverpitch;
    }
  }
  else // interlaced
  {
    for (h = 0; h < dst_height; h += 2)
    {
      memcpy((byte *)dstp, (const byte *)coverbuf1, dst_line_bytes); // copy pure frame size only
      dstp += dst_pitch * 2;
      coverbuf1 += coverpitch;
    }
    // second field is flipped
    dstp -= dst_pitch;
    for (h = 0; h < dst_height; h += 2)
    {
      memcpy((byte *)dstp, (const byte *)coverbuf1, dst_line_bytes); // copy pure frame size only
      dstp -= dst_pitch * 2;
      coverbuf1 += coverpitch;
    }
  }
}

void FrameToCover(EngineParams * ep, int plane, neo_fft3d::BytePlaneView src, neo_fft3d::MutableBytePlaneView cover, int mirw, int mirh)
{
  auto l = ep->IsChroma ? (ep->l >> ep->vi.Format.SSW) : ep->l;
  auto r = ep->IsChroma ? (ep->r >> ep->vi.Format.SSW) : ep->r;
  auto t = ep->IsChroma ? (ep->t >> ep->vi.Format.SSH) : ep->t;
  auto b = ep->IsChroma ? (ep->b >> ep->vi.Format.SSH) : ep->b;
  auto width = ep->framewidth - l - r;
  auto height = ep->frameheight - t - b;
  const int bytes_per_sample = ep->vi.Format.BytesPerSample;
  const int src_pitch = view_stride_samples(src, bytes_per_sample);
  const int coverwidth = view_width_samples(cover, bytes_per_sample);
  const int coverheight = view_height(cover);
  const int coverpitch = view_stride_samples(cover, bytes_per_sample);
  auto new_src_ptr = src.data_handle() + byte_offset_for_sample(t, src_pitch, l, bytes_per_sample);
  switch (ep->vi.Format.BitsPerSample)
  {
  case 8: FrameToCover_impl<uint8_t>(new_src_ptr, width, height, src_pitch, cover.data_handle(), coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced); break;
  case 10:
  case 12:
  case 14:
  case 16: FrameToCover_impl<uint16_t>((uint16_t *)new_src_ptr, width, height, src_pitch, (uint16_t *)cover.data_handle(), coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced); break;
  case 32: FrameToCover_impl<float>((float *)new_src_ptr, width, height, src_pitch, (float *)cover.data_handle(), coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced); break;
  default: break;
  }
}

void CoverToFrame(EngineParams * ep, int plane, neo_fft3d::BytePlaneView cover, neo_fft3d::MutableBytePlaneView dst, int mirw, int mirh)
{
  auto l = ep->IsChroma ? (ep->l >> ep->vi.Format.SSW) : ep->l;
  auto r = ep->IsChroma ? (ep->r >> ep->vi.Format.SSW) : ep->r;
  auto t = ep->IsChroma ? (ep->t >> ep->vi.Format.SSH) : ep->t;
  auto b = ep->IsChroma ? (ep->b >> ep->vi.Format.SSH) : ep->b;
  auto width = ep->framewidth - l - r;
  auto height = ep->frameheight - t - b;
  const int bytes_per_sample = ep->vi.Format.BytesPerSample;
  const int coverwidth = view_width_samples(cover, bytes_per_sample);
  const int coverheight = view_height(cover);
  const int coverpitch = view_stride_samples(cover, bytes_per_sample);
  const int dst_pitch = view_stride_samples(dst, bytes_per_sample);
  auto new_dst_ptr = dst.data_handle() + byte_offset_for_sample(t, dst_pitch, l, bytes_per_sample);
  switch (ep->vi.Format.BitsPerSample)
  {
  case 8: CoverToFrame_impl<uint8_t>(cover.data_handle(), coverwidth, coverheight, coverpitch, new_dst_ptr, width, height, dst_pitch, mirw, mirh, ep->interlaced); break;
  case 10:
  case 12:
  case 14:
  case 16: CoverToFrame_impl<uint16_t>((uint16_t *)cover.data_handle(), coverwidth, coverheight, coverpitch, (uint16_t *)new_dst_ptr, width, height, dst_pitch, mirw, mirh, ep->interlaced); break;
  case 32: CoverToFrame_impl<float>((float *)cover.data_handle(), coverwidth, coverheight, coverpitch, (float *)new_dst_ptr, width, height, dst_pitch, mirw, mirh, ep->interlaced); break;
  default: break;
  }
}
