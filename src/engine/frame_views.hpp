#pragma once

#include "common/fft3d_views.hpp"

#include <dualsynth/video_filter.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace neo_fft3d::engine {

struct CoverPlaneGeometry {
  int width {0};
  int height {0};
  int pitch {0};
  int bytes_per_sample {0};
};

inline ds::VideoFrameView FetchFrame(ds::VideoFrameProvider& provider, int frame_num) {
  auto frame = provider.get(0, frame_num);
  if (!frame.has_value()) {
    throw std::runtime_error("neo_fft3d: failed to fetch frame " + std::to_string(frame_num) + ": " + frame.error().message);
  }
  return frame.value().frame;
}

inline void CopyPlanePixels(const ds::PlaneView& src, const ds::MutablePlaneView& dst, int bytes_per_sample) {
  const auto row_bytes = static_cast<std::size_t>(src.width) * static_cast<std::size_t>(bytes_per_sample);
  auto src_ptr = static_cast<const byte*>(src.data);
  auto dst_ptr = static_cast<byte*>(dst.data);

  for (int y = 0; y < src.height; ++y) {
    std::memcpy(dst_ptr, src_ptr, row_bytes);
    src_ptr += src.stride_bytes;
    dst_ptr += dst.stride_bytes;
  }
}

inline void CopyFramePlanePixels(
  const ds::VideoFrameView& src,
  ds::MutableVideoFrameView& dst,
  int plane,
  int bytes_per_sample
) {
  CopyPlanePixels(src.plane(plane), dst.plane(plane), bytes_per_sample);
}

inline BytePlaneView SourcePlaneView(const ds::PlaneView& src, int bytes_per_sample) {
  return make_byte_plane_view(
    static_cast<const byte*>(src.data),
    src.width * bytes_per_sample,
    src.height,
    src.stride_bytes
  );
}

inline MutableBytePlaneView DestinationPlaneView(const ds::MutablePlaneView& dst, int bytes_per_sample) {
  return make_mutable_byte_plane_view(
    static_cast<byte*>(dst.data),
    dst.width * bytes_per_sample,
    dst.height,
    dst.stride_bytes
  );
}

inline BytePlaneView CoverPlaneView(const byte* data, CoverPlaneGeometry geometry) {
  return make_byte_plane_view(
    data,
    geometry.width * geometry.bytes_per_sample,
    geometry.height,
    static_cast<std::ptrdiff_t>(geometry.pitch) * static_cast<std::ptrdiff_t>(geometry.bytes_per_sample)
  );
}

inline MutableBytePlaneView MutableCoverPlaneView(byte* data, CoverPlaneGeometry geometry) {
  return make_mutable_byte_plane_view(
    data,
    geometry.width * geometry.bytes_per_sample,
    geometry.height,
    static_cast<std::ptrdiff_t>(geometry.pitch) * static_cast<std::ptrdiff_t>(geometry.bytes_per_sample)
  );
}

inline FloatSpan OverlapSpan(float* data, std::size_t size) {
  return FloatSpan{data, size};
}

class FrameViewAdapter {
public:
  FrameViewAdapter(CoverPlaneGeometry cover_geometry, std::size_t overlap_size) :
    cover_geometry_(cover_geometry),
    overlap_size_(overlap_size) {
  }

  BytePlaneView Source(const ds::PlaneView& src) const {
    return SourcePlaneView(src, cover_geometry_.bytes_per_sample);
  }

  MutableBytePlaneView Destination(const ds::MutablePlaneView& dst) const {
    return DestinationPlaneView(dst, cover_geometry_.bytes_per_sample);
  }

  BytePlaneView Cover(const byte* data) const {
    return CoverPlaneView(data, cover_geometry_);
  }

  MutableBytePlaneView MutableCover(byte* data) const {
    return MutableCoverPlaneView(data, cover_geometry_);
  }

  FloatSpan Overlap(float* data) const {
    return OverlapSpan(data, overlap_size_);
  }

  void CopyPlane(const ds::PlaneView& src, const ds::MutablePlaneView& dst) const {
    CopyPlanePixels(src, dst, cover_geometry_.bytes_per_sample);
  }

private:
  CoverPlaneGeometry cover_geometry_;
  std::size_t overlap_size_ {0};
};

} // namespace neo_fft3d::engine
