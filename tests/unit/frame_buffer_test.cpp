#include "engine/frame_buffer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace {

EngineParams make_params(int width, int height, int bits_per_sample, int bytes_per_sample) {
  EngineParams ep {};
  ep.vi.Format.BitsPerSample = bits_per_sample;
  ep.vi.Format.BytesPerSample = bytes_per_sample;
  ep.framewidth = width;
  ep.frameheight = height;
  ep.bw = width;
  ep.bh = height;
  ep.ow = 0;
  ep.oh = 0;
  return ep;
}

neo_fft3d::BytePlaneView byte_view(const std::uint8_t* data, int width_bytes, int height, int stride_bytes) {
  return neo_fft3d::make_byte_plane_view(data, width_bytes, height, stride_bytes);
}

neo_fft3d::MutableBytePlaneView mutable_byte_view(std::uint8_t* data, int width_bytes, int height, int stride_bytes) {
  return neo_fft3d::make_mutable_byte_plane_view(data, width_bytes, height, stride_bytes);
}

} // namespace

TEST_CASE("FrameToCover mirrors a cropped progressive 8-bit plane", "[frame_buffer]") {
  auto ep = make_params(4, 3, 8, 1);
  ep.l = 1;
  ep.r = 1;
  ep.t = 1;

  const std::array<std::uint8_t, 12> source {
    10, 11, 12, 13,
    20, 21, 22, 23,
    30, 31, 32, 33
  };
  std::array<std::uint8_t, 16> cover {};

  FrameToCover(
    ep,
    0,
    byte_view(source.data(), 4, 3, 4),
    mutable_byte_view(cover.data(), 4, 4, 4),
    1,
    1
  );

  const std::array<std::uint8_t, 16> expected {
    32, 31, 32, 31,
    22, 21, 22, 21,
    32, 31, 32, 31,
    22, 21, 22, 21
  };
  REQUIRE(cover == expected);
}

TEST_CASE("CoverToFrame writes only the cropped center from a progressive 8-bit cover plane", "[frame_buffer]") {
  auto ep = make_params(4, 3, 8, 1);
  ep.l = 1;
  ep.r = 1;
  ep.t = 1;

  const std::array<std::uint8_t, 16> cover {
    32, 31, 32, 31,
    22, 21, 22, 21,
    32, 31, 32, 31,
    22, 21, 22, 21
  };
  std::array<std::uint8_t, 12> destination {};
  destination.fill(0xee);

  CoverToFrame(
    ep,
    0,
    byte_view(cover.data(), 4, 4, 4),
    mutable_byte_view(destination.data(), 4, 3, 4),
    1,
    1
  );

  const std::array<std::uint8_t, 12> expected {
    0xee, 0xee, 0xee, 0xee,
    0xee, 21,   22,   0xee,
    0xee, 31,   32,   0xee
  };
  REQUIRE(destination == expected);
}

TEST_CASE("CoverToOverlap and OverlapToCover round trip an 8-bit chroma plane without overlap", "[frame_buffer]") {
  auto ep = make_params(2, 2, 8, 1);
  IOParams iop {};
  iop.nox = 1;
  iop.noy = 1;

  const std::array<std::uint8_t, 4> cover {
    128, 129,
    130, 131
  };
  std::array<float, 4> overlap {};

  CoverToOverlap(
    ep,
    iop,
    neo_fft3d::FloatSpan{overlap.data(), overlap.size()},
    byte_view(cover.data(), 2, 2, 2),
    true
  );

  REQUIRE(overlap[0] == 0.0f);
  REQUIRE(overlap[1] == 1.0f);
  REQUIRE(overlap[2] == 2.0f);
  REQUIRE(overlap[3] == 3.0f);

  std::array<std::uint8_t, 4> destination {};
  OverlapToCover(
    ep,
    iop,
    neo_fft3d::FloatSpan{overlap.data(), overlap.size()},
    1.0f,
    mutable_byte_view(destination.data(), 2, 2, 2),
    true
  );

  REQUIRE(destination == cover);
}
