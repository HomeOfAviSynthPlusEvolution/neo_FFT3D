/*
 * Copyright 2020 Xinyue Lu
 *
 * DualSynth wrapper - DSVideoInfo.
 *
 */

#pragma once

struct DSVideoInfo
{
  DSFormat Format;
  int64_t FPSNum {1}, FPSDenom {1};
  int Width {0}, Height {0};
  int Frames {0};

  DSVideoInfo() {}
  DSVideoInfo(DSFormat format, int64_t fpsnum, int64_t fpsdenom, int width, int height, int frames)
    : Format(format)
    , FPSNum(fpsnum), FPSDenom(fpsdenom)
    , Width(width), Height(height)
    , Frames(frames)
  { }
  DSVideoInfo(const VSVideoInfo* vsvi)
    : Format(vsvi->format)
    , FPSNum(vsvi->fpsNum), FPSDenom(vsvi->fpsDen)
    , Width(vsvi->width), Height(vsvi->height)
    , Frames(vsvi->numFrames)
  { }
  DSVideoInfo(const VideoInfo avsvi)
    : Format(avsvi.pixel_type)
    , FPSNum(avsvi.fps_numerator), FPSDenom(avsvi.fps_denominator)
    , Width(avsvi.width), Height(avsvi.height)
    , Frames(avsvi.num_frames)
  { }
  const VSVideoInfo* ToVSVI(const VSCore* vscore, const VSAPI* vsapi) {
    return new VSVideoInfo {Format.ToVSFormat(vscore, vsapi), FPSNum, FPSDenom, Width, Height, Frames, 0};
  }
  const VideoInfo ToAVSVI() {
    return VideoInfo{Width, Height, static_cast<unsigned>(FPSNum), static_cast<unsigned>(FPSDenom), Frames, Format.ToAVSFormat()};
  }
};
