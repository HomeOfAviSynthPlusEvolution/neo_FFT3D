/*
 * Copyright 2020 Xinyue Lu
 *
 * DualSynth wrapper - DSFormat.
 *
 */

#pragma once

struct DSFormat
{
  bool IsFamilyYUV {true}, IsFamilyRGB {false}, IsFamilyYCC {false};
  bool IsInteger {true}, IsFloat {false};
  int SSW {0}, SSH {0};
  int BitsPerSample {8}, BytesPerSample {1};
  int Planes {3};
  DSFormat() {}
  DSFormat(const VSFormat* format)
  {
    Planes = format->numPlanes;
    IsFamilyYUV = format->colorFamily == cmYUV || format->colorFamily == cmGray;
    IsFamilyRGB = format->colorFamily == cmRGB;
    IsFamilyYCC = format->colorFamily == cmYCoCg;
    SSW = format->subSamplingW;
    SSH = format->subSamplingH;
    BitsPerSample = format->bitsPerSample;
    BytesPerSample = format->bytesPerSample;
    IsInteger = format->sampleType == stInteger;
    IsFloat = format->sampleType == stFloat;
  }

  const VSFormat* ToVSFormat(const VSCore* vscore, const VSAPI* vsapi) const
  {
    VSColorFamily family = cmYUV;
    if (IsFamilyYUV)
      family = Planes == 1 ? cmGray : cmYUV;
    else if (IsFamilyRGB)
      family = cmRGB;
    else if (IsFamilyYCC)
      family = cmYCoCg;
    return vsapi->registerFormat(family, IsInteger ? stInteger : stFloat, BitsPerSample, SSW, SSH, const_cast<VSCore*>(vscore));
  }

  DSFormat(const VideoInfo& vi)
  {
    if (!vi.IsPlanar())
      throw "neo-dfttest only supports planar formats.";
    IsFamilyYUV = vi.IsYUV() || vi.IsYUVA();
    IsFamilyRGB = vi.IsRGB() || vi.IsPlanarRGBA();
    IsFamilyYCC = false;
    IsFloat = (vi.ComponentSize() == 4);
    IsInteger = !IsFloat;
    Planes = vi.NumComponents();
    BitsPerSample = vi.BitsPerComponent();
    BytesPerSample = vi.ComponentSize();

    if (IsFamilyYUV && Planes > 1) {
      SSW = vi.GetPlaneWidthSubsampling(PLANAR_U);
      SSH = vi.GetPlaneHeightSubsampling(PLANAR_U);
    }
  }

  int ToAVSFormat() const
  {
    int pixel_format = VideoInfo::CS_PLANAR | (Planes == 3 ? VideoInfo::CS_YUV : VideoInfo::CS_YUVA) | VideoInfo::CS_VPlaneFirst;
    if (IsFamilyYUV) {
      if (Planes == 1)
        pixel_format = VideoInfo::CS_GENERIC_Y;
      else {
        pixel_format = VideoInfo::CS_PLANAR | (Planes == 3 ? VideoInfo::CS_YUV : VideoInfo::CS_YUVA) | VideoInfo::CS_VPlaneFirst;

        switch (SSW) {
          case 0: pixel_format |= VideoInfo::CS_Sub_Width_1; break;
          case 1: pixel_format |= VideoInfo::CS_Sub_Width_2; break;
          case 2: pixel_format |= VideoInfo::CS_Sub_Width_4; break;
        }

        switch (SSH) {
          case 0: pixel_format |= VideoInfo::CS_Sub_Height_1; break;
          case 1: pixel_format |= VideoInfo::CS_Sub_Height_2; break;
          case 2: pixel_format |= VideoInfo::CS_Sub_Height_4; break;
        }
      }
    }
    else if (IsFamilyRGB || IsFamilyYCC)
      pixel_format = VideoInfo::CS_PLANAR | VideoInfo::CS_BGR | (Planes == 3 ? VideoInfo::CS_RGB_TYPE : VideoInfo::CS_RGBA_TYPE);

    switch(BitsPerSample) {
      case 8: pixel_format |= VideoInfo::CS_Sample_Bits_8; break;
      case 10: pixel_format |= VideoInfo::CS_Sample_Bits_10; break;
      case 12: pixel_format |= VideoInfo::CS_Sample_Bits_12; break;
      case 14: pixel_format |= VideoInfo::CS_Sample_Bits_14; break;
      case 16: pixel_format |= VideoInfo::CS_Sample_Bits_16; break;
      case 32: pixel_format |= VideoInfo::CS_Sample_Bits_32; break;
    }

    return pixel_format;
  }
};
