#ifndef __AVS_FILTER_HPP__
#define __AVS_FILTER_HPP__

#include <avisynth.h>

class AVSFilter : public IClip {
public:
  typedef class ::PClip AClip;
  typedef class ::PVideoFrame AFrame;
  using Frametype = AFrame;
  PClip child;
  VideoInfo vi;
protected:
  AVSValue _args;
  IScriptEnvironment* _env;
  int byte_per_channel;
  int bit_per_channel;
public:
  virtual void initialize() {}
  virtual const char* name() const { return "AVSFilter"; };
  virtual AVSFilter::AFrame get(int n, void *ctx) {
    return child->GetFrame(n, _env);
  }

  AVSFilter(AVSValue args, IScriptEnvironment* env)
    : _env(env) {
    _args = args;
    child = _args[0].AsClip();
    vi = child->GetVideoInfo();
  }

  // Filter
  PClip ArgAsClip  (int index, const char*) const { return _args[index].AsClip();    }

  bool        ArgAsBool  (int index, const char*) const { return _args[index].AsBool();   }
  int         ArgAsInt   (int index, const char*) const { return _args[index].AsInt();    }
  float       ArgAsFloat (int index, const char*) const { return (float)_args[index].AsFloat();  }
  const char* ArgAsString(int index, const char*) const { return _args[index].AsString(); }

  bool        ArgAsBool  (int index, const char*, bool def)        const { return _args[index].AsBool  (def); }
  int         ArgAsInt   (int index, const char*, int def)         const { return _args[index].AsInt   (def); }
  float       ArgAsFloat (int index, const char*, float def)       const { return (float)_args[index].AsFloat (def); }
  const char* ArgAsString(int index, const char*, const char* def) const { return _args[index].AsString(def); }

  // Clip
  PVideoFrame GetFrame(PClip clip, int n, void *) {
    return clip->GetFrame(n, _env);
  }

  // Frame
  PVideoFrame Dup(PVideoFrame frame) {
    _env->MakeWritable(&frame);
    return frame;
  }

  PVideoFrame NewVideoFrame() {
    return _env->NewVideoFrame(vi);
  }

  void FreeFrame(const PVideoFrame frame) { }
  int ssw() const { return vi.GetPlaneWidthSubsampling(PLANAR_U); }
  int ssh() const { return vi.GetPlaneHeightSubsampling(PLANAR_U); }
  int ssw(int plane) const { return vi.GetPlaneWidthSubsampling(plane); }
  int ssh(int plane) const { return vi.GetPlaneHeightSubsampling(plane); }

  int stride(PVideoFrame& frame, int plane) const { return frame->GetPitch(plane); }
  int width (PVideoFrame& frame, int plane) const { return frame->GetRowSize(plane) / byte_per_channel; }
  int height(PVideoFrame& frame, int plane) const { return frame->GetHeight(plane); }
  int width () const { return vi.width;  }
  int height() const { return vi.height; }
  int pixelsize() const { return byte_per_channel; }
  int depth() const { return bit_per_channel; }
  int frames() const { return vi.num_frames; }
  int IsPlanar() const { return vi.IsPlanar(); }
  int IsYUV() const { return vi.IsYUV(); }
  int IsYUVA() const { return vi.IsYUVA(); }
  // int isRGB() const { return vi.IsRGB(); }
  int supported_pixel() const {
    return vi.pixel_type == VideoInfo::CS_I420 || (
      (
        (vi.pixel_type & ~VideoInfo::CS_Sample_Bits_Mask) == VideoInfo::CS_GENERIC_YUV420
        || (vi.pixel_type & ~VideoInfo::CS_Sample_Bits_Mask) == VideoInfo::CS_GENERIC_YUV422
        || (vi.pixel_type & ~VideoInfo::CS_Sample_Bits_Mask) == VideoInfo::CS_GENERIC_YUV444
      ) && (
        (vi.pixel_type & VideoInfo::CS_Sample_Bits_Mask) != VideoInfo::CS_Sample_Bits_32
      )
    );
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
    _env = env;
    return get(n, NULL);
  }
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) { child->GetAudio(buf, start, count, env); }
  const VideoInfo& __stdcall GetVideoInfo() { return child->GetVideoInfo(); }
  bool __stdcall GetParity(int n) { return child->GetParity(n); }
  int __stdcall SetCacheHints(int cachehints, int frame_range) { return 0; } ;

};

#endif
