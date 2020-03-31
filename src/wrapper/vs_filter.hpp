#ifndef __VS_FILTER_HPP__
#define __VS_FILTER_HPP__

#include <VapourSynth.h>
#include <VSHelper.h>

#define PLANAR_Y 0
#define PLANAR_U 1
#define PLANAR_V 2

#define PLANAR_R 0
#define PLANAR_G 1
#define PLANAR_B 2

#define PLANAR_A -1

class VSFilter {

public:
  class AFrame {
  protected:
    VSCore *_core;
    const VSAPI * _vsapi;
    bool readonly;
  public:
    VSVideoInfo _vi;
    VSFrameRef * _frame;
    AFrame(VSFrameRef * frame, VSCore *core, const VSAPI *vsapi, VSVideoInfo vi)
      : _core(core), _vsapi(vsapi), readonly(false), _vi(vi), _frame(frame) {
    }
    AFrame(const VSFrameRef * frame, VSCore *core, const VSAPI *vsapi, VSVideoInfo vi)
      : _core(core), _vsapi(vsapi), readonly(true), _vi(vi), _frame((VSFrameRef*)frame) {
    }
    AFrame* dup() {
      return new AFrame(_vsapi->copyFrame(_frame, _core), _core, _vsapi, _vi);
    }
    int stride(int plane = 0) const { return _vsapi->getStride(_frame, plane); }
    int width (int plane = 0) const { return _vsapi->getFrameWidth(_frame, plane); }
    int height(int plane = 0) const { return _vsapi->getFrameHeight(_frame, plane); }
    unsigned char* GetWritePtr(int plane = 0) const
      { return _vsapi->getWritePtr(_frame, plane); }
    const unsigned char* GetReadPtr(int plane = 0)
      { return _vsapi->getReadPtr(_frame, plane); }
  };

  class AVideoInfo {
  public:
    VSVideoInfo _vi;
    AVideoInfo() {}
    AVideoInfo(VSVideoInfo vi)
      : _vi(vi) {}
    bool HasVideo() const { return _vi.format != nullptr; }
    bool HasAudio() const { return false; } // Who cares?

    int BitsPerComponent() const { return _vi.format->bitsPerSample; }
    int ComponentSize() const { return _vi.format->bytesPerSample; }
    bool IsY() const { return _vi.format->colorFamily == cmGray; }
    bool IsPlanarRGBA() const { return false; }
    bool IsYUV() const { return _vi.format->colorFamily == cmYUV; }
    bool IsYUVA() const { return false; }
  };

  class AClip {
  protected:
    VSCore *_core;
    const VSAPI * _vsapi;
  public:
    VSNodeRef * _clip;
    VSVideoInfo _vi;
    AClip(VSNodeRef * clip, VSCore *core, const VSAPI *vsapi)
      : _core(core), _vsapi(vsapi), _clip(clip) {
      _vi = *_vsapi->getVideoInfo(_clip);
    }
    AVideoInfo vi() {
      return AVideoInfo(_vi);
    }
    AFrame* getFrame(int n) {
      return new AFrame(_vsapi->getFrame(n, _clip, nullptr, 0), _core, _vsapi, _vi);
    }

    int width()     const { return _vi.width; }
    int height()    const { return _vi.height; }
    int ssw()       const { return _vi.format->subSamplingW; }
    int ssh()       const { return _vi.format->subSamplingH; }
  };

  using Frametype = AFrame*;

  virtual AFrame* get(int n) {
    return child->getFrame(n);
  }
  virtual void initialize() {}
  virtual const char* name() const { return "VSFilter"; };
  virtual ~VSFilter(){};
protected:
  int byte_per_channel; // Useless here
  int bit_per_channel;  // Useless here
public:
  AVideoInfo vi;
  AClip* child;
  const VSMap *_in;
  VSCore * _core;
  const VSAPI * _vsapi;

  VSFilter(const VSMap *in, VSMap *out, VSCore *core, const VSAPI *vsapi)
    : _in(in), _core(core), _vsapi(vsapi) {
    child = ArgAsClip(0, "clip");
    vi = child->vi();
  }

  AClip* ArgAsClip(int, const char* name) const { return new AClip(_vsapi->propGetNode(_in, name, 0, 0), _core, _vsapi); }

  bool ArgAsBool(int, const char* name) const {
    return ArgAsInt(0, name) != 0;
  }
  int ArgAsInt(int, const char* name) const {
    int _err;
    auto data = int64ToIntS(_vsapi->propGetInt(_in, name, 0, &_err));
    if (_err) throw(name);
    return data;
  }
  float ArgAsFloat(int, const char* name) const {
    int _err;
    auto data = (float)_vsapi->propGetFloat(_in, name, 0, &_err);
    if (_err) throw(name);
    return data;
  }
  const char* ArgAsString(int, const char* name) const {
    int _err;
    auto data = _vsapi->propGetData(_in, name, 0, &_err);
    if (_err) throw(name);
    return data;
  }

  bool ArgAsBool(int, const char* name, bool def) const {
    return ArgAsInt(0, name, def ? 1 : 0) != 0;
  }
  int ArgAsInt(int, const char* name, int def) const {
    int _err;
    auto data = int64ToIntS(_vsapi->propGetInt(_in, name, 0, &_err));
    if (_err) return def;
    return data;
  }
  float ArgAsFloat(int, const char* name, float def) const {
    int _err;
    auto data = (float)_vsapi->propGetFloat(_in, name, 0, &_err);
    if (_err) return def;
    return data;
  }
  const char* ArgAsString(int, const char* name, const char* def) const {
    int _err;
    auto data = _vsapi->propGetData(_in, name, 0, &_err);
    if (_err) return def;
    return data;
  }

  // Clip
  AFrame* GetFrame(AClip* clip, int n) {
    return clip->getFrame(n);
  }

  // Frame
  AFrame* NewVideoFrame() {
    return new AFrame(_vsapi->newVideoFrame(vi._vi.format, vi._vi.width, vi._vi.height, NULL, _core), _core, _vsapi, vi._vi);
  }

  void FreeFrame(AFrame* frame) {
    _vsapi->freeFrame(frame->_frame);
  }
  int ssw() const { return vi._vi.format->subSamplingW; }
  int ssh() const { return vi._vi.format->subSamplingH; }

  int stride(AFrame* frame, int plane) const { return _vsapi->getStride(frame->_frame, plane); }
  int width (AFrame* frame, int plane) const { return _vsapi->getFrameWidth(frame->_frame, plane); }
  int height(AFrame* frame, int plane) const { return _vsapi->getFrameHeight(frame->_frame, plane); }
  int width () const { return vi._vi.width;  }
  int height() const { return vi._vi.height; }
  int depth () const { return vi._vi.format->bitsPerSample; }
  int frames() const { return vi._vi.numFrames; }

  bool supported_pixel() const {
    return vi._vi.format->colorFamily == cmYUV || vi._vi.format->colorFamily == cmRGB;
  }

  void VS_CC GetFramePre(VSFrameContext *_frameCtx, VSCore *_core, const VSAPI *vsapi, int n) {
    vsapi->requestFrameFilter(n, child->_clip, _frameCtx);
  }

  const VSFrameRef *VS_CC GetFrame(VSFrameContext *_frameCtx, VSCore *_core, const VSAPI *vsapi, int n) {
    auto frame = dynamic_cast<AFrame*>(get(n));
    return frame->_frame;
  }
};

#endif
