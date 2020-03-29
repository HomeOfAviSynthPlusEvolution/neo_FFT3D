#pragma once

#include <climits>
#include "FFT3DEngine.h"

template <typename Interface>
class FFT3D: public Interface {

protected:
  int process[4];
  FFT3DEngine<Interface>* engine[4];
  int plane_index[4];
  int engine_count = 0;
  int copy_count = 0;
  int planes;
  EngineParams* ep;

public:
  virtual const char* name() const override { return "neo_FFT3D"; }
  virtual void initialize() override {
    engine_count = 0;
    copy_count = 0;
    planes = 3;
    this->bit_per_channel = this->vi.BitsPerComponent();
    this->byte_per_channel = this->vi.ComponentSize();

    // Check input
    if (!this->vi.HasVideo())
      throw("where's the video?");
    if (!this->supported_pixel())
      throw("pixel type is not supported");
  
    float sigma1 = this->ArgAsFloat(1, "sigma", 2.0f);
    ep = new EngineParams{
      sigma1, // sigma
      this->ArgAsFloat( 2, "beta", (1.0f)), // beta
      this->ArgAsInt(   4, "bw", (32)), // bw - changed default from 48 to 32 in v.1.9.2
      this->ArgAsInt(   5, "bh", (32)), // bh - changed default from 48 to 32 in v.1.9.2
      this->ArgAsInt(   6, "bt", (3)), //  bt (=0 for Kalman mode) // new default=3 in v.0.9.3
      this->ArgAsInt(   7, "ow", (-1)), //  ow
      this->ArgAsInt(   8, "oh", (-1)), //  oh
      this->ArgAsFloat( 9, "kratio", (2.0f)), // kratio for Kalman mode
      this->ArgAsFloat(10, "sharpen", (0.0f)), // sharpen strength
      this->ArgAsFloat(11, "scutoff", (0.3f)), // sharpen cutoff frequency (relative to max) - v1.7
      this->ArgAsFloat(12, "svr", (1.0f)), // svr - sharpen vertical ratio
      this->ArgAsFloat(13, "smin", (4.0f)), // smin -  minimum limit for sharpen (prevent noise amplifying)
      this->ArgAsFloat(14, "smax", (20.0f)), // smax - maximum limit for sharpen (prevent oversharping)
      this->ArgAsBool( 15, "measure", (true)), // measure - switched to true in v.0.9.2
      this->ArgAsBool( 16, "interlaced", (false)), // interlaced - v.1.3
      this->ArgAsInt(  17, "wintype", (0)), // wintype - v1.4, v1.8
      this->ArgAsInt(  18, "pframe", (0)), //  pframe
      this->ArgAsInt(  19, "px", (0)), //  px
      this->ArgAsInt(  20, "py", (0)), //  py
      this->ArgAsBool( 21, "pshow", (false)), //  pshow
      this->ArgAsFloat(22, "pcutoff", (0.1f)), //  pcutoff
      this->ArgAsFloat(23, "pfactor", (0.0f)), //  pfactor
      this->ArgAsFloat(24, "sigma2", (sigma1)), // sigma2
      this->ArgAsFloat(25, "sigma3", (sigma1)), // sigma3
      this->ArgAsFloat(26, "sigma4", (sigma1)), // sigma4
      this->ArgAsFloat(27, "degrid", (1.0f)), // degrid
      this->ArgAsFloat(28, "dehalo", (0.0f)), // dehalo - v 1.9
      this->ArgAsFloat(29, "hr", (2.0f)), // halo radius - v 1.9
      this->ArgAsFloat(30, "ht", (50.0f)), // halo threshold - v 1.9
      this->ArgAsInt(  31, "ncpu", (1)), //  ncpu
      this->ArgAsInt(  35, "l", (0)),
      this->ArgAsInt(  36, "t", (0)),
      this->ArgAsInt(  37, "r", (0)),
      this->ArgAsInt(  38, "b", (0))
    };
    process[0] = this->ArgAsInt(32, "y", 3);
    process[1] = this->ArgAsInt(33, "u", 3);
    process[2] = this->ArgAsInt(34, "v", 3);
    process[3] = 2;

    if (this->vi.IsY8()) planes = 1;
    else if (this->vi.IsYUVA()) planes = 4;
    else if (this->vi.IsPlanarRGBA()) planes = 4;

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A };
    int *planes = (this->vi.IsYUV() || this->vi.IsYUVA()) ? planes_y : planes_r;

    for (int i = 0; i < 4; i++) {
      plane_index[i] = planes[i];
      if (process[i] == 3) {
        engine[i] = new FFT3DEngine<Interface>(this, *ep, planes[i]);
        engine_count++;
      }
      else if (process[i] == 2) {
        copy_count++;
      }
    }
  }

  virtual typename Interface::AFrame get(int n) override {
  // virtual auto get(int n) -> decltype(Interface::get(n)) {
    if (engine_count == 1 && copy_count == 0)
      for (int i = 0; i < planes; i++)
        if (process[i] == 3)
          return engine[i]->GetFrame(n);

    auto src = this->GetFrame(this->child, n);
    if (engine_count == 0) return src;
    auto dst = this->NewVideoFrame();

    for (int i = 0; i < planes; i++)
    {
      auto frame = src;
      if (process[i] == 3)
        frame = engine[i]->GetFrame(n);
      else if(process[i] == 2)
        ;
      else
        continue;
      auto idx = plane_index[i];
      memcpy(dst->GetWritePtr(idx), frame->GetReadPtr(idx), frame->GetPitch(idx) * frame->GetHeight(idx));
      if (frame != src)
        this->FreeFrame(frame);
    }

    this->FreeFrame(src);
    return dst;
  }

  // Auto register AVS+ mode: serialized
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? (ep->bt==0 ? MT_SERIALIZED : MT_MULTI_INSTANCE) : 0;
  }

  ~FFT3D() {
    for (int i = 0; i < planes; i++)
      if (process[i] == 3)
        delete engine[i];
  }

public:
  using Interface::Interface;

};
