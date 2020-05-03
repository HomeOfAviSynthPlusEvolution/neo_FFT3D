/*
 * Copyright 2020 Xinyue Lu
 *
 * DualSynth bridge - filter.
 *
 */

#pragma once

#include "fft3d_engine.h"
#include <atomic>
#include <queue>
#include <thread>
#include <functional>
#include <chrono>
#include <execution>
using namespace std::chrono_literals;

struct FFT3D final : Filter {
  int process[4];
  FFT3DEngine* engine[4];
  int plane_index[4];
  int engine_count {0};
  int copy_count {0};
  EngineParams* ep {nullptr};

  bool crop;

  const char* VSName() const override { return "FFT3D"; }
  const char* AVSName() const override { return "neo_fft3d"; }
  const MtMode AVSMode() const override { return ep->bt == 0 ? MT_SERIALIZED : MT_NICE_FILTER; }
  const VSFilterMode VSMode() const override { return ep->bt == 0 ? fmSerial : fmParallel; }
  const std::vector<Param> Params() const override {
    return std::vector<Param> {
      Param {"clip", Clip, false, true, true, false},
      Param {"sigma", Float},
      Param {"beta", Float},
      Param {"planes", Integer, true, false, true},
      Param {"bw", Integer},
      Param {"bh", Integer},
      Param {"bt", Integer},
      Param {"ow", Integer},
      Param {"oh", Integer},
      Param {"kratio", Float},
      Param {"sharpen", Float},
      Param {"scutoff", Float},
      Param {"svr", Float},
      Param {"smin", Float},
      Param {"smax", Float},
      Param {"measure", Boolean},
      Param {"interlaced", Boolean},
      Param {"wintype", Integer},
      Param {"pframe", Integer},
      Param {"px", Integer},
      Param {"py", Integer},
      Param {"pshow", Boolean},
      Param {"pcutoff", Float},
      Param {"pfactor", Float},
      Param {"sigma2", Float},
      Param {"sigma3", Float},
      Param {"sigma4", Float},
      Param {"degrid", Float},
      Param {"dehalo", Float},
      Param {"hr", Float},
      Param {"ht", Float},
      Param {"y", Integer, false, true, false},
      Param {"u", Integer, false, true, false},
      Param {"v", Integer, false, true, false},
      Param {"l", Integer},
      Param {"t", Integer},
      Param {"r", Integer},
      Param {"b", Integer},
      Param {"opt", Integer}
    };
  }
  void Initialize(InDelegator* in, DSVideoInfo in_vi, FetchFrameFunctor* fetch_frame) override
  {
    Filter::Initialize(in, in_vi, fetch_frame);
    float sigma1 = 2.0f;
    in->Read("sigma", sigma1);
    ep = new EngineParams {
      sigma1, 1.0f,
      32, 32,
      3,
      -1, -1,
      2.0f,
      0.0f,
      0.3f,
      1.0f,
      4.0f, 20.0f,
      true,
      false,
      0,
      0,
      0, 0,
      false,
      0.1f,
      0.0f,
      sigma1, sigma1, sigma1,
      1.0f,
      0.0f,
      2.0f, 50.0f,
      0, 0, 0, 0,
      0,
      in_vi
    };
    in->Read("beta", ep->beta);
    in->Read("bw", ep->bw);
    in->Read("bh", ep->bh);
    in->Read("bt", ep->bt);
    in->Read("ow", ep->ow);
    in->Read("oh", ep->oh);
    in->Read("kratio", ep->kratio);
    in->Read("sharpen", ep->sharpen);
    in->Read("scutoff", ep->scutoff);
    in->Read("svr", ep->svr);
    in->Read("smin", ep->smin);
    in->Read("smax", ep->smax);
    in->Read("measure", ep->measure);
    in->Read("interlaced", ep->interlaced);
    in->Read("wintype", ep->wintype);
    in->Read("pframe", ep->pframe);
    in->Read("px", ep->px);
    in->Read("py", ep->py);
    in->Read("pshow", ep->pshow);
    in->Read("pcutoff", ep->pcutoff);
    in->Read("pfactor", ep->pfactor);
    in->Read("sigma2", ep->sigma2);
    in->Read("sigma3", ep->sigma3);
    in->Read("sigma4", ep->sigma4);
    in->Read("degrid", ep->degrid);
    in->Read("dehalo", ep->dehalo);
    in->Read("hr", ep->hr);
    in->Read("ht", ep->ht);
    in->Read("l", ep->l); ep->l = MAX(ep->l, 0);
    in->Read("t", ep->t); ep->t = MAX(ep->t, 0);
    in->Read("r", ep->r); ep->r = MAX(ep->r, 0);
    in->Read("b", ep->b); ep->b = MAX(ep->b, 0);
    in->Read("opt", ep->opt);

    this->crop = ep->l > 0 || ep->r > 0 || ep->t > 0 || ep->b > 0;

    if (ep->l + ep->r >= ep->vi.Width)
      throw "Width cannot be cropped to zero or below";

    if (ep->t + ep->b >= ep->vi.Height)
      throw "Height cannot be cropped to zero or below";

    try {
      process[0] =
      process[1] =
      process[2] =
      process[3] = 2;
      std::vector<int> user_planes {0, 1, 2};
      in->Read("planes", user_planes);
      for (auto &&p : user_planes)
      {
        if (p < ep->vi.Format.Planes)
          process[p] = 3;
      }
    }
    catch (const char *) {
      process[0] =
      process[1] =
      process[2] = 3;
      in->Read("y", process[0]);
      in->Read("u", process[1]);
      in->Read("v", process[2]);
    }

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A };
    int *planes = (ep->vi.Format.IsFamilyYUV) ? planes_y : planes_r;

    for (int i = 0; i < ep->vi.Format.Planes; i++) {
      plane_index[i] = planes[i];
      if (process[i] == 3) {
        ep->IsChroma = ep->vi.Format.IsFamilyYUV && i != 0;
        ep->framewidth = ep->IsChroma ? ep->vi.Width >> ep->vi.Format.SSW : ep->vi.Width;
        ep->frameheight = ep->IsChroma ? ep->vi.Height >> ep->vi.Format.SSH : ep->vi.Height;
        engine[i] = new FFT3DEngine(*ep, i, fetch_frame);
        engine_count++;
      }
      else if (process[i] == 2) {
        copy_count++;
      }
    }
  }

  std::vector<int> RequestReferenceFrames(int n) const override
  {
    std::vector<int> req;

    if (ep->bt > 1 && engine_count > 0) {
      int from = MAX(n - ep->bt / 2, 0);
      int to = MIN(n + (ep->bt - 1) / 2, ep->vi.Frames - 1);
      for (int i = from; i <= to; i++) {
        bool exist = true;
        for (int j = 0; j < ep->vi.Format.Planes; j++)
          if (process[j] == 3)
            exist = exist && engine[j]->CacheRefresh(i);
        req.push_back(i);
      }
    }
    else
      req.push_back(n);
    return req;
  }

  DSFrame GetFrame(int n, std::unordered_map<int, DSFrame> in_frames) override
  {
    if (engine_count == 1 && copy_count == 0 && !this->crop)
      for (int i = 0; i < ep->vi.Format.Planes; i++)
        if (process[i] == 3)
          return engine[i]->GetFrame(n, in_frames);

    auto src = in_frames[n];
    if (engine_count == 0) return src;
    auto dst = src.Create(false);

    std::for_each_n(std::execution::par_unseq, reinterpret_cast<char*>(0), ep->vi.Format.Planes, [&](char&idx) {
      int i = static_cast<int>(reinterpret_cast<intptr_t>(&idx));
      bool chroma = ep->vi.Format.IsFamilyYUV && i > 0 && i < 3;

      if (process[i] == 3) {
        auto l = chroma ? (ep->l >> ep->vi.Format.SSW) : ep->l;
        auto r = chroma ? (ep->r >> ep->vi.Format.SSW) : ep->r;
        auto t = chroma ? (ep->t >> ep->vi.Format.SSH) : ep->t;
        auto b = chroma ? (ep->b >> ep->vi.Format.SSH) : ep->b;
        auto frame = engine[i]->GetFrame(n, in_frames);
        copy_frame(dst, src, frame, i, chroma, l, t, r, b);
      }
      else if(process[i] == 2) {
        copy_frame(dst, src, i, chroma);
      }
    });

    return dst;
  }

  int SetCacheHints(int cachehints, int frame_range) override
  {
    return cachehints == CACHE_GET_MTMODE ? (ep->bt == 0 ? MT_SERIALIZED : MT_NICE_FILTER) : 0;
  }

  void copy_frame(DSFrame &dst, DSFrame &src, int plane, bool chroma)
  {
    auto height = ep->vi.Height;
    auto stride = src.StrideBytes[plane];
    auto src_ptr = src.SrcPointers[plane];
    auto dst_ptr = dst.DstPointers[plane];
    if (chroma)
      height >>= ep->vi.Format.SSH;
    memcpy(dst_ptr, src_ptr, stride * height);
  }

  void copy_frame(DSFrame &dst, DSFrame &src, DSFrame &processed, int plane, bool chroma, int l, int t, int r, int b)
  {
    auto width = ep->vi.Width * ep->vi.Format.BytesPerSample;
    auto height = ep->vi.Height;
    auto stride = processed.StrideBytes[plane];
    auto src_ptr = src.SrcPointers[plane];
    auto pcs_ptr = processed.SrcPointers[plane];
    auto dst_ptr = dst.DstPointers[plane];
    if (chroma) {
      width >>= ep->vi.Format.SSW;
      height >>= ep->vi.Format.SSH;
    }
    auto l2 = l * ep->vi.Format.BytesPerSample;
    auto r2 = r * ep->vi.Format.BytesPerSample;

    if (l == 0 && r == 0 && t == 0 && b == 0) {
      memcpy(dst_ptr, pcs_ptr, stride * height);
      return;
    }

    if (t > 0)
      memcpy(dst_ptr, src_ptr, stride * t);
    src_ptr += stride * t;
    dst_ptr += stride * t;
    pcs_ptr += stride * t;
    for (int y = 0; y < (height - t - b); y++) {
      if (l2 > 0) memcpy(dst_ptr, src_ptr, l2);
      memcpy(dst_ptr + l2, pcs_ptr + l2, width - l2 - r2);
      if (r2 > 0) memcpy(dst_ptr + width - r2, src_ptr + width - r2, r2);
      src_ptr += stride;
      dst_ptr += stride;
      pcs_ptr += stride;
    }
    if (b > 0)
      memcpy(dst_ptr, src_ptr, stride * b);
  }

  ~FFT3D()
  {
    if (ep) {
      for (int i = 0; i < ep->vi.Format.Planes; i++)
        if (process[i] == 3)
          delete engine[i];
      delete ep;
    }
  }
};
