/*
 * Copyright 2004-2007 A.G.Balakhnin aka Fizick
 * Copyright 2015 martin53
 * Copyright 2017-2019 Ferenc Pinter aka printerf
 * Copyright 2020 Xinyue Lu
 *
 * Main file.
 *
 */

#pragma once

#include "fft3d_common.h"
#include "engine/filter_backend.hpp"
#include "fft/fftw_lock.hpp"
#include "cache.hpp"
#include "engine/engine_workspace.hpp"
#include "engine/frame_buffer.hpp"
#include "engine/frame_views.hpp"
#include "engine/pattern_analysis.hpp"

#include <algorithm>
#include <dualsynth/video_filter.hpp>

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

class FFT3DEngine {
public:
  std::unique_ptr<EngineParams> ep;
  std::unique_ptr<IOParams> iop;
  int plane; // color plane
  std::unique_ptr<neo_fft3d::engine::FilterBackend> backend;

  std::unique_ptr<neo_fft3d::fft::FFTPlan> plan, planinv, plan1;

  AlignedVector<std::complex<float>> gridsample; //v1.8
  AlignedVector<std::complex<float>> outLast;
  AlignedVector<std::complex<float>> covar;
  AlignedVector<std::complex<float>> covarProcess;

  AlignedVector<float> wsharpen;
  AlignedVector<float> wdehalo;
  AlignedVector<float> pwin;
  AlignedVector<float> pattern2d;
  AlignedVector<float> pattern3d;

  neo_fft3d::EngineWorkspace workspace;

  std::unique_ptr<cache<std::complex<float>>> fftcache;

  char messagebuf[80] {0};

  int outwidth;
  int outpitch; //v.1.7
  int insize;
  int outsize;
  int howmanyblocks;

  float sigmaSquaredNoiseNormed2D {0};
  float sigmaNoiseNormed2D {0};
  float sigmaMotionNormed {0};
  float sigmaSquaredSharpenMinNormed {0};
  float sigmaSquaredSharpenMaxNormed {0};
  float ht2n {0}; // halo threshold squared normed
  float norm {0}; // normalization factor

  int coverwidth;
  int coverheight;
  int coverpitch;

  int mirw; // mirror width for padding
  int mirh; // mirror height for padding

  bool isPatternSet;
  float psigma {0};

  std::mutex cache_mutex;

  bool pattern3d_initialized {false};
  std::mutex init3d_mutex;

private:
  neo_fft3d::engine::CoverPlaneGeometry cover_geometry() const {
    return neo_fft3d::engine::CoverPlaneGeometry{
      coverwidth,
      coverheight,
      coverpitch,
      ep->vi.Format.BytesPerSample
    };
  }

  neo_fft3d::engine::FrameViewAdapter frame_views() const {
    return neo_fft3d::engine::FrameViewAdapter{
      cover_geometry(),
      static_cast<std::size_t>(insize)
    };
  }

  neo_fft3d::FloatPlaneView kernel_float_view(AlignedVector<float>& data) const {
    return neo_fft3d::make_float_plane_view(
      data.data(),
      outpitch,
      ep->bh,
      static_cast<std::ptrdiff_t>(outpitch) * static_cast<std::ptrdiff_t>(sizeof(float))
    );
  }

  neo_fft3d::ConstFloatPlaneView kernel_float_view(const AlignedVector<float>& data) const {
    return neo_fft3d::make_float_plane_view(
      data.data(),
      outpitch,
      ep->bh,
      static_cast<std::ptrdiff_t>(outpitch) * static_cast<std::ptrdiff_t>(sizeof(float))
    );
  }

  neo_fft3d::ComplexBlockView kernel_complex_blocks(AlignedVector<std::complex<float>>& data, int block_count) const {
    return neo_fft3d::make_complex_block_view(data.data(), outpitch, ep->bh, block_count);
  }

  neo_fft3d::ComplexBlockView kernel_complex_blocks(std::complex<float>* data, int block_count) const {
    return neo_fft3d::make_complex_block_view(data, outpitch, ep->bh, block_count);
  }

  void validate_backend() const {
    if (!backend) {
      throw std::invalid_argument("neo_fft3d: filter backend is required");
    }
  }

  void normalize_params_for_sample_format() {
    float factor;
    switch(ep->vi.Format.BytesPerSample) {
      case 1: factor = 1.0F; break;
      case 2: factor = float(1 << (ep->vi.Format.BitsPerSample-8)); break;
      default: factor = 1 / 255.0F;
    }

    ep->sigma *= factor;
    ep->sigma2 *= factor;
    ep->sigma3 *= factor;
    ep->sigma4 *= factor;
    ep->smin *= factor;
    ep->smax *= factor;
  }

  void validate_overlap_params_and_apply_defaults() {
    if (ep->ow * 2 > ep->bw) {
      throw("ow must be less than bw / 2");
    }
    if (ep->oh * 2 > ep->bh) {
      throw("oh must be less than bh / 2");
    }
    if (ep->ow < 0) {
      ep->ow = ep->bw / 3; // changed from ep->bw/4 to ep->bw/3 in v.1.2
    }
    if (ep->oh < 0) {
      ep->oh = ep->bh / 3; // changed from bh/4 to bh/3 in v.1.2
    }
  }

  void validate_temporal_params() const {
    if (ep->bt < -1 || ep->bt > 5) {
      throw("bt must be -1 (Sharpen), 0 (Kalman), 1..5 (Wiener)");
    }
  }

  void configure_geometry() {
    int xRatioShift = ep->IsChroma ? ep->vi.Format.SSW : 0;
    int yRatioShift = ep->IsChroma ? ep->vi.Format.SSH : 0;

    iop->nox = (((ep->vi.Width - ep->l - ep->r) >> xRatioShift) - ep->ow + (ep->bw - ep->ow - 1)) / (ep->bw - ep->ow);
    iop->noy = (((ep->vi.Height - ep->t - ep->b) >> yRatioShift) - ep->oh + (ep->bh - ep->oh - 1)) / (ep->bh - ep->oh);


    // padding by 1 block per side
    iop->nox += 2;
    iop->noy += 2;
    mirw = ep->bw - ep->ow; // set mirror size as block interval
    mirh = ep->bh - ep->oh;

    if (ep->beta < 1) {
      throw("beta must be not less 1.0");
    }

    coverwidth = iop->nox*(ep->bw - ep->ow) + ep->ow;
    coverheight = iop->noy*(ep->bh - ep->oh) + ep->oh;
    coverpitch = ((coverwidth + 15) / 16) * 16; // align to 16 elements. Pitch is element-granularity. For byte pitch, multiply is by pixelsize

    insize = ep->bw*ep->bh*iop->nox*iop->noy;
    insize = ((insize + 15) / 16) * 16;
    outwidth = ep->bw / 2 + 1; // width (pitch) of complex fft block
    outpitch = ((outwidth + 15) / 16) * 16; // must be even for SSE - v1.7
    outsize = outpitch*ep->bh*iop->nox*iop->noy; // replace outwidth to outpitch here and below in v1.7
    howmanyblocks = iop->nox*iop->noy;
  }

  void configure_storage() {
    if (ep->bt == 0) // Kalman
    {
      outLast.resize(outsize);
      covar.resize(outsize);
      covarProcess.resize(outsize);
    }
    // in and out are thread dependent now // neo-r1
    workspace.configure(
      static_cast<std::size_t>(coverheight) * static_cast<std::size_t>(coverpitch) * static_cast<std::size_t>(ep->vi.Format.BytesPerSample),
      static_cast<std::size_t>(insize),
      static_cast<std::size_t>(outsize)
    );
    gridsample.resize(outsize); //v1.8

    if (ep->bt > 1) {
      fftcache = std::make_unique<cache<std::complex<float>>>(ep->bt + 3, outsize);
    }
  }

  void create_main_plans(const neo_fft3d::fft::PlanOptions& plan_options, const neo_fft3d::fft::PlanBuffers& plan_buffers) {
    neo_fft3d::fft::GlobalLockGuard fftw_lock;

    plan = backend->CreatePlan(
      ep->bh,
      ep->bw,
      outpitch,
      neo_fft3d::fft::Direction::r2c,
      howmanyblocks,
      plan_options,
      plan_buffers
    );

    planinv = backend->CreatePlan(
      ep->bh,
      ep->bw,
      outpitch,
      neo_fft3d::fft::Direction::c2r,
      howmanyblocks,
      plan_options,
      plan_buffers
    );
  }

  void resize_window_buffers() {
    iop->wanxl.resize(ep->ow);
    iop->wanxr.resize(ep->ow);
    iop->wanyl.resize(ep->oh);
    iop->wanyr.resize(ep->oh);

    iop->wsynxl.resize(ep->ow);
    iop->wsynxr.resize(ep->ow);
    iop->wsynyl.resize(ep->oh);
    iop->wsynyr.resize(ep->oh);

    wsharpen.resize(ep->bh*outpitch);
    wdehalo.resize(ep->bh*outpitch);
  }

  void initialize_overlap_windows() {
    // define analysis and synthesis windows
    // combining window (analize mult by synthesis) is raised cosine (Hanning)

    float pi = 3.1415926535897932384626433832795F;
    if (ep->wintype == 0) // window type
    { // , used in all version up to 1.3
      // half-cosine, the same for analysis and synthesis
      // define analysis windows
      for (int i = 0; i < ep->ow; i++)
      {
        iop->wanxl[i] = cosf(pi*(i - ep->ow + 0.5F) / (ep->ow * 2)); // left analize window (half-cosine)
        iop->wanxr[i] = cosf(pi*(i + 0.5F) / (ep->ow * 2)); // right analize window (half-cosine)
      }
      for (int i = 0; i < ep->oh; i++)
      {
        iop->wanyl[i] = cosf(pi*(i - ep->oh + 0.5F) / (ep->oh * 2));
        iop->wanyr[i] = cosf(pi*(i + 0.5F) / (ep->oh * 2));
      }
      // use the same windows for synthesis too.
      for (int i = 0; i < ep->ow; i++)
      {
        iop->wsynxl[i] = iop->wanxl[i]; // left  window (half-cosine)

        iop->wsynxr[i] = iop->wanxr[i]; // right  window (half-cosine)
      }
      for (int i = 0; i < ep->oh; i++)
      {
        iop->wsynyl[i] = iop->wanyl[i];
        iop->wsynyr[i] = iop->wanyr[i];
      }
    }
    else if (ep->wintype == 1) // added in v.1.4
    {
      // define analysis windows as more flat (to decrease grid)
      for (int i = 0; i < ep->ow; i++)
      {
        iop->wanxl[i] = sqrt(cosf(pi*(i - ep->ow + 0.5F) / (ep->ow * 2)));
        iop->wanxr[i] = sqrt(cosf(pi*(i + 0.5F) / (ep->oh * 2)));
      }
      for (int i = 0; i < ep->oh; i++)
      {
        iop->wanyl[i] = sqrt(cosf(pi*(i - ep->oh + 0.5F) / (ep->oh * 2)));
        iop->wanyr[i] = sqrt(cosf(pi*(i + 0.5F) / (ep->oh * 2)));
      }
      // define synthesis as supplenent to rised cosine (Hanning)
      for (int i = 0; i < ep->ow; i++)
      {
        iop->wsynxl[i] = iop->wanxl[i] * iop->wanxl[i] * iop->wanxl[i]; // left window
        iop->wsynxr[i] = iop->wanxr[i] * iop->wanxr[i] * iop->wanxr[i]; // right window
      }
      for (int i = 0; i < ep->oh; i++)
      {
        iop->wsynyl[i] = iop->wanyl[i] * iop->wanyl[i] * iop->wanyl[i];
        iop->wsynyr[i] = iop->wanyr[i] * iop->wanyr[i] * iop->wanyr[i];
      }
    }
    else //  (ep->wintype==2) - added in v.1.4
    {
      // define analysis windows as flat (to prevent grid)
      std::fill_n(iop->wanxl.data(), ep->ow, 1.0F);
      std::fill_n(iop->wanxr.data(), ep->ow, 1.0F);
      std::fill_n(iop->wanyl.data(), ep->oh, 1.0F);
      std::fill_n(iop->wanyr.data(), ep->oh, 1.0F);

      // define synthesis as rised cosine (Hanning)
      for (int i = 0; i < ep->ow; i++)
      {
        iop->wsynxl[i] = cosf(pi*(i - ep->ow + 0.5F) / (ep->ow * 2));
        iop->wsynxl[i] = iop->wsynxl[i] * iop->wsynxl[i];// left window (rised cosine)
        iop->wsynxr[i] = cosf(pi*(i + 0.5F) / (ep->ow * 2));
        iop->wsynxr[i] = iop->wsynxr[i] * iop->wsynxr[i]; // right window (falled cosine)
      }
      for (int i = 0; i < ep->oh; i++)
      {
        iop->wsynyl[i] = cosf(pi*(i - ep->oh + 0.5F) / (ep->oh * 2));
        iop->wsynyl[i] = iop->wsynyl[i] * iop->wsynyl[i];
        iop->wsynyr[i] = cosf(pi*(i + 0.5F) / (ep->oh * 2));
        iop->wsynyr[i] = iop->wsynyr[i] * iop->wsynyr[i];
      }
    }
  }

  void initialize_sharpen_window() {
    auto *tmp_wsharpen = wsharpen.data();
    for (int j = 0; j < ep->bh; j++)
    {
      int dj = j;
      if (j >= ep->bh / 2) {
        dj = ep->bh - j;
      }
      float d2v = float(dj*dj)*(ep->svr*ep->svr) / ((ep->bh / 2)*(ep->bh / 2)); // v1.7
      for (int i = 0; i < outwidth; i++)
      {
        float d2 = d2v + (float(i*i) / ((ep->bw / 2)*(ep->bw / 2))); // distance_2 - v1.7
        tmp_wsharpen[i] = 1 - exp(-d2 / (2 * ep->scutoff*ep->scutoff));
      }
      tmp_wsharpen += outpitch;
    }
  }

  void initialize_dehalo_window() {
    auto *tmp_wdehalo = wdehalo.data();
    float wmax = 0;
    for (int j = 0; j < ep->bh; j++)
    {
      int dj = j;
      if (j >= ep->bh / 2) {
        dj = ep->bh - j;
      }
      float d2v = float(dj*dj)*(ep->svr*ep->svr) / ((ep->bh / 2)*(ep->bh / 2));
      for (int i = 0; i < outwidth; i++)
      {
        float d2 = d2v + (float(i*i) / ((ep->bw / 2)*(ep->bw / 2))); // squared distance in frequency domain
        tmp_wdehalo[i] = exp(-0.7F*d2*ep->hr*ep->hr) - exp(-d2*ep->hr*ep->hr); // some window with max around 1/hr, small at low and high frequencies
        wmax = std::max(tmp_wdehalo[i], wmax); // for normalization
      }
      tmp_wdehalo += outpitch;
    }

    tmp_wdehalo = wdehalo.data();
    for (int j = 0; j < ep->bh; j++)
    {
      for (int i = 0; i < outwidth; i++)
      {
        tmp_wdehalo[i] /= wmax; // normalize
      }
      tmp_wdehalo += outpitch;
    }
  }

  void initialize_filter_windows() {
    initialize_overlap_windows();
    initialize_sharpen_window();
    initialize_dehalo_window();
  }

  void initialize_normalization_params() {
    norm = 1.0F / (ep->bw*ep->bh); // do not forget set FFT normalization factor

    sigmaSquaredNoiseNormed2D = ep->sigma*ep->sigma / norm;
    sigmaNoiseNormed2D = ep->sigma / sqrtf(norm);
    sigmaMotionNormed = ep->sigma*ep->kratio / sqrtf(norm);
    sigmaSquaredSharpenMinNormed = ep->smin*ep->smin / norm;
    sigmaSquaredSharpenMaxNormed = ep->smax*ep->smax / norm;
    ht2n = ep->ht*ep->ht / norm; // halo threshold squared and normed - v1.9
  }

  void initialize_kalman_buffers() {
    if (ep->bt == 0) // Kalman
    {
      std::fill_n(reinterpret_cast<float*>(outLast.data()), outsize * 2, 0.0F);
      std::fill_n(reinterpret_cast<float*>(covar.data()), outsize * 2, sigmaSquaredNoiseNormed2D);
      std::fill_n(reinterpret_cast<float*>(covarProcess.data()), outsize * 2, sigmaSquaredNoiseNormed2D);
    }
  }

  void initialize_pattern_buffers() {
    pwin.resize(ep->bh*outpitch); // pattern window array

    float fw2;
    float fh2;
    auto *tmp_pwin = pwin.data();
    for (int j = 0; j < ep->bh; j++)
    {
      if (j < ep->bh / 2) {
        fh2 = (j*2.0F / ep->bh)*(j*2.0F / ep->bh);
      } else {
        fh2 = ((ep->bh - 1 - j)*2.0F / ep->bh)*((ep->bh - 1 - j)*2.0F / ep->bh);
      }
      for (int i = 0; i < outwidth; i++)
      {
        fw2 = (i*2.0F / ep->bw)*(j*2.0F / ep->bw);
        tmp_pwin[i] = (fh2 + fw2) / (fh2 + fw2 + ep->pcutoff*ep->pcutoff);
      }
      tmp_pwin += outpitch;
    }

    pattern2d.resize(ep->bh*outpitch); // noise pattern window array
    pattern3d.resize(ep->bh*outpitch); // noise pattern window array

    if ((ep->sigma2 != ep->sigma || ep->sigma3 != ep->sigma || ep->sigma4 != ep->sigma) && ep->pfactor == 0)
    {// we have different sigmas, so create pattern from sigmas
      neo_fft3d::SigmasToPattern(ep->sigma, ep->sigma2, ep->sigma3, ep->sigma4, outwidth, norm, kernel_float_view(pattern2d));
      isPatternSet = true;
      ep->pfactor = 1;
    }
    else
    {
      isPatternSet = false; // pattern must be estimated
    }
  }

  void create_grid_plan(const neo_fft3d::fft::PlanOptions& plan_options, const neo_fft3d::fft::PlanBuffers& plan_buffers) {
    neo_fft3d::fft::GlobalLockGuard fftw_lock;

    plan1 = backend->CreatePlan(
      ep->bh,
      ep->bw,
      outpitch,
      neo_fft3d::fft::Direction::r2c,
      1,
      plan_options,
      plan_buffers
    ); // 1 block
  }

  void initialize_grid_sample(float* in, const neo_fft3d::fft::PlanOptions& plan_options,
                              const neo_fft3d::fft::PlanBuffers& plan_buffers) {
    create_grid_plan(plan_options, plan_buffers);

    const auto views = frame_views();
    std::vector<byte> coverbuf(coverheight*coverpitch*ep->vi.Format.BytesPerSample);
    // avs+
    switch (ep->vi.Format.BytesPerSample) {
    case 1: std::memset(coverbuf.data(), 255, coverbuf.size());
      break;
    case 2: std::fill_n(reinterpret_cast<uint16_t*>(coverbuf.data()), coverheight*coverpitch, (1 << ep->vi.Format.BitsPerSample) - 1);
      break; // 255
    case 4: std::fill_n(reinterpret_cast<float*>(coverbuf.data()), coverheight*coverpitch, 1.0F);
      break; // 255
    }
    CoverToOverlap(ep.get(), iop.get(), views.Overlap(in), views.Cover(coverbuf.data()), false);
    // make FFT 2D
    plan1->Execute(in, gridsample.data(), 1);
  }

  void ensure_fft_cache_capacity(std::size_t workspace_slot_count) {
    if (!fftcache) {
      return;
    }
    const auto required_cache_size = static_cast<std::size_t>(ep->bt) + workspace_slot_count + 2;
    std::lock_guard<std::mutex> lock_cache(cache_mutex);
    fftcache->resize(required_cache_size);
  }

  void transform_source_plane_to_spectrum(
    const ds::PlaneView& src_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* spectrum,
    bool chroma
  ) {
    FrameToCover(ep.get(), plane, views.Source(src_plane), views.MutableCover(coverbuf), mirw, mirh);
    CoverToOverlap(ep.get(), iop.get(), views.Overlap(in), views.Cover(coverbuf), chroma);
    plan->Execute(in, spectrum, howmanyblocks);
  }

  void write_spectrum_to_destination(
    std::complex<float>* spectrum,
    float* in,
    byte* coverbuf,
    const neo_fft3d::engine::FrameViewAdapter& views,
    const ds::MutablePlaneView& dst_plane,
    bool chroma
  ) {
    planinv->Execute(spectrum, in, howmanyblocks);
    OverlapToCover(ep.get(), iop.get(), views.Overlap(in), norm, views.MutableCover(coverbuf), chroma);
    CoverToFrame(ep.get(), plane, views.Cover(coverbuf), views.Destination(dst_plane), mirw, mirh);
  }

  void show_pattern_frame(
    int n,
    int pxf,
    int pyf,
    ds::MutablePlaneView& dst_plane,
    const ds::PlaneView& src_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez
  ) {
    // change analysis and synthesis window to constant to show
    std::fill_n(iop->wanxl.data(), ep->ow, 1.0F);
    std::fill_n(iop->wanxr.data(), ep->ow, 1.0F);
    std::fill_n(iop->wsynxl.data(), ep->ow, 1.0F);
    std::fill_n(iop->wsynxr.data(), ep->ow, 1.0F);
    std::fill_n(iop->wanyl.data(), ep->oh, 1.0F);
    std::fill_n(iop->wanyr.data(), ep->oh, 1.0F);
    std::fill_n(iop->wsynyl.data(), ep->oh, 1.0F);
    std::fill_n(iop->wsynyr.data(), ep->oh, 1.0F);

    // put source bytes to float array of overlapped blocks
    // cur frame
    transform_source_plane_to_spectrum(src_plane, views, coverbuf, in, outrez, !ep->vi.Format.IsFamilyRGB);

    neo_fft3d::PutPatternOnly(kernel_complex_blocks(outrez, howmanyblocks), outwidth, iop->nox, iop->noy, pxf, pyf);
    // do inverse 2D FFT, get filtered 'in' array
    write_spectrum_to_destination(outrez, in, coverbuf, views, dst_plane, !ep->vi.Format.IsFamilyRGB);
    int psigmaint = ((int)(10 * psigma)) / 10;
    int psigmadec = (int)((psigma - psigmaint) * 10);
    wsprintf(messagebuf, " frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);
    // TODO: DrawString(dst, vi, 0, 0, messagebuf);
  }

  bool handle_pattern_frame(
    int n,
    ds::VideoFrameProvider& provider,
    ds::MutableVideoFrameView& dst,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez
  ) {
    if (ep->pfactor == 0) {
      return false;
    }

    neo_fft3d::fft::GlobalLockGuard fftw_lock;
    if (ep->pfactor != 0 && !isPatternSet && !ep->pshow) // get noise pattern
    {
      auto psrc = neo_fft3d::engine::FetchFrame(provider, ep->pframe);
      const auto& psrc_plane = psrc.plane(plane);
      ep->framepitch = static_cast<int>(psrc_plane.stride_bytes) / ep->vi.Format.BytesPerSample;

      // put source bytes to float array of overlapped blocks
      transform_source_plane_to_spectrum(psrc_plane, views, coverbuf, in, outrez, ep->IsChroma);
      if (ep->px == 0 && ep->py == 0) { // try find pattern block with minimal noise sigma
        neo_fft3d::FindPatternBlock(kernel_complex_blocks(outrez, howmanyblocks), outwidth, iop->nox, iop->noy, ep->px, ep->py, kernel_float_view(pwin), ep->degrid, kernel_complex_blocks(gridsample, 1));
      }
      neo_fft3d::SetPattern(kernel_complex_blocks(outrez, howmanyblocks), outwidth, iop->nox, iop->noy, ep->px, ep->py, kernel_float_view(pwin), kernel_float_view(pattern2d), psigma, ep->degrid, kernel_complex_blocks(gridsample, 1));
      isPatternSet = true;
    }
    else if (ep->pfactor != 0 && ep->pshow)
    {
      int pxf;
      int pyf;

      // show noise pattern window
      auto src = neo_fft3d::engine::FetchFrame(provider, n);
      const auto& src_plane = src.plane(plane);
      auto& dst_plane = dst.plane(plane);
      ep->framepitch = static_cast<int>(src_plane.stride_bytes) / ep->vi.Format.BytesPerSample;
      ep->framepitch_dst = static_cast<int>(dst_plane.stride_bytes) / ep->vi.Format.BytesPerSample;

      // put source bytes to float array of overlapped blocks
      transform_source_plane_to_spectrum(src_plane, views, coverbuf, in, outrez, ep->IsChroma);
      if (ep->px == 0 && ep->py == 0) { // try find pattern block with minimal noise sigma
        neo_fft3d::FindPatternBlock(kernel_complex_blocks(outrez, howmanyblocks), outwidth, iop->nox, iop->noy, pxf, pyf, kernel_float_view(pwin), ep->degrid, kernel_complex_blocks(gridsample, 1));
      } else
      {
        pxf = ep->px; // fixed bug in v1.6
        pyf = ep->py;
      }
      neo_fft3d::SetPattern(kernel_complex_blocks(outrez, howmanyblocks), outwidth, iop->nox, iop->noy, pxf, pyf, kernel_float_view(pwin), kernel_float_view(pattern2d), psigma, ep->degrid, kernel_complex_blocks(gridsample, 1));
      show_pattern_frame(n, pxf, pyf, dst_plane, src_plane, views, coverbuf, in, outrez);
      return true;
    }

    return false;
  }

  int current_bt_for_frame(int n) const {
    int btcur = ep->bt; // bt used for current frame
  //	if ( (bt/2 > n) || bt==3 && n==vi.num_frames-1 )
    if ((ep->bt / 2 > n) || (ep->bt - 1) / 2 > (ep->vi.Frames - 1 - n)) {
      btcur = 1; //	do 2D filter for first and last frames
    }
    return btcur;
  }

  SharedFunctionParams make_shared_function_params(int btcur) {
    return SharedFunctionParams {
      outwidth,
      outpitch,
      ep->bh,
      howmanyblocks,
      btcur * ep->sigma * ep->sigma / norm,
      ep->pfactor,
      kernel_float_view(pattern2d),
      kernel_float_view(pattern3d),
      ep->beta,
      ep->degrid,
      kernel_complex_blocks(gridsample, 1),
      ep->sharpen,
      sigmaSquaredSharpenMinNormed,
      sigmaSquaredSharpenMaxNormed,
      kernel_float_view(wsharpen),
      ep->dehalo,
      kernel_float_view(wdehalo),
      ht2n,
      kernel_complex_blocks(covar, howmanyblocks),
      kernel_complex_blocks(covarProcess, howmanyblocks),
      sigmaSquaredNoiseNormed2D,
      ep->kratio * ep->kratio
    };
  }

  void ensure_3d_pattern_initialized(int btcur) {
    if (!pattern3d_initialized) {
      std::lock_guard<std::mutex> lock(init3d_mutex);
      if (!pattern3d_initialized) {
        neo_fft3d::Pattern2Dto3D(kernel_float_view(pattern2d), (float)btcur, kernel_float_view(pattern3d));
      }
      pattern3d_initialized = true;
    }
  }

  void process_wiener_3d_frame(
    int n,
    int btcur,
    ds::VideoFrameProvider& provider,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  ) {
    ensure_3d_pattern_initialized(btcur);

    int from = -btcur / 2;
    int to = (btcur - 1) / 2;
    /* apply_in[]
    *
    * |   |   0   |   1   |   2   |   3   |   4   |
    * |   | prev2 | prev1 |current| next1 | next2 |
    * +---+-------+-------+-------+-------+-------+
    * | 2 |       |   o   |   o   |       |       |
    * | 3 |       |   o   |   o   |   o   |       |
    * | 4 |   o   |   o   |   o   |   o   |       |
    * | 5 |   o   |   o   |   o   |   o   |   o   |
    *
    */

    std::shared_ptr<AlignedVector<std::complex<float>>> apply_in_lease[5];
    std::complex<float>* apply_in_ptr[5] {nullptr};

    // We can't hold the lock during the entire loop because plan->Execute takes a long time.
    // We need a two-phase "reserve" then "publish" to prevent other threads from reading half-written data.
    for (auto i = from; i <= to; i++)
    {
      bool needs_compute = false;
      {
        std::lock_guard<std::mutex> lock1(cache_mutex);
        if (auto cached_lease = fftcache->get_read(n+i)) {
          apply_in_lease[2+i] = cached_lease;
          apply_in_ptr[2+i] = cached_lease->data();
        }
        else {
          // Get a write slot but do NOT publish the key yet!
          apply_in_lease[2+i] = fftcache->get_write(n+i);
          apply_in_ptr[2+i] = apply_in_lease[2+i]->data();
          needs_compute = true;
        }
      }

      if (needs_compute) {
        auto frame = neo_fft3d::engine::FetchFrame(provider, n + i);
        const auto& frame_plane = frame.plane(plane);
        transform_source_plane_to_spectrum(frame_plane, views, coverbuf, in, apply_in_ptr[2+i], ep->IsChroma);

        // Publish the key now that the data is ready
        {
          std::lock_guard<std::mutex> lock1(cache_mutex);
          fftcache->publish(apply_in_lease[2+i], n+i);
        }
      }
    }

    neo_fft3d::TemporalComplexBlockViews apply_in_views {};
    for (int slot = 0; slot < 5; ++slot) {
      apply_in_views[slot] = kernel_complex_blocks(apply_in_ptr[slot], howmanyblocks);
    }
    backend->Apply3D(apply_in_views, kernel_complex_blocks(outrez, howmanyblocks), sfp);
    backend->Sharpen(kernel_complex_blocks(outrez, howmanyblocks), sfp);
  }

  void process_wiener_frame(
    int n,
    int btcur,
    ds::VideoFrameProvider& provider,
    const ds::PlaneView& src_plane,
    const ds::MutablePlaneView& dst_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  ) {
    if (btcur == 1) // 2D
    {
      // cur frame
      transform_source_plane_to_spectrum(src_plane, views, coverbuf, in, outrez, ep->IsChroma);
      backend->Apply2D(kernel_complex_blocks(outrez, howmanyblocks), sfp);
      if (ep->pfactor != 0) {
        backend->Sharpen(kernel_complex_blocks(outrez, howmanyblocks), sfp);
      }
    }
    else // 3D
    {
      process_wiener_3d_frame(n, btcur, provider, views, coverbuf, in, outrez, sfp);
    }

    // do inverse FFT, get filtered 'in' array
    // make destination frame plane from current overlaped blocks
    write_spectrum_to_destination(outrez, in, coverbuf, views, dst_plane, ep->IsChroma);
  }

  void process_kalman_frame(
    int n,
    const ds::PlaneView& src_plane,
    const ds::MutablePlaneView& dst_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  ) {
    if (n == 0) {
      views.CopyPlane(src_plane, dst_plane);
      return; // first frame  not processed
    }
    /* PF 170302 comment: accumulated error?
      orig = BlankClip(...)
      new = orig.FFT3DFilter(sigma = 3, plane = 4, bt = 0, degrid = 0)
      Subtract(orig,new).Levels(120, 1, 255 - 120, 0, 255, coring = false)
    */

    // put source bytes to float array of overlapped blocks
    // cur frame
    transform_source_plane_to_spectrum(src_plane, views, coverbuf, in, outrez, ep->IsChroma);
    backend->Kalman(kernel_complex_blocks(outrez, howmanyblocks), kernel_complex_blocks(outLast, howmanyblocks), sfp);

    // copy outLast to outrez
    std::memcpy(outrez, outLast.data(), outsize * sizeof(std::complex<float>));  //v.0.9.2
    backend->Sharpen(kernel_complex_blocks(outrez, howmanyblocks), sfp);
    // do inverse FFT 2D, get filtered 'in' array
    // note: input "out" array is destroyed by execute algo.
    // that is why we must have its copy in "outLast" array
    // make destination frame plane from current overlaped blocks
    write_spectrum_to_destination(outrez, in, coverbuf, views, dst_plane, ep->IsChroma);
  }

  void process_sharpen_frame(
    const ds::PlaneView& src_plane,
    const ds::MutablePlaneView& dst_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  ) {
    transform_source_plane_to_spectrum(src_plane, views, coverbuf, in, outrez, ep->IsChroma);
    backend->Sharpen(kernel_complex_blocks(outrez, howmanyblocks), sfp);
    // do inverse FFT 2D, get filtered 'in' array
    // make destination frame plane from current overlaped blocks
    write_spectrum_to_destination(outrez, in, coverbuf, views, dst_plane, ep->IsChroma);
  }

public:
  FFT3DEngine(EngineParams _ep, int _plane, std::unique_ptr<neo_fft3d::engine::FilterBackend> _backend) :
  ep(std::make_unique<EngineParams>(_ep)), iop(std::make_unique<IOParams>()), plane(_plane), backend(std::move(_backend)) {
    validate_backend();
    normalize_params_for_sample_format();
    validate_overlap_params_and_apply_defaults();
    validate_temporal_params();
    configure_geometry();
    configure_storage();

    auto *in = workspace.initial_overlap();
    auto *outrez = workspace.initial_spectrum();
    const neo_fft3d::fft::PlanOptions plan_options {ep->measure};
    const neo_fft3d::fft::PlanBuffers plan_buffers {
      in,
      outrez
    };

    create_main_plans(plan_options, plan_buffers);
    resize_window_buffers();
    initialize_filter_windows();
    initialize_normalization_params();
    initialize_kalman_buffers();
    initialize_pattern_buffers();
    backend->Configure(*ep);
    initialize_grid_sample(in, plan_options, plan_buffers);
  }
  ~FFT3DEngine() {
    {
      neo_fft3d::fft::GlobalLockGuard fftw_lock;
      plan.reset();
      plan1.reset();
      planinv.reset();
    }
  }

  void ProcessFrame(int n, ds::VideoFrameProvider& provider, ds::MutableVideoFrameView dst) {
    auto workspace_lease = workspace.acquire();
    const auto& workspace_slot = workspace_lease.get();
    const auto views = frame_views();
    byte* coverbuf = workspace_slot.cover;
    float* in = workspace_slot.overlap;
    std::complex<float>* outrez = workspace_slot.spectrum;

    ensure_fft_cache_capacity(workspace_slot.slot_count);

    if (handle_pattern_frame(n, provider, dst, views, coverbuf, in, outrez)) {
      return;
    }

    // Request frame 'n' from the child (source) clip.
    auto src = neo_fft3d::engine::FetchFrame(provider, n);
    const auto& src_plane = src.plane(plane);
    auto& dst_plane = dst.plane(plane);
    ep->framepitch = static_cast<int>(src_plane.stride_bytes) / ep->vi.Format.BytesPerSample;
    ep->framepitch_dst = static_cast<int>(dst_plane.stride_bytes) / ep->vi.Format.BytesPerSample;

    int btcur = current_bt_for_frame(n);
    SharedFunctionParams sfp = make_shared_function_params(btcur);

    if (btcur > 0) // Wiener
    {
      process_wiener_frame(n, btcur, provider, src_plane, dst_plane, views, coverbuf, in, outrez, sfp);
    }
    else if (ep->bt == 0) //Kalman filter
    {
      process_kalman_frame(n, src_plane, dst_plane, views, coverbuf, in, outrez, sfp);
    }
    else if (ep->bt == -1) /// sharpen only
    {
      process_sharpen_frame(src_plane, dst_plane, views, coverbuf, in, outrez, sfp);
    }

    // As we now are finished processing the image, the destination image has been written in place.
  }

  bool CacheRefresh(int n) {
    if (!fftcache) {
      return false;
    }
    std::lock_guard<std::mutex> lock(cache_mutex);
    return fftcache->refresh(n);
  }
};
