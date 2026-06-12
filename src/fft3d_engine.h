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
#include "fft/fft_backend.hpp"
#include "functions.h"
#include "helper.h"
#include "buffer.h"
#include "cache.hpp"
#include <atomic>
#include <thread>
#include <unordered_map>

class FFT3DEngine {
public:
  std::unique_ptr<EngineParams> ep;
  std::unique_ptr<IOParams> iop;
  int plane; // color plane
  FetchFrameFunctor* fetch_frame;
  std::shared_ptr<neo_fft3d::fft::FFTBackend> fft_backend;

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

  std::vector<AlignedVector<float>> mt_in;
  std::vector<AlignedVector<std::complex<float>>> mt_out;
  std::vector<AlignedVector<byte>> mt_coverbuf;

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

  int CPUFlags;

  std::mutex cache_mutex;
  std::mutex thread_check_mutex;

  std::vector<int> thread_id_store;

  struct FilterFunctionPointers ffp;
  bool pattern3d_initialized {false};
  std::mutex init3d_mutex;

public:
  FFT3DEngine(EngineParams _ep, int _plane, FetchFrameFunctor* _fetch_frame, std::shared_ptr<neo_fft3d::fft::FFTBackend> _fft_backend) :
  ep(std::make_unique<EngineParams>(_ep)), iop(std::make_unique<IOParams>()), plane(_plane), fetch_frame(_fetch_frame), fft_backend(_fft_backend) {
    int i, j;

    float factor;
    switch(ep->vi.Format.BytesPerSample) {
      case 1: factor = 1.0f; break;
      case 2: factor = float(1 << (ep->vi.Format.BitsPerSample-8)); break;
      default: factor = 1 / 255.0f;
    }

    ep->sigma *= factor;
    ep->sigma2 *= factor;
    ep->sigma3 *= factor;
    ep->sigma4 *= factor;
    ep->smin *= factor;
    ep->smax *= factor;

    if (ep->ow * 2 > ep->bw) throw("ow must be less than bw / 2");
    if (ep->oh * 2 > ep->bh) throw("oh must be less than bh / 2");
    if (ep->ow < 0) ep->ow = ep->bw / 3; // changed from ep->bw/4 to ep->bw/3 in v.1.2
    if (ep->oh < 0) ep->oh = ep->bh / 3; // changed from bh/4 to bh/3 in v.1.2

    if (ep->bt < -1 || ep->bt > 5) throw("bt must be -1 (Sharpen), 0 (Kalman), 1..5 (Wiener)");

    int xRatioShift = ep->IsChroma ? ep->vi.Format.SSW : 0;
    int yRatioShift = ep->IsChroma ? ep->vi.Format.SSH : 0;

    iop->nox = (((ep->vi.Width - ep->l - ep->r) >> xRatioShift) - ep->ow + (ep->bw - ep->ow - 1)) / (ep->bw - ep->ow);
    iop->noy = (((ep->vi.Height - ep->t - ep->b) >> yRatioShift) - ep->oh + (ep->bh - ep->oh - 1)) / (ep->bh - ep->oh);


    // padding by 1 block per side
    iop->nox += 2;
    iop->noy += 2;
    mirw = ep->bw - ep->ow; // set mirror size as block interval
    mirh = ep->bh - ep->oh;

    if (ep->beta < 1)
      throw("beta must be not less 1.0");

    coverwidth = iop->nox*(ep->bw - ep->ow) + ep->ow;
    coverheight = iop->noy*(ep->bh - ep->oh) + ep->oh;
    coverpitch = ((coverwidth + 15) / 16) * 16; // align to 16 elements. Pitch is element-granularity. For byte pitch, multiply is by pixelsize

    insize = ep->bw*ep->bh*iop->nox*iop->noy;
    insize = ((insize + 15) / 16) * 16;
    outwidth = ep->bw / 2 + 1; // width (pitch) of complex fft block
    outpitch = ((outwidth + 15) / 16) * 16; // must be even for SSE - v1.7
    outsize = outpitch*ep->bh*iop->nox*iop->noy; // replace outwidth to outpitch here and below in v1.7
    if (ep->bt == 0) // Kalman
    {
      outLast.resize(outsize);
      covar.resize(outsize);
      covarProcess.resize(outsize);
    }
    // in and out are thread dependent now // neo-r1
    mt_in.emplace_back(insize);
    auto in = mt_in[0].data();
    mt_out.emplace_back(outsize); //v1.8
    auto outrez = as_fftw(mt_out[0].data());
    gridsample.resize(outsize); //v1.8

    if (ep->bt > 1)
      fftcache = std::make_unique<cache<std::complex<float>>>(ep->bt + 3, outsize);

    howmanyblocks = iop->nox*iop->noy;
    const neo_fft3d::fft::PlanOptions plan_options {ep->measure};
    const neo_fft3d::fft::PlanBuffers plan_buffers {
      in,
      reinterpret_cast<neo_fft3d::fft::Complex*>(outrez)
    };

    //	*inembed = NULL;
    //	*onembed = NULL;

    {
      GlobalLockGuard fftw_lock(ep->avs_env, "fftw", ep->has_at_least_v12);

      plan = fft_backend->CreatePlan(
        ep->bh,
        ep->bw,
        outpitch,
        neo_fft3d::fft::Direction::r2c,
        howmanyblocks,
        plan_options,
        plan_buffers
      );

      planinv = fft_backend->CreatePlan(
        ep->bh,
        ep->bw,
        outpitch,
        neo_fft3d::fft::Direction::c2r,
        howmanyblocks,
        plan_options,
        plan_buffers
      );
    }

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

    // define analysis and synthesis windows
    // combining window (analize mult by synthesis) is raised cosine (Hanning)

    float pi = 3.1415926535897932384626433832795f;
    if (ep->wintype == 0) // window type
    { // , used in all version up to 1.3
      // half-cosine, the same for analysis and synthesis
      // define analysis windows
      for (i = 0; i < ep->ow; i++)
      {
        iop->wanxl[i] = cosf(pi*(i - ep->ow + 0.5f) / (ep->ow * 2)); // left analize window (half-cosine)
        iop->wanxr[i] = cosf(pi*(i + 0.5f) / (ep->ow * 2)); // right analize window (half-cosine)
      }
      for (i = 0; i < ep->oh; i++)
      {
        iop->wanyl[i] = cosf(pi*(i - ep->oh + 0.5f) / (ep->oh * 2));
        iop->wanyr[i] = cosf(pi*(i + 0.5f) / (ep->oh * 2));
      }
      // use the same windows for synthesis too.
      for (i = 0; i < ep->ow; i++)
      {
        iop->wsynxl[i] = iop->wanxl[i]; // left  window (half-cosine)

        iop->wsynxr[i] = iop->wanxr[i]; // right  window (half-cosine)
      }
      for (i = 0; i < ep->oh; i++)
      {
        iop->wsynyl[i] = iop->wanyl[i];
        iop->wsynyr[i] = iop->wanyr[i];
      }
    }
    else if (ep->wintype == 1) // added in v.1.4
    {
      // define analysis windows as more flat (to decrease grid)
      for (i = 0; i < ep->ow; i++)
      {
        iop->wanxl[i] = sqrt(cosf(pi*(i - ep->ow + 0.5f) / (ep->ow * 2)));
        iop->wanxr[i] = sqrt(cosf(pi*(i + 0.5f) / (ep->oh * 2)));
      }
      for (i = 0; i < ep->oh; i++)
      {
        iop->wanyl[i] = sqrt(cosf(pi*(i - ep->oh + 0.5f) / (ep->oh * 2)));
        iop->wanyr[i] = sqrt(cosf(pi*(i + 0.5f) / (ep->oh * 2)));
      }
      // define synthesis as supplenent to rised cosine (Hanning)
      for (i = 0; i < ep->ow; i++)
      {
        iop->wsynxl[i] = iop->wanxl[i] * iop->wanxl[i] * iop->wanxl[i]; // left window
        iop->wsynxr[i] = iop->wanxr[i] * iop->wanxr[i] * iop->wanxr[i]; // right window
      }
      for (i = 0; i < ep->oh; i++)
      {
        iop->wsynyl[i] = iop->wanyl[i] * iop->wanyl[i] * iop->wanyl[i];
        iop->wsynyr[i] = iop->wanyr[i] * iop->wanyr[i] * iop->wanyr[i];
      }
    }
    else //  (ep->wintype==2) - added in v.1.4
    {
      // define analysis windows as flat (to prevent grid)
      std::fill_n(iop->wanxl.data(), ep->ow, 1.0f);
      std::fill_n(iop->wanxr.data(), ep->ow, 1.0f);
      std::fill_n(iop->wanyl.data(), ep->oh, 1.0f);
      std::fill_n(iop->wanyr.data(), ep->oh, 1.0f);

      // define synthesis as rised cosine (Hanning)
      for (i = 0; i < ep->ow; i++)
      {
        iop->wsynxl[i] = cosf(pi*(i - ep->ow + 0.5f) / (ep->ow * 2));
        iop->wsynxl[i] = iop->wsynxl[i] * iop->wsynxl[i];// left window (rised cosine)
        iop->wsynxr[i] = cosf(pi*(i + 0.5f) / (ep->ow * 2));
        iop->wsynxr[i] = iop->wsynxr[i] * iop->wsynxr[i]; // right window (falled cosine)
      }
      for (i = 0; i < ep->oh; i++)
      {
        iop->wsynyl[i] = cosf(pi*(i - ep->oh + 0.5f) / (ep->oh * 2));
        iop->wsynyl[i] = iop->wsynyl[i] * iop->wsynyl[i];
        iop->wsynyr[i] = cosf(pi*(i + 0.5f) / (ep->oh * 2));
        iop->wsynyr[i] = iop->wsynyr[i] * iop->wsynyr[i];
      }
    }

    // window for sharpen
    auto tmp_wsharpen = wsharpen.data();
    for (j = 0; j < ep->bh; j++)
    {
      int dj = j;
      if (j >= ep->bh / 2)
        dj = ep->bh - j;
      float d2v = float(dj*dj)*(ep->svr*ep->svr) / ((ep->bh / 2)*(ep->bh / 2)); // v1.7
      for (i = 0; i < outwidth; i++)
      {
        float d2 = d2v + float(i*i) / ((ep->bw / 2)*(ep->bw / 2)); // distance_2 - v1.7
        tmp_wsharpen[i] = 1 - exp(-d2 / (2 * ep->scutoff*ep->scutoff));
      }
      tmp_wsharpen += outpitch;
    }

    // window for dehalo - added in v1.9
    auto tmp_wdehalo = wdehalo.data();
    float wmax = 0;
    for (j = 0; j < ep->bh; j++)
    {
      int dj = j;
      if (j >= ep->bh / 2)
        dj = ep->bh - j;
      float d2v = float(dj*dj)*(ep->svr*ep->svr) / ((ep->bh / 2)*(ep->bh / 2));
      for (i = 0; i < outwidth; i++)
      {
        float d2 = d2v + float(i*i) / ((ep->bw / 2)*(ep->bw / 2)); // squared distance in frequency domain
        tmp_wdehalo[i] = exp(-0.7f*d2*ep->hr*ep->hr) - exp(-d2*ep->hr*ep->hr); // some window with max around 1/hr, small at low and high frequencies
        if (tmp_wdehalo[i] > wmax) wmax = tmp_wdehalo[i]; // for normalization
      }
      tmp_wdehalo += outpitch;
    }

    tmp_wdehalo = wdehalo.data();
    for (j = 0; j < ep->bh; j++)
    {
      for (i = 0; i < outwidth; i++)
      {
        tmp_wdehalo[i] /= wmax; // normalize
      }
      tmp_wdehalo += outpitch;
    }

    norm = 1.0f / (ep->bw*ep->bh); // do not forget set FFT normalization factor

    sigmaSquaredNoiseNormed2D = ep->sigma*ep->sigma / norm;
    sigmaNoiseNormed2D = ep->sigma / sqrtf(norm);
    sigmaMotionNormed = ep->sigma*ep->kratio / sqrtf(norm);
    sigmaSquaredSharpenMinNormed = ep->smin*ep->smin / norm;
    sigmaSquaredSharpenMaxNormed = ep->smax*ep->smax / norm;
    ht2n = ep->ht*ep->ht / norm; // halo threshold squared and normed - v1.9

    // init Kalman
    if (ep->bt == 0) // Kalman
    {
      std::fill_n(reinterpret_cast<float*>(outLast.data()), outsize * 2, 0.0f);
      std::fill_n(reinterpret_cast<float*>(covar.data()), outsize * 2, sigmaSquaredNoiseNormed2D);
      std::fill_n(reinterpret_cast<float*>(covarProcess.data()), outsize * 2, sigmaSquaredNoiseNormed2D);
    }

    pwin.resize(ep->bh*outpitch); // pattern window array

    float fw2, fh2;
    auto tmp_pwin = pwin.data();
    for (j = 0; j < ep->bh; j++)
    {
      if (j < ep->bh / 2)
        fh2 = (j*2.0f / ep->bh)*(j*2.0f / ep->bh);
      else
        fh2 = ((ep->bh - 1 - j)*2.0f / ep->bh)*((ep->bh - 1 - j)*2.0f / ep->bh);
      for (i = 0; i < outwidth; i++)
      {
        fw2 = (i*2.0f / ep->bw)*(j*2.0f / ep->bw);
        tmp_pwin[i] = (fh2 + fw2) / (fh2 + fw2 + ep->pcutoff*ep->pcutoff);
      }
      tmp_pwin += outpitch;
    }

    pattern2d.resize(ep->bh*outpitch); // noise pattern window array
    pattern3d.resize(ep->bh*outpitch); // noise pattern window array

    if ((ep->sigma2 != ep->sigma || ep->sigma3 != ep->sigma || ep->sigma4 != ep->sigma) && ep->pfactor == 0)
    {// we have different sigmas, so create pattern from sigmas
      SigmasToPattern(ep->sigma, ep->sigma2, ep->sigma3, ep->sigma4, ep->bh, outwidth, outpitch, norm, pattern2d.data());
      isPatternSet = true;
      ep->pfactor = 1;
    }
    else
    {
      isPatternSet = false; // pattern must be estimated
    }

    CPUFlags = GetCPUFlags();
    ffp.set_ffp(CPUFlags, ep->degrid, ep->pfactor, ep->bt, ep->opt);

    // prepare  window compensation array gridsample
    // allocate large array for simplicity :)
    // but use one block only for speed
    // Attention: other block could be the same, but we do not calculate them!
    {
      GlobalLockGuard fftw_lock(ep->avs_env, "fftw", ep->has_at_least_v12);

      plan1 = fft_backend->CreatePlan(
        ep->bh,
        ep->bw,
        outpitch,
        neo_fft3d::fft::Direction::r2c,
        1,
        plan_options,
        plan_buffers
      ); // 1 block
    }

    std::vector<byte> coverbuf(coverheight*coverpitch*ep->vi.Format.BytesPerSample);
    // avs+
    switch (ep->vi.Format.BytesPerSample) {
    case 1: std::memset(coverbuf.data(), 255, coverbuf.size());
      break;
    case 2: std::fill_n(reinterpret_cast<uint16_t*>(coverbuf.data()), coverheight*coverpitch, (1 << ep->vi.Format.BitsPerSample) - 1);
      break; // 255
    case 4: std::fill_n(reinterpret_cast<float*>(coverbuf.data()), coverheight*coverpitch, 1.0f);
      break; // 255
    }
    CoverToOverlap(ep.get(), iop.get(), in, coverbuf.data(), coverwidth, coverpitch, false);
    // make FFT 2D
    plan1->Execute(in, gridsample.data(), 1);

  }
  ~FFT3DEngine() {
    {
      GlobalLockGuard fftw_lock(ep->avs_env, "fftw", ep->has_at_least_v12);
      plan.reset();
      plan1.reset();
      planinv.reset();
    }
  }

  DSFrame GetFrame(int n, std::unordered_map<int, DSFrame> in_frames) {
    unsigned int thread_id;
    DSFrame src, psrc, dst;
    int pxf, pyf;

    byte* coverbuf {nullptr};
    float *in {nullptr};
    fftwf_complex *outrez {nullptr};
    std::size_t required_cache_size {0};
    {
      std::lock_guard<std::mutex> lock(thread_check_mutex);
      // Find empty slot
      auto it = std::find(thread_id_store.begin(), thread_id_store.end(), 0);
      thread_id = static_cast<int>(std::distance(thread_id_store.begin(), it));
      if (it == thread_id_store.end()) {
        thread_id_store.push_back(1);

        while (mt_coverbuf.size() <= thread_id)
          mt_coverbuf.emplace_back(coverheight * coverpitch * ep->vi.Format.BytesPerSample);
        while (mt_in.size() <= thread_id)
          mt_in.emplace_back(insize);
        while (mt_out.size() <= thread_id)
          mt_out.emplace_back(outsize);
      }
      else
        thread_id_store[thread_id] = 1;

      if (fftcache) {
        required_cache_size = static_cast<std::size_t>(ep->bt) + thread_id_store.size() + 2;
      }
      coverbuf = mt_coverbuf[thread_id].data();
      in = mt_in[thread_id].data();
      outrez = as_fftw(mt_out[thread_id].data());
    }

    if (fftcache) {
      std::lock_guard<std::mutex> lock_cache(cache_mutex);
      fftcache->resize(required_cache_size);
    }

    if (ep->pfactor != 0) {
      GlobalLockGuard fftw_lock(ep->avs_env, "fftw", ep->has_at_least_v12);
      if (ep->pfactor != 0 && isPatternSet == false && ep->pshow == false) // get noise pattern
      {
        if (in_frames.find(ep->pframe) == in_frames.end())
          psrc = (*fetch_frame)(ep->pframe);
        else
          psrc = in_frames[ep->pframe];
        auto sptr = psrc.SrcPointers[plane];
        ep->framepitch = psrc.StrideBytes[plane] / ep->vi.Format.BytesPerSample;

        // put source bytes to float array of overlapped blocks
        FrameToCover(ep.get(), plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
        plan->Execute(in, reinterpret_cast<std::complex<float>*>(outrez), howmanyblocks);
        if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
          FindPatternBlock(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, ep->px, ep->py, pwin.data(), ep->degrid, as_fftw(gridsample.data()));
        SetPattern(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, ep->px, ep->py, pwin.data(), pattern2d.data(), psigma, ep->degrid, as_fftw(gridsample.data()));
        isPatternSet = true;
      }
      else if (ep->pfactor != 0 && ep->pshow == true)
      {
        // show noise pattern window
        if (in_frames.find(n) == in_frames.end())
          src = (*fetch_frame)(n);
        else
          src = in_frames[n];
        dst = src.Create(true);
        auto sptr = src.SrcPointers[plane];
        auto dptr = dst.DstPointers[plane];
        ep->framepitch = src.StrideBytes[plane] / ep->vi.Format.BytesPerSample;
        ep->framepitch_dst = dst.StrideBytes[plane] / ep->vi.Format.BytesPerSample;

        // put source bytes to float array of overlapped blocks
        FrameToCover(ep.get(), plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
        // make FFT 2D
        plan->Execute(in, reinterpret_cast<std::complex<float>*>(outrez), howmanyblocks);
        if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
          FindPatternBlock(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf, pwin.data(), ep->degrid, as_fftw(gridsample.data()));
        else
        {
          pxf = ep->px; // fixed bug in v1.6
          pyf = ep->py;
        }
        SetPattern(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf, pwin.data(), pattern2d.data(), psigma, ep->degrid, as_fftw(gridsample.data()));

        // change analysis and synthesis window to constant to show
        std::fill_n(iop->wanxl.data(), ep->ow, 1.0f);
        std::fill_n(iop->wanxr.data(), ep->ow, 1.0f);
        std::fill_n(iop->wsynxl.data(), ep->ow, 1.0f);
        std::fill_n(iop->wsynxr.data(), ep->ow, 1.0f);
        std::fill_n(iop->wanyl.data(), ep->oh, 1.0f);
        std::fill_n(iop->wanyr.data(), ep->oh, 1.0f);
        std::fill_n(iop->wsynyl.data(), ep->oh, 1.0f);
        std::fill_n(iop->wsynyr.data(), ep->oh, 1.0f);

        // put source bytes to float array of overlapped blocks
        // cur frame
        FrameToCover(ep.get(), plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, !ep->vi.Format.IsFamilyRGB);
        // make FFT 2D
        plan->Execute(in, reinterpret_cast<std::complex<float>*>(outrez), howmanyblocks);

        PutPatternOnly(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf);
        // do inverse 2D FFT, get filtered 'in' array
        planinv->Execute(reinterpret_cast<std::complex<float>*>(outrez), in, howmanyblocks);

        // make destination frame plane from current overlaped blocks
        OverlapToCover(ep.get(), iop.get(), in, norm, coverbuf, coverwidth, coverpitch, !ep->vi.Format.IsFamilyRGB);
        CoverToFrame(ep.get(), plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);
        int psigmaint = ((int)(10 * psigma)) / 10;
        int psigmadec = (int)((psigma - psigmaint) * 10);
        wsprintf(messagebuf, " frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);
        // TODO: DrawString(dst, vi, 0, 0, messagebuf);
        {
          std::lock_guard<std::mutex> lock(thread_check_mutex);
          thread_id_store[thread_id] = 0;
        }
        return dst; // return pattern frame to show
      }
    }

    // Request frame 'n' from the child (source) clip.
    if (in_frames.find(n) == in_frames.end())
      src = (*fetch_frame)(n);
    else
      src = in_frames[n];
    dst = src.Create(false);
    auto sptr = src.SrcPointers[plane];
    auto dptr = dst.DstPointers[plane];
    ep->framepitch = src.StrideBytes[plane] / ep->vi.Format.BytesPerSample;
    ep->framepitch_dst = dst.StrideBytes[plane] / ep->vi.Format.BytesPerSample;

    int btcur = ep->bt; // bt used for current frame
  //	if ( (bt/2 > n) || bt==3 && n==vi.num_frames-1 )
    if ((ep->bt / 2 > n) || (ep->bt - 1) / 2 > (ep->vi.Frames - 1 - n))
      btcur = 1; //	do 2D filter for first and last frames

    SharedFunctionParams sfp {
      outwidth,
      outpitch,
      ep->bh,
      howmanyblocks,
      btcur * ep->sigma * ep->sigma / norm,
      ep->pfactor,
      pattern2d.data(),
      pattern3d.data(),
      ep->beta,
      ep->degrid,
      as_fftw(gridsample.data()),
      ep->sharpen,
      sigmaSquaredSharpenMinNormed,
      sigmaSquaredSharpenMaxNormed,
      wsharpen.data(),
      ep->dehalo,
      wdehalo.data(),
      ht2n,
      as_fftw(covar.data()),
      as_fftw(covarProcess.data()),
      sigmaSquaredNoiseNormed2D,
      ep->kratio * ep->kratio
    };

    std::shared_ptr<AlignedVector<std::complex<float>>> apply_in_lease[5];
    fftwf_complex* apply_in_ptr[5] {nullptr};

    if (btcur > 0) // Wiener
    {
      if (btcur == 1) // 2D
      {
        // cur frame
        FrameToCover(ep.get(), plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
        plan->Execute(in, reinterpret_cast<std::complex<float>*>(outrez), howmanyblocks);
        ffp.Apply2D(outrez, sfp);
        if (ep->pfactor != 0)
          ffp.Sharpen(outrez, sfp);
      }
      else // 3D
      {
        if (!pattern3d_initialized) {
          std::lock_guard<std::mutex> lock(init3d_mutex);
          if (!pattern3d_initialized)
            Pattern2Dto3D(pattern2d.data(), ep->bh, outwidth, outpitch, (float)btcur, pattern3d.data());
          pattern3d_initialized = true;
        }

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

        {
          // We can't hold the lock during the entire loop because plan->Execute takes a long time.
          // We need a two-phase "reserve" then "publish" to prevent other threads from reading half-written data.
          for (auto i = from; i <= to; i++)
          {
            bool needs_compute = false;
            {
              std::lock_guard<std::mutex> lock1(cache_mutex);
              if (auto cached_lease = fftcache->get_read(n+i)) {
                apply_in_lease[2+i] = cached_lease;
                apply_in_ptr[2+i] = as_fftw(cached_lease);
              }
              else {
                // Get a write slot but do NOT publish the key yet!
                apply_in_lease[2+i] = fftcache->get_write(n+i);
                apply_in_ptr[2+i] = as_fftw(apply_in_lease[2+i]);
                needs_compute = true;
              }
            }

            if (needs_compute) {
              DSFrame frame;
              DSFrame* pframe;
              if (in_frames.find(n+i) == in_frames.end())
                pframe = &(frame = (*fetch_frame)(n+i));
              else
                pframe = &in_frames[n+i];
              FrameToCover(ep.get(), plane, pframe->SrcPointers[plane], coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
              CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, ep->IsChroma);

              // Execute FFT OUTSIDE the cache lock
              plan->Execute(in, as_complex(apply_in_ptr[2+i]), howmanyblocks);

              // Publish the key now that the data is ready
              {
                std::lock_guard<std::mutex> lock1(cache_mutex);
                fftcache->publish(apply_in_lease[2+i], n+i);
              }
            }
          }
        }

        ffp.Apply3D(apply_in_ptr, outrez, sfp);
        ffp.Sharpen(outrez, sfp);
      }

      // do inverse FFT, get filtered 'in' array
      planinv->Execute(reinterpret_cast<std::complex<float>*>(outrez), in, howmanyblocks);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep.get(), iop.get(), in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep.get(), plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);
    }
    else if (ep->bt == 0) //Kalman filter
    {
      if (n == 0) {
       {
         std::lock_guard<std::mutex> lock(thread_check_mutex);
         thread_id_store[thread_id] = 0;
       }
       return src; // first frame  not processed
      }
      /* PF 170302 comment: accumulated error?
        orig = BlankClip(...)
        new = orig.FFT3DFilter(sigma = 3, plane = 4, bt = 0, degrid = 0)
        Subtract(orig,new).Levels(120, 1, 255 - 120, 0, 255, coring = false)
      */

      // put source bytes to float array of overlapped blocks
      // cur frame
      FrameToCover(ep.get(), plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
      CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      plan->Execute(in, reinterpret_cast<std::complex<float>*>(outrez), howmanyblocks);
      ffp.Kalman(outrez, as_fftw(outLast.data()), sfp);

      // copy outLast to outrez
      memcpy(outrez, outLast.data(), outsize * sizeof(fftwf_complex));  //v.0.9.2
      ffp.Sharpen(outrez, sfp);
      // do inverse FFT 2D, get filtered 'in' array
      // note: input "out" array is destroyed by execute algo.
      // that is why we must have its copy in "outLast" array
      planinv->Execute(reinterpret_cast<std::complex<float>*>(outrez), in, howmanyblocks);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep.get(), iop.get(), in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep.get(), plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);

    }
    else if (ep->bt == -1) /// sharpen only
    {
      FrameToCover(ep.get(), plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
      CoverToOverlap(ep.get(), iop.get(), in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      plan->Execute(in, reinterpret_cast<std::complex<float>*>(outrez), howmanyblocks);
      ffp.Sharpen(outrez, sfp);
      // do inverse FFT 2D, get filtered 'in' array
      planinv->Execute(reinterpret_cast<std::complex<float>*>(outrez), in, howmanyblocks);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep.get(), iop.get(), in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep.get(), plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);

    }

    {
      std::lock_guard<std::mutex> lock(thread_check_mutex);
      thread_id_store[thread_id] = 0;
    }

    // As we now are finished processing the image, we return the destination image.
    return dst;
  }

  bool CacheRefresh(int n) {
    if (!fftcache) return false;
    std::lock_guard<std::mutex> lock(cache_mutex);
    return fftcache->refresh(n);
  }
};
