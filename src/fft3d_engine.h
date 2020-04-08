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
#include "functions.h"
#include "helper.h"
#include "buffer.h"
#include "cache.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

class FFT3DEngine {
  EngineParams* ep;
  IOParams* iop;
  int plane; // color plane
  FetchFrameFunctor* fetch_frame;

  // additional parameterss
  fftwf_complex *gridsample; //v1.8
  fftwf_plan plan, planinv, plan1;
  int outwidth;
  int outpitch; //v.1.7
  int insize;
  int outsize;
  int howmanyblocks;

  int ndim[2];
  int inembed[2];
  int onembed[2];

  float *wsharpen;
  float *wdehalo;

  fftwf_complex *outLast, *covar, *covarProcess;
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

  float *pwin;
  float *pattern2d;
  float *pattern3d;
  bool isPatternSet;
  float psigma {0};
  char *messagebuf;

  // added in v.0.9 for delayed FFTW3.DLL loading
  struct FFTFunctionPointers fftfp;

  int CPUFlags;

  cache<fftwf_complex> *fftcache;

  std::mutex fft_mutex;
  std::mutex cache_mutex;
  std::mutex thread_check_mutex;

  std::vector<bool> thread_id_store;
  std::vector<float *> mt_in;
  std::vector<fftwf_complex *> mt_out;
  std::vector<byte *> mt_coverbuf;

  struct FilterFunctionPointers ffp;

public:
  FFT3DEngine(EngineParams _ep, int _plane, FetchFrameFunctor* _fetch_frame) :
  ep(new EngineParams(_ep)), iop(new IOParams()), plane(_plane), fetch_frame(_fetch_frame) {
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

    int istat;

    fftfp.load();

    istat = fftfp.fftwf_init_threads();

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
      outLast = (fftwf_complex *)_aligned_malloc(sizeof(fftwf_complex) * outsize, FRAME_ALIGN);
      covar = (fftwf_complex *)_aligned_malloc(sizeof(fftwf_complex) * outsize, FRAME_ALIGN);
      covarProcess = (fftwf_complex *)_aligned_malloc(sizeof(fftwf_complex) * outsize, FRAME_ALIGN);
    }
    // in and out are thread dependent now // neo-r1
    mt_in.push_back((float *)_aligned_malloc(sizeof(float) * insize, FRAME_ALIGN));
    auto in = mt_in[0];
    mt_out.push_back((fftwf_complex *)_aligned_malloc(sizeof(fftwf_complex) * outsize, FRAME_ALIGN)); //v1.8
    auto outrez = mt_out[0];
    gridsample = (fftwf_complex *)_aligned_malloc(sizeof(fftwf_complex) * outsize, FRAME_ALIGN); //v1.8

    fftcache = NULL;
    if (ep->bt > 1)
      fftcache = new cache<fftwf_complex>(ep->bt + 3, outsize);

    int planFlags;
    // use FFTW_ESTIMATE or FFTW_MEASURE (more optimal plan, but with time calculation at load stage)
    if (ep->measure)
      planFlags = FFTW_MEASURE;
    else
      planFlags = FFTW_ESTIMATE;

    int rank = 2; // 2d
    ndim[0] = ep->bh; // size of block along height
    ndim[1] = ep->bw; // size of block along width
    int istride = 1;
    int ostride = 1;
    int idist = ep->bw*ep->bh;
    int odist = outpitch*ep->bh;//  v1.7 (was outwidth)
    inembed[0] = ep->bh;
    inembed[1] = ep->bw;
    onembed[0] = ep->bh;
    onembed[1] = outpitch;//  v1.7 (was outwidth)
    howmanyblocks = iop->nox*iop->noy;

    //	*inembed = NULL;
    //	*onembed = NULL;

    fftfp.fftwf_plan_with_nthreads(1);

    plan = fftfp.fftwf_plan_many_dft_r2c(rank, ndim, howmanyblocks,
      in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags);
    if (plan == NULL)
      throw("FFTW plan error");

    planinv = fftfp.fftwf_plan_many_dft_c2r(rank, ndim, howmanyblocks,
      outrez, onembed, ostride, odist, in, inembed, istride, idist, planFlags);
    if (planinv == NULL)
      throw("FFTW plan error");


    iop->wanxl = new float[ep->ow];
    iop->wanxr = new float[ep->ow];
    iop->wanyl = new float[ep->oh];
    iop->wanyr = new float[ep->oh];

    iop->wsynxl = new float[ep->ow];
    iop->wsynxr = new float[ep->ow];
    iop->wsynyl = new float[ep->oh];
    iop->wsynyr = new float[ep->oh];

    wsharpen = (float*)_aligned_malloc(ep->bh*outpitch * sizeof(float), FRAME_ALIGN);
    wdehalo = (float*)_aligned_malloc(ep->bh*outpitch * sizeof(float), FRAME_ALIGN);

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
      std::fill_n(iop->wanxl, ep->ow, 1.0f);
      std::fill_n(iop->wanxr, ep->ow, 1.0f);
      std::fill_n(iop->wanyl, ep->oh, 1.0f);
      std::fill_n(iop->wanyr, ep->oh, 1.0f);
      
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
    auto tmp_wsharpen = wsharpen;
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
    auto tmp_wdehalo = wdehalo;
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

    tmp_wdehalo = wdehalo;
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
      std::fill_n((float*)outLast, outsize * 2, 0.0f);
      std::fill_n((float*)covar, outsize * 2, sigmaSquaredNoiseNormed2D);
      std::fill_n((float*)covarProcess, outsize * 2, sigmaSquaredNoiseNormed2D);
    }

    CPUFlags = GetCPUFlags(); //re-enabled in v.1.9
    ffp.set_ffp(CPUFlags, ep->degrid, ep->pfactor, ep->bt);

    pwin = new float[ep->bh*outpitch]; // pattern window array

    float fw2, fh2;
    for (j = 0; j < ep->bh; j++)
    {
      if (j < ep->bh / 2)
        fh2 = (j*2.0f / ep->bh)*(j*2.0f / ep->bh);
      else
        fh2 = ((ep->bh - 1 - j)*2.0f / ep->bh)*((ep->bh - 1 - j)*2.0f / ep->bh);
      for (i = 0; i < outwidth; i++)
      {
        fw2 = (i*2.0f / ep->bw)*(j*2.0f / ep->bw);
        pwin[i] = (fh2 + fw2) / (fh2 + fw2 + ep->pcutoff*ep->pcutoff);
      }
      pwin += outpitch;
    }
    pwin -= outpitch*ep->bh; // restore pointer

    pattern2d = (float*)_aligned_malloc(ep->bh*outpitch * sizeof(float), FRAME_ALIGN); // noise pattern window array
    pattern3d = (float*)_aligned_malloc(ep->bh*outpitch * sizeof(float), FRAME_ALIGN); // noise pattern window array

    if ((ep->sigma2 != ep->sigma || ep->sigma3 != ep->sigma || ep->sigma4 != ep->sigma) && ep->pfactor == 0)
    {// we have different sigmas, so create pattern from sigmas
      SigmasToPattern(ep->sigma, ep->sigma2, ep->sigma3, ep->sigma4, ep->bh, outwidth, outpitch, norm, pattern2d);
      isPatternSet = true;
      ep->pfactor = 1;
    }
    else
    {
      isPatternSet = false; // pattern must be estimated
    }

    // prepare  window compensation array gridsample
    // allocate large array for simplicity :)
    // but use one block only for speed
    // Attention: other block could be the same, but we do not calculate them!
    plan1 = fftfp.fftwf_plan_many_dft_r2c(rank, ndim, 1,
      in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags); // 1 block

    byte* coverbuf = new byte[coverheight*coverpitch*ep->vi.Format.BytesPerSample];
    // avs+
    switch (ep->vi.Format.BytesPerSample) {
    case 1: memset(coverbuf, 255, coverheight*coverpitch); 
      break;
    case 2: std::fill_n((uint16_t *)coverbuf, coverheight*coverpitch, (1 << ep->vi.Format.BitsPerSample) - 1); 
      break; // 255 
    case 4: std::fill_n((float *)coverbuf, coverheight*coverpitch, 1.0f); 
      break; // 255 
    }
    CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, false);
    delete[] coverbuf;
    // make FFT 2D
    fftfp.fftwf_execute_dft_r2c(plan1, in, gridsample);

    messagebuf = (char *)malloc(80); //1.8.5

  }
  ~FFT3DEngine() {
    // This is where you can deallocate any memory you might have used.
    fftfp.fftwf_destroy_plan(plan);
    fftfp.fftwf_destroy_plan(plan1);
    fftfp.fftwf_destroy_plan(planinv);
    //	fftfp.fftwf_free(out);
    delete[] iop->wanxl;
    delete[] iop->wanxr;
    delete[] iop->wanyl;
    delete[] iop->wanyr;
    delete[] iop->wsynxl;
    delete[] iop->wsynxr;
    delete[] iop->wsynyl;
    delete[] iop->wsynyr;
    _aligned_free(wsharpen);
    _aligned_free(wdehalo);
    delete[] pwin;
    _aligned_free(pattern2d);
    _aligned_free(pattern3d);
    if (ep->bt == 0) // Kalman
    {
      _aligned_free(outLast);
      _aligned_free(covar);
      _aligned_free(covarProcess);
    }
    delete fftcache;
    delete ep;
    delete iop;
    _aligned_free(gridsample); //fixed memory leakage in v1.8.5
    for (auto it : mt_in)
      _aligned_free(it);
    for (auto it : mt_out)
      _aligned_free(it);
    for (auto it : mt_coverbuf)
      _aligned_free(it);

    fftfp.free();
    free(messagebuf); //v1.8.5
  }

  DSFrame GetFrame(int n, std::unordered_map<int, DSFrame> in_frames) {
    unsigned int thread_id;
    DSFrame src, psrc, dst;
    int pxf, pyf;

    byte* coverbuf;
    float *in;
    fftwf_complex *outrez;
    {
      std::lock_guard<std::mutex> lock(thread_check_mutex);
      // Find empty slot
      auto it = std::find(thread_id_store.begin(), thread_id_store.end(), false);
      thread_id = static_cast<int>(std::distance(thread_id_store.begin(), it));
      if (it == thread_id_store.end()) {
        thread_id_store.push_back(false);
        if (fftcache)
          fftcache->resize(ep->bt + thread_id_store.size() + 2);
      }
      thread_id_store[thread_id] = true;

      while (mt_coverbuf.size() <= thread_id)
        mt_coverbuf.push_back((byte*)_aligned_malloc(coverheight*coverpitch*ep->vi.Format.BytesPerSample, FRAME_ALIGN));
      while (mt_in.size() <= thread_id)
        mt_in.push_back((float *)_aligned_malloc(sizeof(float) * insize, FRAME_ALIGN));
      while (mt_out.size() <= thread_id)
        mt_out.push_back((fftwf_complex *)_aligned_malloc(sizeof(fftwf_complex) * outsize, FRAME_ALIGN));
    }
    coverbuf = mt_coverbuf[thread_id];
    in = mt_in[thread_id];
    outrez = mt_out[thread_id];

    if (ep->pfactor != 0) {
      std::lock_guard<std::mutex> guard(fft_mutex);
      if (ep->pfactor != 0 && isPatternSet == false && ep->pshow == false) // get noise pattern
      {
        if (in_frames.find(ep->pframe) == in_frames.end())
          psrc = (*fetch_frame)(ep->pframe);
        else
          psrc = in_frames[ep->pframe];
        auto sptr = psrc.SrcPointers[plane];
        ep->framepitch = psrc.StrideBytes[plane] / ep->vi.Format.BytesPerSample;

        // put source bytes to float array of overlapped blocks
        FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
        fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
        if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
          FindPatternBlock(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, ep->px, ep->py, pwin, ep->degrid, gridsample);
        SetPattern(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, ep->px, ep->py, pwin, pattern2d, psigma, ep->degrid, gridsample);
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

        // put source bytes to float array of overlapped blocks
        FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
        // make FFT 2D
        fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
        if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
          FindPatternBlock(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf, pwin, ep->degrid, gridsample);
        else
        {
          pxf = ep->px; // fixed bug in v1.6
          pyf = ep->py;
        }
        SetPattern(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf, pwin, pattern2d, psigma, ep->degrid, gridsample);

        // change analysis and synthesis window to constant to show
        std::fill_n(iop->wanxl, ep->ow, 1.0f);
        std::fill_n(iop->wanxr, ep->ow, 1.0f);
        std::fill_n(iop->wsynxl, ep->ow, 1.0f);
        std::fill_n(iop->wsynxr, ep->ow, 1.0f);
        std::fill_n(iop->wanyl, ep->oh, 1.0f);
        std::fill_n(iop->wanyr, ep->oh, 1.0f);
        std::fill_n(iop->wsynyl, ep->oh, 1.0f);
        std::fill_n(iop->wsynyr, ep->oh, 1.0f);

        // put source bytes to float array of overlapped blocks
        // cur frame
        FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, !ep->vi.Format.IsFamilyRGB);
        // make FFT 2D
        fftfp.fftwf_execute_dft_r2c(plan, in, outrez);

        PutPatternOnly(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf);
        // do inverse 2D FFT, get filtered 'in' array
        fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);

        // make destination frame plane from current overlaped blocks
        OverlapToCover(ep, iop, in, norm, coverbuf, coverwidth, coverpitch, !ep->vi.Format.IsFamilyRGB);
        CoverToFrame(ep, plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);
        int psigmaint = ((int)(10 * psigma)) / 10;
        int psigmadec = (int)((psigma - psigmaint) * 10);
        wsprintf(messagebuf, " frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);
        // TODO: DrawString(dst, vi, 0, 0, messagebuf);
        thread_id_store[thread_id] = false;
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
      pattern2d,
      pattern3d,
      ep->beta,
      ep->degrid,
      gridsample,
      ep->sharpen,
      sigmaSquaredSharpenMinNormed,
      sigmaSquaredSharpenMaxNormed,
      wsharpen,
      ep->dehalo,
      wdehalo,
      ht2n,
      covar,
      covarProcess,
      sigmaSquaredNoiseNormed2D,
      ep->kratio * ep->kratio
    };

    fftwf_complex* apply_in[5];

    if (btcur > 0) // Wiener
    {
      if (btcur == 1) // 2D
      {
        // cur frame
        FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
        fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
        ffp.Apply2D(outrez, sfp);
        if (ep->pfactor != 0)
          ffp.Sharpen(outrez, sfp);
      }
      else // 3D
      {
        static bool pattern3d_initialized = false;
        static std::mutex init3d_mutex;
        if (!pattern3d_initialized) {
          std::lock_guard<std::mutex> lock(init3d_mutex);
          if (!pattern3d_initialized)
            Pattern2Dto3D(pattern2d, ep->bh, outwidth, outpitch, (float)btcur, pattern3d);
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
        for (auto i = from; i <= to; i++)
        {
          std::lock_guard<std::mutex> lock1(cache_mutex);
          if (fftcache->exists(n+i)) {
            apply_in[2+i] = fftcache->get_read(n+i);
          }
          else {
            DSFrame frame;
            DSFrame* pframe;
            apply_in[2+i] = fftcache->get_write(n+i);
            if (in_frames.find(n+i) == in_frames.end())
              pframe = &(frame = (*fetch_frame)(n+i));
            else
              pframe = &in_frames[n+i];
            FrameToCover(ep, plane, pframe->SrcPointers[plane], coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
            CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);

            fftfp.fftwf_execute_dft_r2c(plan, in, apply_in[2+i]);
          }
        }

        ffp.Apply3D(apply_in, outrez, sfp);
        ffp.Sharpen(outrez, sfp);
      }

      // do inverse FFT, get filtered 'in' array
      fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep, iop, in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep, plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);
    }
    else if (ep->bt == 0) //Kalman filter
    {
      if (n == 0) {
        thread_id_store[thread_id] = false;
        return src; // first frame  not processed
      }
      /* PF 170302 comment: accumulated error?
        orig = BlankClip(...)
        new = orig.FFT3DFilter(sigma = 3, plane = 4, bt = 0, degrid = 0)
        Subtract(orig,new).Levels(120, 1, 255 - 120, 0, 255, coring = false)
      */

      // put source bytes to float array of overlapped blocks
      // cur frame
      FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
      CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
      ffp.Kalman(outrez, outLast, sfp);

      // copy outLast to outrez
      memcpy(outrez, outLast, outsize * sizeof(fftwf_complex));  //v.0.9.2
      ffp.Sharpen(outrez, sfp);
      // do inverse FFT 2D, get filtered 'in' array
      // note: input "out" array is destroyed by execute algo.
      // that is why we must have its copy in "outLast" array
      fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep, iop, in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep, plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);

    }
    else if (ep->bt == -1) /// sharpen only
    {
      FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
      CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
      ffp.Sharpen(outrez, sfp);
      // do inverse FFT 2D, get filtered 'in' array
      fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep, iop, in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep, plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);

    }

    thread_id_store[thread_id] = false;

    // As we now are finished processing the image, we return the destination image.
    return dst;
  }

  bool CacheRefresh(int n) {
    if (!fftcache) return false;
    std::lock_guard<std::mutex> lock(cache_mutex);
    return fftcache->refresh(n);
  }
};
