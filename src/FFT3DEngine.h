#ifndef __FFT3DENGINE_H__
#define __FFT3DENGINE_H__

#include "common.h"
#include "functions.h"
#include "helper.h"
#include "buffer.h"
#include "cache.hpp"
#include <atomic>

template <typename Interface>
class FFT3DEngine {
  EngineParams* ep;
  IOParams *iop;
  Interface* super;
  int plane; // color plane

  // additional parameterss
  float *in;
  fftwf_complex *out, *outprev, *outnext, *outtemp, *outprev2, *outnext2;
  fftwf_complex *outrez, *gridsample; //v1.8
  fftwf_plan plan, planinv, plan1;
  int outwidth;
  int outpitch; //v.1.7

  int outsize;
  int howmanyblocks;

  int ndim[2];
  int inembed[2];
  int onembed[2];

  float *wsharpen;
  float *wdehalo;

  int nlast;// frame number at last step, PF: multithread warning, used for cacheing when sequential access detected
  int btcurlast;  //v1.7 to prevent multiple Pattern2Dto3D for the same btcurrent. btcurrent can change and may differ from bt for e.g. first/last frame

  fftwf_complex *outLast, *covar, *covarProcess;
  float sigmaSquaredNoiseNormed;
  float sigmaSquaredNoiseNormed2D;
  float sigmaNoiseNormed2D;
  float sigmaMotionNormed;
  float sigmaSquaredSharpenMinNormed;
  float sigmaSquaredSharpenMaxNormed;
  float ht2n; // halo threshold squared normed
  float norm; // normalization factor

  BYTE *coverbuf; //  block buffer covering the frame without remainders (with sufficient width and heigth)
  int coverwidth;
  int coverheight;
  int coverpitch;

  int mirw; // mirror width for padding
  int mirh; // mirror height for padding

  float *pwin;
  float *pattern2d;
  float *pattern3d;
  bool isPatternSet;
  float psigma;
  char *messagebuf;

  // added in v.0.9 for delayed FFTW3.DLL loading
  struct FFTFunctionPointers fftfp;

  int CPUFlags;

  // avs+
  int pixelsize;
  int bits_per_pixel;

  int cachesize;//v1.8
  cache<fftwf_complex> *fftcache;

  int _instance_id; // debug unique id
  std::atomic<bool> reentrancy_check;

  struct FilterFunctionPointers ffp;

public:
  FFT3DEngine(Interface* _super, EngineParams _ep, int _plane) :
  super(_super), ep(new EngineParams(_ep)), plane(_plane), iop(new IOParams()) {
    static int id = 0; _instance_id = id++;
    reentrancy_check = false;

    int i, j;

    float factor;
    switch(ep->byte_per_channel) {
      case 1: factor = 1.0f; break;
      case 2: factor = float(1 << (ep->bit_per_channel-8)); break;
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

    int xRatioShift = ep->IsChroma ? ep->ssw : 0;
    int yRatioShift = ep->IsChroma ? ep->ssh : 0;

    iop->nox = ((super->width() >> xRatioShift) - ep->ow + (ep->bw - ep->ow - 1)) / (ep->bw - ep->ow);
    iop->noy = ((super->height() >> yRatioShift) - ep->oh + (ep->bh - ep->oh - 1)) / (ep->bh - ep->oh);


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
    coverbuf = (byte*)malloc(coverheight*coverpitch*ep->byte_per_channel);

    int insize = ep->bw*ep->bh*iop->nox*iop->noy;
    in = (float *)fftfp.fftwf_malloc(sizeof(float) * insize);
    outwidth = ep->bw / 2 + 1; // width (pitch) of complex fft block
    outpitch = ((outwidth + 3) / 4) * 4; // must be even for SSE - v1.7
    outsize = outpitch*ep->bh*iop->nox*iop->noy; // replace outwidth to outpitch here and below in v1.7
    if (ep->bt == 0) // Kalman
    {
      outLast = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
      covar = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
      covarProcess = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
    }
    outrez = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize); //v1.8
    gridsample = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize); //v1.8

    cachesize = ep->bt + 4;
    fftcache = new cache<fftwf_complex>(cachesize, outsize, fftfp.fftwf_malloc, fftfp.fftwf_free);

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

    fftfp.fftwf_plan_with_nthreads(ep->ncpu);

    plan = fftfp.fftwf_plan_many_dft_r2c(rank, ndim, howmanyblocks,
      in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags);
    if (plan == NULL)
      throw("FFTW plan error");

    planinv = fftfp.fftwf_plan_many_dft_c2r(rank, ndim, howmanyblocks,
      outrez, onembed, ostride, odist, in, inembed, istride, idist, planFlags);
    if (planinv == NULL)
      throw("FFTW plan error");

    fftfp.fftwf_plan_with_nthreads(1);

    iop->wanxl = new float[ep->ow];
    iop->wanxr = new float[ep->ow];
    iop->wanyl = new float[ep->oh];
    iop->wanyr = new float[ep->oh];

    iop->wsynxl = new float[ep->ow];
    iop->wsynxr = new float[ep->ow];
    iop->wsynyl = new float[ep->oh];
    iop->wsynyr = new float[ep->oh];

    wsharpen = (float*)fftfp.fftwf_malloc(ep->bh*outpitch * sizeof(float));
    wdehalo = (float*)fftfp.fftwf_malloc(ep->bh*outpitch * sizeof(float));

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
        float d1 = sqrt(d2);
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

    // init nlast
    nlast = -999; // init as nonexistant
    btcurlast = -999; // init as nonexistant

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

    pattern2d = (float*)fftfp.fftwf_malloc(ep->bh*outpitch * sizeof(float)); // noise pattern window array
    pattern3d = (float*)fftfp.fftwf_malloc(ep->bh*outpitch * sizeof(float)); // noise pattern window array

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

    // avs+
    switch (ep->byte_per_channel) {
    case 1: memset(coverbuf, 255, coverheight*coverpitch); 
      break;
    case 2: std::fill_n((uint16_t *)coverbuf, coverheight*coverpitch, (1 << ep->bit_per_channel) - 1); 
      break; // 255 
    case 4: std::fill_n((float *)coverbuf, coverheight*coverpitch, 1.0f); 
      break; // 255 
    }
    CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, false);
    // make FFT 2D
    fftfp.fftwf_execute_dft_r2c(plan1, in, gridsample);

    messagebuf = (char *)malloc(80); //1.8.5

  }
  ~FFT3DEngine() {
    // This is where you can deallocate any memory you might have used.
    fftfp.fftwf_destroy_plan(plan);
    fftfp.fftwf_destroy_plan(plan1);
    fftfp.fftwf_destroy_plan(planinv);
    fftfp.fftwf_free(in);
    //	fftfp.fftwf_free(out);
    delete[] iop->wanxl;
    delete[] iop->wanxr;
    delete[] iop->wanyl;
    delete[] iop->wanyr;
    delete[] iop->wsynxl;
    delete[] iop->wsynxr;
    delete[] iop->wsynyl;
    delete[] iop->wsynyr;
    fftfp.fftwf_free(wsharpen);
    fftfp.fftwf_free(wdehalo);
    delete[] pwin;
    fftfp.fftwf_free(pattern2d);
    fftfp.fftwf_free(pattern3d);
    fftfp.fftwf_free(outrez);
    if (ep->bt == 0) // Kalman
    {
      fftfp.fftwf_free(outLast);
      fftfp.fftwf_free(covar);
      fftfp.fftwf_free(covarProcess);
    }
    free(coverbuf);
    delete fftcache;
    delete ep;
    delete iop;
    fftfp.fftwf_free(gridsample); //fixed memory leakage in v1.8.5

    fftfp.free();
    free(messagebuf); //v1.8.5
  }

  typename Interface::Frametype GetFrame(int n) {

    typename Interface::Frametype src, psrc, dst;
    int pxf, pyf;
    //	char debugbuf[1536];
    //	wsprintf(debugbuf,"FFT3DFilter: n=%d \n", n);
    //	OutputDebugString(debugbuf);
    //	fftwf_complex * tmpoutrez, *tmpoutnext, *tmpoutprev, *tmpoutnext2; // store pointers
    if (reentrancy_check) {
      throw("cannot work in reentrant multithread mode!");
    }
    reentrancy_check = true;

    if (ep->pfactor != 0 && isPatternSet == false && ep->pshow == false) // get noise pattern
    {
      psrc = super->GetFrame(super->child, ep->pframe); // get noise patterme
      auto sptr = psrc->GetReadPtr(plane);
      ep->framewidth = super->width(psrc, plane);
      ep->frameheight = super->height(psrc, plane);
      ep->framepitch = super->stride(psrc, plane) / ep->byte_per_channel;

      // put source bytes to float array of overlapped blocks
      FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
      CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      // make FFT 2D
      fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
      if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
        FindPatternBlock(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, ep->px, ep->py, pwin, ep->degrid, gridsample);
      SetPattern(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, ep->px, ep->py, pwin, pattern2d, psigma, ep->degrid, gridsample);
      isPatternSet = true;
    }
    else if (ep->pfactor != 0 && ep->pshow == true)
    {
      // show noise pattern window
      src = super->GetFrame(super->child, n); // get noise pattern frame
      dst = super->NewVideoFrame();
      auto sptr = src->GetReadPtr(plane);
      auto dptr = dst->GetWritePtr(plane);
      ep->framewidth = super->width(src, plane);
      ep->frameheight = super->height(src, plane);
      ep->framepitch = super->stride(src, plane) / ep->byte_per_channel;
      // TODO: CopyFrame(src, dst, vi, plane, env);

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
      CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, !ep->IsRGB);
      // make FFT 2D
      fftfp.fftwf_execute_dft_r2c(plan, in, outrez);

      PutPatternOnly(outrez, outwidth, outpitch, ep->bh, iop->nox, iop->noy, pxf, pyf);
      // do inverse 2D FFT, get filtered 'in' array
      fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);

      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep, iop, in, norm, coverbuf, coverwidth, coverpitch, !ep->IsRGB);
      CoverToFrame(ep, plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);
      int psigmaint = ((int)(10 * psigma)) / 10;
      int psigmadec = (int)((psigma - psigmaint) * 10);
      wsprintf(messagebuf, " frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);
      // TODO: DrawString(dst, vi, 0, 0, messagebuf);

      reentrancy_check = false;
      return dst; // return pattern frame to show
    }

    // Request frame 'n' from the child (source) clip.
    src = super->GetFrame(super->child, n);
    dst = super->NewVideoFrame();
    auto sptr = src->GetReadPtr(plane);
    auto dptr = dst->GetWritePtr(plane);
    ep->framewidth = super->width(src, plane);
    ep->frameheight = super->height(src, plane);
    ep->framepitch = super->stride(src, plane) / ep->byte_per_channel;

    int btcur = ep->bt; // bt used for current frame
  //	if ( (bt/2 > n) || bt==3 && n==vi.num_frames-1 )
    if ((ep->bt / 2 > n) || (ep->bt - 1) / 2 > (ep->frames - 1 - n))
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
      if (btcur != btcurlast) // !! global state, multithreading warning
        Pattern2Dto3D(pattern2d, ep->bh, outwidth, outpitch, (float)btcur, pattern3d);

      if (btcur == 1) // 2D
      {
        // cur frame
        FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
        CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
              // make FFT 2D
        fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
        ffp.Apply2D(outrez, sfp);
        if (ep->pfactor != 0)
          ffp.Sharpen(outrez, sfp);
      }
      else // 3D
      {
        int from = -btcur / 2;
        int to = (btcur - 1) / 2;
        /* apply_in[]
        *
        * |   |   0   |   1   |   2   |   3   |   4   |
        * |   | prev2 | prev1 |current| next1 | next2 |
        * +---+-------+-------+-------+-------+-------+
        * | 2 |       |   o   |   o   |       |       |
        * | 3 |       |   o   |   o   |   o   |       |
        * | 4 |   o   |   o   |   o   |   o   |       |build\Release_\msvc-x64
        * | 5 |   o   |   o   |   o   |   o   |   o   |
        *
        */
        for (auto i = from; i <= to; i++)
        {
          if (fftcache->exists(n+i)) {
            apply_in[2+i] = fftcache->get_read(n+i);
          }
          else {
            apply_in[2+i] = fftcache->get_write(n+i);
            auto frame = i == 0 ? src : super->GetFrame(super->child, n+i);
            FrameToCover(ep, plane, frame->GetReadPtr(), coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
            CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
            fftfp.fftwf_execute_dft_r2c(plan, in, apply_in[2+i]);
            if (i != 0) super->FreeFrame(frame);
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
      if (n == 0)
      {
        reentrancy_check = false;
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
      // make FFT 2D
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
      //		env->MakeWritable(&src);
          // put source bytes to float array of overlapped blocks
      FrameToCover(ep, plane, sptr, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh);
      CoverToOverlap(ep, iop, in, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      // make FFT 2D
      fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
      ffp.Sharpen(outrez, sfp);
      // do inverse FFT 2D, get filtered 'in' array
      fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);
      // make destination frame plane from current overlaped blocks
      OverlapToCover(ep, iop, in, norm, coverbuf, coverwidth, coverpitch, ep->IsChroma);
      CoverToFrame(ep, plane, coverbuf, coverwidth, coverheight, coverpitch, dptr, mirw, mirh);

    }

    if (btcur == ep->bt)
    {// for normal step
      nlast = n; // set last frame to current
    }
    btcurlast = btcur;

    // As we now are finished processing the image, we return the destination image.
    reentrancy_check = false;
    return dst;
  }
};

#endif