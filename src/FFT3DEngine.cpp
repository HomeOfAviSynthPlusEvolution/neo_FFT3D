#define VERSION_NUMBER 2.6

#include "FFT3DEngine.h"

template class FFT3DEngine<AVSFilter>;

template <class Interface>
void FFT3DEngine<Interface>::load_buffer(int plane, typename Interface::AFrame &src, byte *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, int bits_per_pixel)
{
  const byte *srcp;
  int src_width, src_height, src_pitch;

  srcp = src->GetReadPtr(plane);
  src_height = super->height(src, plane);
  src_width = super->width(src, plane);
  src_pitch = super->stride(src, plane) / super->pixelsize();
  switch (super->depth())
  {
  case 8: PlanarPlaneToCoverbuf<uint8_t>(srcp, src_width, src_height, src_pitch, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced); break;
  case 10: case 12: case 14: case 16: PlanarPlaneToCoverbuf<uint16_t>((uint16_t *)srcp, src_width, src_height, src_pitch, (uint16_t *)coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced); break;
  case 32: PlanarPlaneToCoverbuf<float>((float *)srcp, src_width, src_height, src_pitch, (float *)coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced); break;
  }
}

template <class Interface>
void FFT3DEngine<Interface>::store_buffer(int plane, const byte *coverbuf, int coverwidth, int coverheight, int coverpitch, typename Interface::AFrame &dst, int mirw, int mirh, bool interlaced, int bits_per_pixel)
{
  byte *dstp;
  int dst_width, dst_height, dst_pitch;

  dstp = dst->GetWritePtr(plane);
  dst_height = super->height(dst, plane);
  dst_width = super->width(dst, plane);
  dst_pitch = dst->GetPitch(plane) / super->pixelsize();
  switch (bits_per_pixel)
  {
  case 8: CoverbufToPlanarPlane<uint8_t>(coverbuf, coverwidth, coverheight, coverpitch, dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced); break;
  case 10: case 12: case 14: case 16: CoverbufToPlanarPlane<uint16_t>((uint16_t *)coverbuf, coverwidth, coverheight, coverpitch, (uint16_t *)dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced); break;
  case 32: CoverbufToPlanarPlane<float>((float *)coverbuf, coverwidth, coverheight, coverpitch, (float *)dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced); break;
  }
}

template <class Interface>
FFT3DEngine<Interface>::FFT3DEngine(Interface* _super, EngineParams _ep, int _plane) :
  super(_super), ep(new EngineParams(_ep)), plane(_plane) {

  static int id = 0; _instance_id = id++;
  reentrancy_check = false;
  _RPT1(0, "FFT3DFilter.Create instance_id=%d\n", _instance_id);

  int i, j;

  pixelsize = super->pixelsize();
  bits_per_pixel = super->depth();

  float factor;
  if (pixelsize == 1) factor = 1.0f;
  else if (pixelsize == 2) factor = float(1 << (bits_per_pixel-8));
  else // float
    factor = 1 / 255.0f;

  ep->sigma *= factor;
  ep->sigma2 *= factor;
  ep->sigma3 *= factor;
  ep->sigma4 *= factor;
  ep->smin *= factor;
  ep->smax *= factor;

  if (ep->ow * 2 > ep->bw) throw("Must not be 2*ow > ep->bw");
  if (ep->oh * 2 > ep->bh) throw("Must not be 2*oh > bh");
  if (ep->ow < 0) ep->ow = ep->bw / 3; // changed from ep->bw/4 to ep->bw/3 in v.1.2
  if (ep->oh < 0) ep->oh = ep->bh / 3; // changed from bh/4 to bh/3 in v.1.2

  if (ep->bt < -1 || ep->bt >5) throw("bt must be -1(Sharpen), 0(Kalman), 1,2,3,4,5(Wiener)");

/*
    (Parameter bt = 1) 
      2D(spatial) Wiener filter for spectrum data. Use current frame data only.
      Reduce weak frequencies(with small power spectral density) by optimal Wiener filter with some 
      given noise value. Sharpening and denoising are simultaneous in this mode.
    (Parameter bt = 2) 
      3D Wiener filter for spectrum data.
      Add third dimension to FFT by using previous and current frame data. 
      Reduce weak frequencies (with small power spectral density) by optimal Wiener filter with some 
      given noise value.
    (Parameter bt = 3) 
      Also 3D Wiener filter for spectrum data with previous, current and next frame data.
    (Parameter bt = 4) 
      Also 3D Wiener filter for spectrum data with two previous, current and next frame data.
    (Parameter bt = 5) 
      Also 3D Wiener filter for spectrum data with two previous, current and two next frames data.
    (Parameter bt = 0) 
      Temporal Kalman filter for spectrum data.
      Use all previous frames data to get estimation of cleaned current data with optimal recursive 
      data process algorithm.
      The filter starts work with small(= 1) gain(degree of noise reducing), and than gradually(in frames sequence) increases the gain 
      if inter - frame local spectrum(noise) variation is small.
      So, Kalman filter can provide stronger noise reduction than Wiener filter.
      The Kalman filter gain is limited by some given noise value.
      The local gain(and filter work) is reset to 1 when local variation exceeds the given threshold
      (due to motion, scene change, etc).
      So, the Kalman filter output is history - dependent(on frame taken as a start filtered frame).

      PF: bt==0 have to be MT_SERIALIZED because it maintains internal history
*/

  // plane: 0 - luma(Y), 1 - chroma U, 2 - chroma V
  if (super->IsPlanar()) // also for grey
  {
    isRGB = !(super->IsYUV() || super->IsYUVA());
    int xRatioShift = isRGB ? 0 : super->ssw(plane);
    int yRatioShift = isRGB ? 0 : super->ssh(plane);

    nox = ((super->width() >> xRatioShift) - ep->ow + (ep->bw - ep->ow - 1)) / (ep->bw - ep->ow);
    noy = ((super->height() >> yRatioShift) - ep->oh + (ep->bh - ep->oh - 1)) / (ep->bh - ep->oh);
  }
  else
    throw("video must be planar");


  // padding by 1 block per side
  nox += 2;
  noy += 2;
  mirw = ep->bw - ep->ow; // set mirror size as block interval
  mirh = ep->bh - ep->oh;

  if (ep->beta < 1)
    throw("beta must be not less 1.0");

  int istat;

  fftfp.load();

  istat = fftfp.fftwf_init_threads();

  coverwidth = nox*(ep->bw - ep->ow) + ep->ow;
  coverheight = noy*(ep->bh - ep->oh) + ep->oh;
  coverpitch = ((coverwidth + 15) / 16) * 16; // align to 16 elements. Pitch is element-granularity. For byte pitch, multiply is by pixelsize
  coverbuf = (byte*)malloc(coverheight*coverpitch*pixelsize);

  int insize = ep->bw*ep->bh*nox*noy;
  in = (float *)fftfp.fftwf_malloc(sizeof(float) * insize);
  outwidth = ep->bw / 2 + 1; // width (pitch) of complex fft block
  outpitch = ((outwidth + 3) / 4) * 4; // must be even for SSE - v1.7
  outsize = outpitch*ep->bh*nox*noy; // replace outwidth to outpitch here and below in v1.7
//	out = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
//	if (bt >= 2)
//		outprev = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
//	if (bt >= 3)
//		outnext = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
//	if (bt >= 4)
//		outprev2 = (fftwf_complex *)fftfp.fftwf_malloc(sizeof(fftwf_complex) * outsize);
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
  howmanyblocks = nox*noy;

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

  wanxl = (float*)malloc(ep->ow * sizeof(float));
  wanxr = (float*)malloc(ep->ow * sizeof(float));
  wanyl = (float*)malloc(ep->oh * sizeof(float));
  wanyr = (float*)malloc(ep->oh * sizeof(float));

  wsynxl = (float*)malloc(ep->ow * sizeof(float));
  wsynxr = (float*)malloc(ep->ow * sizeof(float));
  wsynyl = (float*)malloc(ep->oh * sizeof(float));
  wsynyr = (float*)malloc(ep->oh * sizeof(float));

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
      wanxl[i] = cosf(pi*(i - ep->ow + 0.5f) / (ep->ow * 2)); // left analize window (half-cosine)
      wanxr[i] = cosf(pi*(i + 0.5f) / (ep->ow * 2)); // right analize window (half-cosine)
    }
    for (i = 0; i < ep->oh; i++)
    {
      wanyl[i] = cosf(pi*(i - ep->oh + 0.5f) / (ep->oh * 2));
      wanyr[i] = cosf(pi*(i + 0.5f) / (ep->oh * 2));
    }
    // use the same windows for synthesis too.
    for (i = 0; i < ep->ow; i++)
    {
      wsynxl[i] = wanxl[i]; // left  window (half-cosine)

      wsynxr[i] = wanxr[i]; // right  window (half-cosine)
    }
    for (i = 0; i < ep->oh; i++)
    {
      wsynyl[i] = wanyl[i];
      wsynyr[i] = wanyr[i];
    }
  }
  else if (ep->wintype == 1) // added in v.1.4
  {
    // define analysis windows as more flat (to decrease grid)
    for (i = 0; i < ep->ow; i++)
    {
      wanxl[i] = sqrt(cosf(pi*(i - ep->ow + 0.5f) / (ep->ow * 2)));
      wanxr[i] = sqrt(cosf(pi*(i + 0.5f) / (ep->oh * 2)));
    }
    for (i = 0; i < ep->oh; i++)
    {
      wanyl[i] = sqrt(cosf(pi*(i - ep->oh + 0.5f) / (ep->oh * 2)));
      wanyr[i] = sqrt(cosf(pi*(i + 0.5f) / (ep->oh * 2)));
    }
    // define synthesis as supplenent to rised cosine (Hanning)
    for (i = 0; i < ep->ow; i++)
    {
      wsynxl[i] = wanxl[i] * wanxl[i] * wanxl[i]; // left window
      wsynxr[i] = wanxr[i] * wanxr[i] * wanxr[i]; // right window
    }
    for (i = 0; i < ep->oh; i++)
    {
      wsynyl[i] = wanyl[i] * wanyl[i] * wanyl[i];
      wsynyr[i] = wanyr[i] * wanyr[i] * wanyr[i];
    }
  }
  else //  (ep->wintype==2) - added in v.1.4
  {
    // define analysis windows as flat (to prevent grid)
    for (i = 0; i < ep->ow; i++)
    {
      wanxl[i] = 1;
      wanxr[i] = 1;
    }
    for (i = 0; i < ep->oh; i++)
    {
      wanyl[i] = 1;
      wanyr[i] = 1;
    }
    // define synthesis as rised cosine (Hanning)
    for (i = 0; i < ep->ow; i++)
    {
      wsynxl[i] = cosf(pi*(i - ep->ow + 0.5f) / (ep->ow * 2));
      wsynxl[i] = wsynxl[i] * wsynxl[i];// left window (rised cosine)
      wsynxr[i] = cosf(pi*(i + 0.5f) / (ep->ow * 2));
      wsynxr[i] = wsynxr[i] * wsynxr[i]; // right window (falled cosine)
    }
    for (i = 0; i < ep->oh; i++)
    {
      wsynyl[i] = cosf(pi*(i - ep->oh + 0.5f) / (ep->oh * 2));
      wsynyl[i] = wsynyl[i] * wsynyl[i];
      wsynyr[i] = cosf(pi*(i + 0.5f) / (ep->oh * 2));
      wsynyr[i] = wsynyr[i] * wsynyr[i];
    }
  }

  // window for sharpen
  for (j = 0; j < ep->bh; j++)
  {
    int dj = j;
    if (j >= ep->bh / 2)
      dj = ep->bh - j;
    float d2v = float(dj*dj)*(ep->svr*ep->svr) / ((ep->bh / 2)*(ep->bh / 2)); // v1.7
    for (i = 0; i < outwidth; i++)
    {
      float d2 = d2v + float(i*i) / ((ep->bw / 2)*(ep->bw / 2)); // distance_2 - v1.7
      wsharpen[i] = 1 - exp(-d2 / (2 * ep->scutoff*ep->scutoff));
    }
    wsharpen += outpitch;
  }
  wsharpen -= outpitch*ep->bh; // restore pointer

  // window for dehalo - added in v1.9
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
      wdehalo[i] = exp(-0.7f*d2*ep->hr*ep->hr) - exp(-d2*ep->hr*ep->hr); // some window with max around 1/hr, small at low and high frequencies
      if (wdehalo[i] > wmax) wmax = wdehalo[i]; // for normalization
    }
    wdehalo += outpitch;
  }
  wdehalo -= outpitch*ep->bh; // restore pointer

  for (j = 0; j < ep->bh; j++)
  {
    for (i = 0; i < outwidth; i++)
    {
      wdehalo[i] /= wmax; // normalize
    }
    wdehalo += outpitch;
  }
  wdehalo -= outpitch*ep->bh; // restore pointer

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
    fill_complex(outLast, outsize, 0, 0);
    fill_complex(covar, outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D); // fixed bug in v.1.1
    fill_complex(covarProcess, outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D);// fixed bug in v.1.1
  }

  CPUFlags = GetCPUFlags(); //re-enabled in v.1.9
  ffp.set_ffp(CPUFlags, ep->degrid, ep->pfactor, ep->bt);
  mean = (float*)malloc(nox*noy * sizeof(float));

  pwin = (float*)malloc(ep->bh*outpitch * sizeof(float)); // pattern window array

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
  switch (pixelsize) {
  case 1: memset(coverbuf, 255, coverheight*coverpitch); 
    break;
  case 2: std::fill_n((uint16_t *)coverbuf, coverheight*coverpitch, (1 << bits_per_pixel) - 1); 
    break; // 255 
  case 4: std::fill_n((float *)coverbuf, coverheight*coverpitch, 1.0f); 
    break; // 255 
  }
  FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, false);
  // make FFT 2D
  fftfp.fftwf_execute_dft_r2c(plan1, in, gridsample);

  messagebuf = (char *)malloc(80); //1.8.5

//	fullwinan = (float *)fftfp.fftwf_malloc(sizeof(float) * insize);
//	FFT3DEngine<Interface>::InitFullWin(fullwinan, wanxl, wanxr, wanyl, wanyr);
//	fullwinsyn = (float *)fftfp.fftwf_malloc(sizeof(float) * insize);
//	FFT3DEngine<Interface>::InitFullWin(fullwinsyn, wsynxl, wsynxr, wsynyl, wsynyr);

}
//-------------------------------------------------------------------------------------------

// This is where any actual destructor code used goes
template <class Interface>
FFT3DEngine<Interface>::~FFT3DEngine() {
  // This is where you can deallocate any memory you might have used.
  fftfp.fftwf_destroy_plan(plan);
  fftfp.fftwf_destroy_plan(plan1);
  fftfp.fftwf_destroy_plan(planinv);
  fftfp.fftwf_free(in);
  //	fftfp.fftwf_free(out);
  free(wanxl);
  free(wanxr);
  free(wanyl);
  free(wanyr);
  free(wsynxl);
  free(wsynxr);
  free(wsynyl);
  free(wsynyr);
  fftfp.fftwf_free(wsharpen);
  fftfp.fftwf_free(wdehalo);
  free(mean);
  free(pwin);
  fftfp.fftwf_free(pattern2d);
  fftfp.fftwf_free(pattern3d);
  //	if (bt >= 2)
  //		fftfp.fftwf_free(outprev);
  //	if (bt >= 3)
  //		fftfp.fftwf_free(outnext);
  //	if (bt >= 4)
  //		fftfp.fftwf_free(outprev2);
  fftfp.fftwf_free(outrez);
  if (ep->bt == 0) // Kalman
  {
    fftfp.fftwf_free(outLast);
    fftfp.fftwf_free(covar);
    fftfp.fftwf_free(covarProcess);
  }
  free(coverbuf);
  delete fftcache;
  fftfp.fftwf_free(gridsample); //fixed memory leakage in v1.8.5
//	fftfp.fftwf_free(fullwinan);
//	fftfp.fftwf_free(fullwinsyn);
//	fftfp.fftwf_free(shiftedprev);
//	fftfp.fftwf_free(shiftedprev2);
//	fftfp.fftwf_free(shiftednext);
//	fftfp.fftwf_free(shiftednext2);
//	fftfp.fftwf_free(fftcorrel);
//	fftfp.fftwf_free(correl);
//	free(xshifts);
//	free(yshifts);

  fftfp.free();
  free(messagebuf); //v1.8.5
}

//-----------------------------------------------------------------------
// put source bytes to float array of overlapped blocks
// use analysis windows
//
template <class Interface>
void FFT3DEngine<Interface>::InitOverlapPlane(float * inp0, const byte *srcp0, int src_pitch, bool chroma)
{
  // for float: chroma center is also 0.0
  if (chroma) {
    switch (bits_per_pixel) {
    case 8: do_InitOverlapPlane<uint8_t, 8, true>(inp0, srcp0, src_pitch); break;
    case 10: do_InitOverlapPlane<uint16_t, 10, true>(inp0, srcp0, src_pitch); break;
    case 12: do_InitOverlapPlane<uint16_t, 12, true>(inp0, srcp0, src_pitch); break;
    case 14: do_InitOverlapPlane<uint16_t, 14, true>(inp0, srcp0, src_pitch); break;
    case 16: do_InitOverlapPlane<uint16_t, 16, true>(inp0, srcp0, src_pitch); break;
    case 32: do_InitOverlapPlane<float, 8 /*n/a*/, true>(inp0, srcp0, src_pitch); break;
    }
  }
  else {
    switch (bits_per_pixel) {
    case 8: do_InitOverlapPlane<uint8_t, 8, false>(inp0, srcp0, src_pitch); break;
    case 10: do_InitOverlapPlane<uint16_t, 10, false>(inp0, srcp0, src_pitch); break;
    case 12: do_InitOverlapPlane<uint16_t, 12, false>(inp0, srcp0, src_pitch); break;
    case 14: do_InitOverlapPlane<uint16_t, 14, false>(inp0, srcp0, src_pitch); break;
    case 16: do_InitOverlapPlane<uint16_t, 16, false>(inp0, srcp0, src_pitch); break;
    case 32: do_InitOverlapPlane<float, 8 /*n/a*/, false>(inp0, srcp0, src_pitch); break;
    }
  }
}

template <class Interface>
template<typename pixel_t, int _bits_per_pixel, bool chroma>
void FFT3DEngine<Interface>::do_InitOverlapPlane(float * inp0, const byte *srcp0, int src_pitch)
{
  // pitch is pixel_t granularity, can be used directly as scrp+=pitch
  int w, h;
  int ihx, ihy;
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp0);// + (hrest/2)*src_pitch + wrest/2; // centered
  float ftmp;
  int xoffset = ep->bh*ep->bw - (ep->bw - ep->ow); // skip frames
  int yoffset = ep->bw*nox*ep->bh - ep->bw*(ep->bh - ep->oh); // vertical offset of same block (overlap)

  float *inp = inp0;
  //	char debugbuf[1536];
  //	wsprintf(debugbuf,"FFT3DFilter: InitOverlapPlane");
  //	OutputDebugString(debugbuf);
  typedef typename std::conditional<sizeof(pixel_t) == 4, float, int>::type cast_t;
  // for float: chroma center is also 0.0
  constexpr cast_t planeBase = sizeof(pixel_t) == 4 ? cast_t(chroma ? 0.0f : 0.0f) : cast_t(chroma ? (1 << (_bits_per_pixel - 1)) : 0); // anti warning

  ihy = 0; // first top (big non-overlapped) part
  {
    for (h = 0; h < ep->oh; h++)
    {
      inp = inp0 + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanxl[w] * wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ep->ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *wanxl[w];   // overlapped Copy - next block
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last part (non-overlapped) of line of last block
      {
        inp[w] = float(wanxr[w] * wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->ow;
      srcp += ep->ow;
      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
    for (h = ep->oh; h < ep->bh - ep->oh; h++)
    {
      inp = inp0 + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanxl[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ep->ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *wanxl[w];   // overlapped Copy - next block
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last part (non-overlapped) line of last block
      {
        inp[w] = float(wanxr[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
  }

  for (ihy = 1; ihy < noy; ihy += 1) // middle vertical
  {
    for (h = 0; h < ep->oh; h++) // top overlapped part
    {
      inp = inp0 + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w] * (srcp[w] - planeBase));
        inp[w] = ftmp*wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*wanyl[h];   // y overlapped
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        ftmp = float((srcp[w] - planeBase));
        inp[w] = ftmp*wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*wanyl[h];   // y overlapped
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w] * wanyr[h];
          inp[w + xoffset] = ftmp *wanxl[w] * wanyr[h];   // x overlapped
          inp[w + yoffset] = ftmp * wanxr[w] * wanyl[h];
          inp[w + xoffset + yoffset] = ftmp *wanxl[w] * wanyl[h];   // x overlapped
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanyr[h];
          inp[w + yoffset] = ftmp * wanyl[h];
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp*wanyr[h];
        inp[w + yoffset] = ftmp*wanyl[h];
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical nonovelapped part
    for (h = 0; h < ep->bh - ep->oh - ep->oh; h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh)*ep->bw + h*ep->bw + yoffset;
      for (w = 0; w < ep->ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        ftmp = float((srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w];
          inp[w + xoffset] = ftmp *wanxl[w];   // x overlapped
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp;
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp;
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }

  }

  ihy = noy; // last bottom  part
  {
    for (h = 0; h < ep->oh; h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;
      for (w = 0; w < ep->ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w] * wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ep->ow; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        ftmp = float(wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep->bw - ep->ow;
      srcp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half line of block
        {
          float ftmp = float(wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w];
          inp[w + xoffset] = ftmp *wanxl[w];   // overlapped Copy
        }
        inp += ep->ow;
        inp += xoffset;
        srcp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += ep->bw - ep->ow - ep->ow;
        srcp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w] * wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ep->ow;
      srcp += ep->ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }

  }
}

//
//-----------------------------------------------------------------------------------------
// make destination frame plane from overlaped blocks
// use synthesis windows wsynxl, wsynxr, wsynyl, wsynyr
template <class Interface>
void FFT3DEngine<Interface>::DecodeOverlapPlane(float *inp0, float norm, byte *dstp0, int dst_pitch, bool chroma)
{
  if (chroma) {
    switch (bits_per_pixel) {
    case 8: do_DecodeOverlapPlane<uint8_t, 8, true>(inp0, norm, dstp0, dst_pitch); break;
    case 10: do_DecodeOverlapPlane<uint16_t, 10, true>(inp0, norm, dstp0, dst_pitch); break;
    case 12: do_DecodeOverlapPlane<uint16_t, 12, true>(inp0, norm, dstp0, dst_pitch); break;
    case 14: do_DecodeOverlapPlane<uint16_t, 14, true>(inp0, norm, dstp0, dst_pitch); break;
    case 16: do_DecodeOverlapPlane<uint16_t, 16, true>(inp0, norm, dstp0, dst_pitch); break;
    case 32: do_DecodeOverlapPlane<float, 8 /*n/a*/, true>(inp0, norm, dstp0, dst_pitch); break;
    }
  }
  else {
    switch (bits_per_pixel) {
    case 8: do_DecodeOverlapPlane<uint8_t, 8, false>(inp0, norm, dstp0, dst_pitch); break;
    case 10: do_DecodeOverlapPlane<uint16_t, 10, false>(inp0, norm, dstp0, dst_pitch); break;
    case 12: do_DecodeOverlapPlane<uint16_t, 12, false>(inp0, norm, dstp0, dst_pitch); break;
    case 14: do_DecodeOverlapPlane<uint16_t, 14, false>(inp0, norm, dstp0, dst_pitch); break;
    case 16: do_DecodeOverlapPlane<uint16_t, 16, false>(inp0, norm, dstp0, dst_pitch); break;
    case 32: do_DecodeOverlapPlane<float, 8 /*n/a*/, false>(inp0, norm, dstp0, dst_pitch); break;
    }
  }
}

template <class Interface>
template<typename pixel_t, int _bits_per_pixel, bool chroma>
void FFT3DEngine<Interface>::do_DecodeOverlapPlane(float *inp0, float norm, byte *dstp0, int dst_pitch)
{
  int w, h;
  int ihx, ihy;
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp0);// + (hrest/2)*dst_pitch + wrest/2; // centered
  float *inp = inp0;
  int xoffset = ep->bh*ep->bw - (ep->bw - ep->ow);
  int yoffset = ep->bw*nox*ep->bh - ep->bw*(ep->bh - ep->oh); // vertical offset of same block (overlap)
  typedef typename std::conditional<sizeof(pixel_t) == 4, float, int>::type cast_t;

  constexpr float rounder = sizeof(pixel_t) == 4 ? 0.0f : 0.5f; // v2.6

  // for float: chroma center is also 0.0
  constexpr cast_t planeBase = sizeof(pixel_t) == 4 ? cast_t(chroma ? 0.0f : 0.0f) : cast_t(chroma ? (1 << (_bits_per_pixel-1)) : 0); // anti warning

  constexpr cast_t max_pixel_value = sizeof(pixel_t) == 4 ? (pixel_t)1.0f : (pixel_t)((1 << _bits_per_pixel) - 1);

  ihy = 0; // first top big non-overlapped) part
  {
    for (h = 0; h < ep->bh - ep->oh; h++)
    {
      inp = inp0 + h*ep->bw;
      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX((cast_t)0, (cast_t)(inp[w] * norm + rounder) + planeBase));   // Copy each byte from float array to dest with windows
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle horizontal half-blocks
      {
        for (w = 0; w < ep->ow; w++)   // half line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm + rounder) + planeBase));   // overlapped Copy
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // first half line of first block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));   // Copy each byte from float array to dest with windows
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the dest image.
    }
  }

  for (ihy = 1; ihy < noy; ihy += 1) // middle vertical
  {
    for (h = 0; h < ep->oh; h++) // top overlapped part
    {
      inp = inp0 + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;

      float wsynyrh = wsynyr[h] * norm; // remove from cycle for speed
      float wsynylh = wsynyl[h] * norm;

      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder)+ planeBase));   // y overlapped
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*wsynyrh
            + (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w])*wsynylh) + rounder) + planeBase));   // x overlapped
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // double minus - half non-overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder) + planeBase));
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh) + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical non-ovelapped part
    for (h = 0; h < (ep->bh - ep->oh - ep->oh); h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh)*ep->bw + h*ep->bw + yoffset;
      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm + rounder) + planeBase));   // x overlapped
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half non-overlapped line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }

  }

  ihy = noy; // last bottom part
  {
    for (h = 0; h < ep->oh; h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (ep->bh - ep->oh)*ep->bw) + (ep->bh - ep->oh)*ep->bw + h*ep->bw;
      for (w = 0; w < ep->bw - ep->ow; w++)   // first half line of first block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));
      }
      inp += ep->bw - ep->ow;
      dstp += ep->bw - ep->ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ep->ow; w++)   // half line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm + rounder) + planeBase));   // overlapped Copy
        }
        inp += xoffset + ep->ow;
        dstp += ep->ow;
        for (w = 0; w < ep->bw - ep->ow - ep->ow; w++)   // half line of block
        {
          dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)((inp[w])*norm + rounder) + planeBase));
        }
        inp += ep->bw - ep->ow - ep->ow;
        dstp += ep->bw - ep->ow - ep->ow;
      }
      for (w = 0; w < ep->ow; w++)   // last half line of last block
      {
        dstp[w] = MIN(cast_t(max_pixel_value), MAX(0, (cast_t)(inp[w] * norm + rounder) + planeBase));
      }
      inp += ep->ow;
      dstp += ep->ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
  }
}

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

template <class Interface>
typename Interface::AFrame FFT3DEngine<Interface>::GetFrame(int n) {

  typename Interface::AFrame prev2, prev, src, next, psrc, dst, next2;
  int pxf, pyf;
  int i;
  //	char debugbuf[1536];
  //	wsprintf(debugbuf,"FFT3DFilter: n=%d \n", n);
  //	OutputDebugString(debugbuf);
  //	fftwf_complex * tmpoutrez, *tmpoutnext, *tmpoutprev, *tmpoutnext2; // store pointers
  _RPT2(0, "FFT3DFilter GetFrame, frame=%d instance_id=%d\n", n, _instance_id);
  if (reentrancy_check) {
    _RPT2(0, "FFT3DFilter GetFrame, Reentrant call detected! Frame=%d instance_id=%d\n", n, _instance_id);
    throw("cannot work in reentrant multithread mode!");
  }
  reentrancy_check = true;

  const bool plane_is_chroma = !(plane == 0 || isRGB || plane == PLANAR_Y);

  if (ep->pfactor != 0 && isPatternSet == false && ep->pshow == false) // get noise pattern
  {
    psrc = super->GetFrame(super->child, ep->pframe); // get noise patterme

    // put source bytes to float array of overlapped blocks
    load_buffer(plane, psrc, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
    FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma);
    // make FFT 2D
    fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
    if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
      FindPatternBlock(outrez, outwidth, outpitch, ep->bh, nox, noy, ep->px, ep->py, pwin, ep->degrid, gridsample);
    SetPattern(outrez, outwidth, outpitch, ep->bh, nox, noy, ep->px, ep->py, pwin, pattern2d, psigma, ep->degrid, gridsample);
    isPatternSet = true;
  }
  else if (ep->pfactor != 0 && ep->pshow == true)
  {
    // show noise pattern window
    src = super->GetFrame(super->child, n); // get noise pattern frame
    dst = super->NewVideoFrame();
    // TODO: CopyFrame(src, dst, vi, plane, env);

    // put source bytes to float array of overlapped blocks
    load_buffer(plane, src, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
    FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma);
    // make FFT 2D
    fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
    if (ep->px == 0 && ep->py == 0) // try find pattern block with minimal noise sigma
      FindPatternBlock(outrez, outwidth, outpitch, ep->bh, nox, noy, pxf, pyf, pwin, ep->degrid, gridsample);
    else
    {
      pxf = ep->px; // fixed bug in v1.6
      pyf = ep->py;
    }
    SetPattern(outrez, outwidth, outpitch, ep->bh, nox, noy, pxf, pyf, pwin, pattern2d, psigma, ep->degrid, gridsample);

    // change analysis and synthesis window to constant to show
    for (i = 0; i < ep->ow; i++)
    {
      wanxl[i] = 1;	wanxr[i] = 1;	wsynxl[i] = 1;	wsynxr[i] = 1;
    }
    for (i = 0; i < ep->oh; i++)
    {
      wanyl[i] = 1;	wanyr[i] = 1;	wsynyl[i] = 1;	wsynyr[i] = 1;
    }

    const bool plane_is_chroma2 = !isRGB;

    // put source bytes to float array of overlapped blocks
    // cur frame
    load_buffer(plane, src, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
    FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma2);
    // make FFT 2D
    fftfp.fftwf_execute_dft_r2c(plan, in, outrez);

    PutPatternOnly(outrez, outwidth, outpitch, ep->bh, nox, noy, pxf, pyf);
    // do inverse 2D FFT, get filtered 'in' array
    fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);

    // make destination frame plane from current overlaped blocks
    FFT3DEngine<Interface>::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, plane_is_chroma2);
    store_buffer(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, mirw, mirh, ep->interlaced, bits_per_pixel);
    int psigmaint = ((int)(10 * psigma)) / 10;
    int psigmadec = (int)((psigma - psigmaint) * 10);
    wsprintf(messagebuf, " frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);
    // TODO: DrawString(dst, vi, 0, 0, messagebuf);

    _RPT2(0, "FFT3DFilter GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
    reentrancy_check = false;
    return dst; // return pattern frame to show
  }

  _RPT2(0, "FFT3DFilter child GetFrame, frame=%d instance_id=%d\n", n, _instance_id);
  // Request frame 'n' from the child (source) clip.
  src = super->GetFrame(super->child, n);
  _RPT2(0, "FFT3DFilter child GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
  dst = super->NewVideoFrame();


  int btcur = ep->bt; // bt used for current frame
//	if ( (bt/2 > n) || bt==3 && n==vi.num_frames-1 )
  if ((ep->bt / 2 > n) || (ep->bt - 1) / 2 > (super->frames() - 1 - n))
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
      load_buffer(plane, src, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
      FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma);
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
       * | 4 |   o   |   o   |   o   |   o   |       |
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
          load_buffer(plane, frame, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
          FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma);
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
    FFT3DEngine<Interface>::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, plane_is_chroma);
    store_buffer(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, mirw, mirh, ep->interlaced, bits_per_pixel);
  }
  else if (ep->bt == 0) //Kalman filter
  {
    if (n == 0)
    {
      _RPT2(0, "FFT3DFilter GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
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
    load_buffer(plane, src, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
    FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma);
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
    FFT3DEngine<Interface>::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, plane_is_chroma);
    store_buffer(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, mirw, mirh, ep->interlaced, bits_per_pixel);

  }
  else if (ep->bt == -1) /// sharpen only
  {
    //		env->MakeWritable(&src);
        // put source bytes to float array of overlapped blocks
    load_buffer(plane, src, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, ep->interlaced, bits_per_pixel);
    FFT3DEngine<Interface>::InitOverlapPlane(in, coverbuf, coverpitch, plane_is_chroma);
    // make FFT 2D
    fftfp.fftwf_execute_dft_r2c(plan, in, outrez);
    ffp.Sharpen(outrez, sfp);
    // do inverse FFT 2D, get filtered 'in' array
    fftfp.fftwf_execute_dft_c2r(planinv, outrez, in);
    // make destination frame plane from current overlaped blocks
    FFT3DEngine<Interface>::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, plane_is_chroma);
    store_buffer(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, mirw, mirh, ep->interlaced, bits_per_pixel);

  }

  if (btcur == ep->bt)
  {// for normal step
    nlast = n; // set last frame to current
  }
  btcurlast = btcur;

  // As we now are finished processing the image, we return the destination image.
  _RPT2(0, "FFT3DFilter GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
  reentrancy_check = false;
  return dst;
}
