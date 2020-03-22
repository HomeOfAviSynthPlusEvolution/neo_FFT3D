#include "code_impl.h"
#include <avs/config.h> // x64
#include <emmintrin.h>

// PF 170302 simd, SSE2 x64 C -> x64 simd: 11.37 -> 13.26
void ApplyWiener3D2_SSE2(
  fftwf_complex *outcur,
  fftwf_complex *outprev,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float sigmaSquaredNoiseNormed
  // float beta
  // return result in outprev

  int totalbytes = sfp.howmanyblocks*sfp.bh*sfp.outpitch * 8;

  //  for (block=0; block <howmanyblocks; block++)
  //  {
  //    for (h=0; h<bh; h++)  
  //    {
  //      for (w=0; w<outwidth; w++) 
  //      {

  __m128 lowlimit = _mm_set1_ps((sfp.beta - 1) / sfp.beta); //     (beta-1)/beta>=0
  __m128 smallf = _mm_set1_ps(1e-15f);
  __m128 onehalf = _mm_set1_ps(0.5f);
  __m128 sigma = _mm_set1_ps(sfp.sigmaSquaredNoiseNormed);

  __m128 xmm1, xmm2, xmm3;
  for (int n = 0; n < totalbytes; n += 16) {
    // take two complex numbers
    __m128 prev = _mm_load_ps(reinterpret_cast<float *>((uint8_t *)outprev + n)); // prev real | img
    __m128 cur = _mm_load_ps(reinterpret_cast<float *>((uint8_t *)outcur + n)); // cur real | img 

    // f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
    // f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
    __m128 sum = _mm_add_ps(cur, prev);

    // f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
    // f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
    __m128 diff = _mm_sub_ps(cur, prev);

    // Part#1 - sum
    xmm1 = _mm_mul_ps(sum, sum); // sumre*sumre | sumin*sumim
    xmm3 = _mm_shuffle_ps(xmm1, xmm1, (1 + 0 + 48 + 128)); // swap re & im
    xmm1 = _mm_add_ps(xmm1, xmm3); // sumre*sumre + sumim*sumim
    xmm1 = _mm_add_ps(xmm1, smallf); // psd of sum = sumre*sumre + sumim*sumim + smallf
    // psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0

    xmm3 = _mm_sub_ps(xmm1, sigma); // psd - sigma
    xmm1 = _mm_rcp_ps(xmm1); // 1/psd
    xmm3 = _mm_mul_ps(xmm3, xmm1); // (psd-sigma)/psd

    // WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
    xmm3 = _mm_max_ps(xmm3, lowlimit); // wienerfactor
    // f3d0r *= WienerFactor; // apply filter on real part
    // f3d0i *= WienerFactor; // apply filter on imaginary part
    __m128 sum_res = _mm_mul_ps(sum, xmm3); // final wiener sum f3d0

    // Part#2 - diff
    xmm1 = _mm_mul_ps(diff, diff); // difre*difre | difim*difim
    xmm3 = _mm_shuffle_ps(xmm1, xmm1, (1 + 0 + 48 + 128)); // swap re & im
    xmm1 = _mm_add_ps(xmm1, xmm3);
    xmm1 = _mm_add_ps(xmm1, smallf); // psd of dif
    // psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1

    xmm3 = _mm_sub_ps(xmm1, sigma); // psd - sigma
    xmm1 = _mm_rcp_ps(xmm1); // 1 / psd
    xmm3 = _mm_mul_ps(xmm3, xmm1); // (psd-sigma)/psd

    // WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
    xmm3 = _mm_max_ps(xmm3, lowlimit); // wienerfactor

    // f3d1r *= WienerFactor; // apply filter on real part
    // f3d1i *= WienerFactor; // apply filter on imaginary part
    __m128 diff_res = _mm_mul_ps(diff, xmm3); // final wiener dif f3d1

    // Part#3 - finalize
    // reverse dft for 2 points
    xmm2 = _mm_add_ps(sum_res, diff_res); // filterd sum + dif

    // outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get real part
    // outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
    xmm2 = _mm_mul_ps(xmm2, onehalf); // filterd(sum + dif)*0.5
    _mm_store_ps(reinterpret_cast<float *>((byte *)outprev + n), xmm2);
    // Attention! return filtered "outcur" in "outprev" to preserve "outcur" for next step
  }
}
