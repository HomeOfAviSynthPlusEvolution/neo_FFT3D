#include "code_impl_SSE2.h"

// true-true:
// bt=3 sharpen=1 dehalo=1
//       new       old
// x86  14.05      13.91 fps
// x64  14.60      12.91
// 170531 moved to simd
template<bool do_sharpen, bool do_dehalo>
static void Sharpen_degrid_SSE2_impl(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float sharpen
  // float sigmaSquaredSharpenMin
  // float sigmaSquaredSharpenMax
  // float *wsharpen
  // float degrid
  // fftwf_complex *gridsample
  // float dehalo
  // float *wdehalo
  // float ht2n

  int bytesperblock = sfp.bh*sfp.outpitch * 8;

  if (!do_sharpen && !do_dehalo)
    return;

  for (int block = 0; block < sfp.howmanyblocks; block++) // blockscounter is bytesperblock
  {
    const float gridfraction1 = sfp.degrid*outcur[0][0] / sfp.gridsample[0][0];
    auto gridfraction_ps = _mm_load1_ps(&gridfraction1);

    for (int w = 0; w < bytesperblock; w += 16)
    {
      auto outcur_ps = _mm_loadu_ps((const float *)((byte *)(outcur)+w));
      auto gridsample_ps = _mm_loadu_ps((const float *)((byte *)(sfp.gridsample)+w));
      auto gridcorrection_ps = _mm_mul_ps(gridfraction_ps, gridsample_ps);
      auto diff_ourcur_gridcorr_reim = _mm_sub_ps(outcur_ps, gridcorrection_ps);
      auto reim = diff_ourcur_gridcorr_reim;
      /*
      float gridcorrection0 = gridfraction*gridsample[w][0];
      float re = outcur[w][0] - gridcorrection0;
      float gridcorrection1 = gridfraction*gridsample[w][1];
      float im = outcur[w][1] - gridcorrection1;
      */
      auto mul1 = _mm_mul_ps(reim, reim); // sumre*sumre | sumim*sumim
      auto sum = _mm_add_ps(mul1, _mm_shuffle_ps(mul1, mul1, 128 + 48 + 0 + 1)); // 10 11 00 01 low - swap re & im
      auto smallf = _mm_set1_ps(1e-15f);
      auto psd = _mm_add_ps(sum, smallf); // psd of sum = sumre*sumre + sumim*sumim + smallf
      //psd = (re*re + im*im) + 1e-15f;// power spectrum density

      __m128 sfact;
      /*
      if (sharpen != 0 && dehalo == 0) {
        sfact = (1 + sharpen*wsharpen[w] * sqrt(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))));
      }
      else if (sharpen == 0 && dehalo != 0) {
        sfact = (psd + ht2n) / ((psd + ht2n) + dehalo*wdehalo[w] * psd);
      }
      else if (sharpen != 0 || dehalo != 0) {
        sfact = (1 + sharpen*wsharpen[w] * sqrt(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)))) *
        (psd + ht2n) / ((psd + ht2n) + dehalo*wdehalo[w] * psd);
        // (1 + x) * n = n + x * n
      */
      __m128 res1_sharpen;
      __m128 res2_dehalo;

      if (do_sharpen) {
      // take only two elements from dehalo and wsharpen
        auto wsharpen_ps = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)((byte *)(sfp.wsharpen)+(w >> 1)))); // two floats
        wsharpen_ps = _mm_shuffle_ps(wsharpen_ps, wsharpen_ps, (64 + 16 + 0 + 0)); // 01 01 00 00 low
        auto smax_ps = _mm_load1_ps(&sfp.sigmaSquaredSharpenMaxNormed);
        auto smin_ps = _mm_load1_ps(&sfp.sigmaSquaredSharpenMinNormed);
        auto sharpen_ps = _mm_load1_ps(&sfp.sharpen);

        auto psd_plus_min = _mm_add_ps(psd, smin_ps);
        auto psd_plus_max = _mm_add_ps(psd, smax_ps);
        auto one_per_mulminmax = _mm_rcp_ps(_mm_mul_ps(psd_plus_min, psd_plus_max));
        auto tmp = _mm_sqrt_ps(_mm_mul_ps(_mm_mul_ps(one_per_mulminmax, psd), smax_ps));
        res1_sharpen = _mm_mul_ps(_mm_mul_ps(tmp, sharpen_ps), wsharpen_ps);
      }

      if (do_dehalo) {
        auto dehalo_ps = _mm_load1_ps(&sfp.dehalo);
        auto ht2n_ps = _mm_load1_ps(&sfp.ht2n);
        auto wdehalo_ps = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)((byte *)(sfp.wdehalo)+(w >> 1)))); // two floats
        wdehalo_ps = _mm_shuffle_ps(wdehalo_ps, wdehalo_ps, (64 + 16 + 0 + 0)); // 01 01 00 00 low

        auto dehalo_mul = _mm_mul_ps(_mm_mul_ps(dehalo_ps, wdehalo_ps), psd);
        auto psd_plus_ht2n = _mm_add_ps(psd, ht2n_ps);
        auto one_per_sum = _mm_rcp_ps(_mm_add_ps(dehalo_mul, psd_plus_ht2n));
        res2_dehalo = _mm_mul_ps(one_per_sum, psd_plus_ht2n);
      }

      if (do_sharpen && do_dehalo)
        sfact = _mm_add_ps(res2_dehalo, _mm_mul_ps(res1_sharpen, res2_dehalo));
      else if (do_sharpen) {
        const float one = 1.0f;
        sfact = _mm_add_ps(res1_sharpen, _mm_load1_ps(&one)); // 1 + ()
      }
      else if (do_dehalo) {
        sfact = res2_dehalo;
      }
      reim = _mm_mul_ps(reim, sfact);
      //re *= sfact; // apply filter on real part
      //im *= sfact; // apply filter on imaginary part
      auto result = _mm_add_ps(reim, gridcorrection_ps);
      //outcur[w][0] = re + gridcorrection0;
      //outcur[w][1] = im + gridcorrection1;
      _mm_storeu_ps((float *)((byte *)(outcur)+w), result);
    }
    outcur += sfp.bh*sfp.outpitch;
  }
}

void Sharpen_degrid_SSE2(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_degrid_SSE2_impl<true, false>(outcur, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_degrid_SSE2_impl<false, true>(outcur, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_degrid_SSE2_impl<true, true>(outcur, sfp);
}
