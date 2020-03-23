#include "code_impl_SSE2.h"

// bt=3
void ApplyWiener3D3_degrid_SSE2(
  fftwf_complex *outcur,
  fftwf_complex *outprev,
  fftwf_complex *outnext,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float sigmaSquaredNoiseNormed
  // float beta
  // float degrid
  // fftwf_complex *gridsample

  // dft 3d (very short - 3 points)
  // optimized for SSE assembler
  // return result in outprev
  //  float fcr, fci, fpr, fpi, fnr, fni;
  //  float pnr, pni, di, dr;
  //  float WienerFactor =1;
  //  float psd;
  __m128 lowlimit = _mm_set1_ps((sfp.beta - 1) / sfp.beta); //     (beta-1)/beta>=0
  __m128 sin120 = _mm_set1_ps(0.86602540378443864676372317075294f); //sqrtf(3.0f)*0.5f;
  __m128 smallf = _mm_set1_ps(1e-15f);
  __m128 onethird = _mm_set1_ps(0.33333333333f);
  __m128 onehalf = _mm_set1_ps(0.5f);

  //__m128 gridcorrection;
  float gridfraction;

  //  int block;
  //  int h,w;

  //  int totalbytes = howmanyblocks*bh*outpitch*8;
  int bytesperblock = sfp.bh*sfp.outpitch * 8;
  int blockscounter = sfp.howmanyblocks;

  //  for (line=0; line <totalnumber; line++)
  //  {
  int ecx_bytesperblock = bytesperblock; //  mov ecx, bytesperblock; // counter

  byte *pOutcur = (byte *)outcur; //  mov esi, outcur; // current
  byte *pOutPrev = (byte *)outprev; //  mov edi, outprev;
  byte *pOutNext = (byte *)outnext; //  mov edx, outnext;

  __m128 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

  for (int block = 0; block < sfp.howmanyblocks; block++) { // for (block=0; block <howmanyblocks; block++)
    // Orig_C: float gridfraction = degrid*outcur[0][0] / gridsample[0][0];
    byte *pGridSample = (byte *)sfp.gridsample;

    __m128 outCur00 = _mm_load_ps((float *)pOutcur); // cur real | img
    xmm7 = _mm_mul_ps(outCur00, _mm_load1_ps(&sfp.degrid));
    __m128 reciprGridSample = _mm_rcp_ps(_mm_load1_ps((float *)pGridSample)); // rcpps xmm3, xmm3;
    gridfraction = _mm_cvtss_f32(_mm_mul_ps(xmm7, reciprGridSample)); // movss gridfraction, xmm7;

    for (int eax = 0; eax < ecx_bytesperblock; eax += 16) {

      //Orig_C: float gridcorrection0_3 = gridfraction*gridsample[w][0] * 3;
      //        float gridcorrection1_3 = gridfraction*gridsample[w][1] * 3;
      xmm3 = _mm_load_ps((float *)(pGridSample + eax)); // movaps xmm3, [ebx + n]; // mm3=grid real | img
      xmm7 = _mm_load1_ps(&gridfraction);
      xmm3 = _mm_mul_ps(xmm3, xmm7); // mulps xmm3, xmm7; // fraction*sample
      __m128 gridcorrection = _mm_add_ps(_mm_add_ps(xmm3, xmm3), xmm3);;

      __m128 prev = _mm_load_ps((float *)(pOutPrev + eax)); // xmm0=prev real | img
      __m128 next = _mm_load_ps((float *)(pOutNext + eax)); // xmm1=next real | img
      __m128 current = _mm_load_ps((float *)(pOutcur + eax)); // mm3=cur real | img

      //pnr = outprev[w][0] + outnext[w][0];
      //pni = outprev[w][1] + outnext[w][1];
      __m128 pn_r_i = _mm_add_ps(prev, next);

      //fcr = outcur[w][0] + pnr; // real cur
      //fci = outcur[w][1] + pni; // im cur
      __m128 fc_r_i = _mm_sub_ps(_mm_add_ps(current, pn_r_i), gridcorrection); // - gridcorrection

      xmm2 = _mm_mul_ps(pn_r_i, onehalf); // 0.5*pnr | 0.5*pni

      // psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
      xmm5 = _mm_mul_ps(fc_r_i, fc_r_i); // fcr*fcr | fci*fci
      xmm5 = _mm_add_ps(xmm5, _mm_shuffle_ps(xmm5, xmm5, 128 + 48 + 0 + 1)); // 10 11 00 01 low - swap re & im
      /*
    movaps xmm6, xmm5; //copy
    shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
    addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
    */
      __m128 psd = _mm_add_ps(xmm5, smallf); // + 1e-15f, xmm5 =psd cur
      xmm6 = _mm_sub_ps(psd, _mm_load1_ps(&sfp.sigmaSquaredNoiseNormed)); // psd - sigma
      xmm5 = _mm_mul_ps(_mm_rcp_ps(psd), xmm6); // (psd-sigma) * 1/psd
      __m128 WienerFactor = _mm_max_ps(xmm5, lowlimit); // WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter

      // fcr *= WienerFactor; // apply filter on real part
      // fci *= WienerFactor; // apply filter on imaginary part
      fc_r_i = _mm_mul_ps(fc_r_i, WienerFactor); // final wiener fcr | fci, *= WienerFactor; // apply filter on real and im part

      xmm5 = _mm_sub_ps(next, prev); // // next-prev   real | img
      xmm6 = _mm_sub_ps(prev, next); // prev-next  real | img

      xmm5 = _mm_shuffle_ps(xmm5, xmm5, 128 + 0 + 8 + 0); // shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
      xmm6 = _mm_shuffle_ps(xmm6, xmm6, 192 + 16 + 12 + 1); // shufps xmm6, xmm6, (192 + 16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
      xmm6 = _mm_unpacklo_ps(xmm6, xmm5); //unpcklps xmm6, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)

      xmm6 = _mm_mul_ps(xmm6, sin120); // di | dr
      // di = sin120*(outprev[w][1]-outnext[w][1]);
      // dr = sin120*(outnext[w][0]-outprev[w][0]);

      xmm3 = _mm_sub_ps(current, xmm2); // cur-0.5*pn
      xmm1 = xmm3; // copy

      // fpr = out[w][0] - 0.5f*pnr + di; // real prev
      // fpi = out[w][1] - 0.5f*pni + dr; // im prev
      xmm3 = _mm_add_ps(xmm3, xmm6); // fpr | fpi

      // fnr = out[w][0] - 0.5f*pnr - di; // real next
      // fni = out[w][1] - 0.5f*pni - dr ; // im next
      xmm1 = _mm_sub_ps(xmm1, xmm6); //  fnr | fni

      xmm5 = _mm_mul_ps(xmm3, xmm3); // fpr*fpr | fpi*fpi

      xmm5 = _mm_add_ps(xmm5, _mm_shuffle_ps(xmm5, xmm5, 128 + 48 + 0 + 1)); // //  10 11 00 01 low// swap re & im
      psd = _mm_add_ps(xmm5, smallf); // psd cur = sumre*sumre + sumim*sumim + smallf
      // psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur

      __m128 sigmaSquaredNoiseNormed_m128 = _mm_load1_ps(&sfp.sigmaSquaredNoiseNormed);
      xmm6 = _mm_sub_ps(psd, sigmaSquaredNoiseNormed_m128); // psd - sigma
      xmm5 = _mm_mul_ps(_mm_rcp_ps(psd), xmm6); // (psd-sigma)/psd
      xmm5 = _mm_max_ps(xmm5, lowlimit);
      // WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter

      // fpr *= WienerFactor; // apply filter on real part
      // fpi *= WienerFactor; // apply filter on imaginary part
      xmm3 = _mm_mul_ps(xmm3, xmm5); // final wiener fpr | fpi

      xmm5 = _mm_mul_ps(xmm1, xmm1); // fnr*fnr | fni*fni

      xmm5 = _mm_add_ps(xmm5, _mm_shuffle_ps(xmm5, xmm5, 128 + 48 + 0 + 1)); // 10 11 00 01 low// swap re & im, sumre*sumre + sumim*sumim

      // psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
      xmm5 = _mm_add_ps(xmm5, smallf); // xmm5 =psd cur

      xmm6 = _mm_sub_ps(xmm5, sigmaSquaredNoiseNormed_m128); // psd - sigma
      xmm5 = _mm_mul_ps(_mm_rcp_ps(xmm5), xmm6); // (psd-sigma)/psd

      xmm5 = _mm_max_ps(xmm5, lowlimit); // wienerfactor
      // WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
      // fnr *= WienerFactor; // apply filter on real part
      // fni *= WienerFactor; // apply filter on imaginary part
      xmm1 = _mm_mul_ps(xmm1, xmm5); // final wiener fmr | fmi

      xmm4 = _mm_add_ps(_mm_add_ps(fc_r_i, xmm3), xmm1); // fc + fp + fn
      // reverse dft for 3 points

      xmm4 = _mm_add_ps(xmm4, gridcorrection);

      xmm4 = _mm_mul_ps(xmm4, onethird);
      // outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real part
      // outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part

      _mm_store_ps((float *)(pOutPrev + eax), xmm4); // movaps[edi + n], xmm4; // write output to prev array
                               // Attention! return filtered "out" in "outprev" to preserve "out" for next step
    }
    pOutNext += ecx_bytesperblock;
    pOutPrev += ecx_bytesperblock;
    pOutcur += ecx_bytesperblock;
  }
}
