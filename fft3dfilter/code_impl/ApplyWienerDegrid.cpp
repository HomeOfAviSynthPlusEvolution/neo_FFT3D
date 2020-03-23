#include "code_impl.h"

template <bool sharpen, bool dehalo>
void ApplyWiener2D_degrid_C_impl(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float sigmaSquaredNoiseNormed
  // float beta
  // float sharpen
  // float sigmaSquaredSharpenMin
  // float sigmaSquaredSharpenMax
  // float *wsharpen
  // float degrid
  // fftwf_complex *gridsample
  // float dehalo
  // float *wdehalo
  // float ht2n

  // this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
  float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
  int h,w, block;
  float psd;
  float WienerFactor;
  float gridfraction;

  float *wsharpen;
  float *wdehalo;
  fftwf_complex *gridsample;

  for (block =0; block <sfp.howmanyblocks; block++)
  {
    wsharpen = sfp.wsharpen;
    wdehalo = sfp.wdehalo;
    gridsample = sfp.gridsample;

    gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
    for (h=0; h<sfp.bh; h++) // middle
    {
      for (w=0; w<sfp.outwidth; w++) // not skip first
      {
        float gridcorrection0 = gridfraction*gridsample[w][0];
        float corrected0 = outcur[w][0] - gridcorrection0;
        float gridcorrection1 = gridfraction*gridsample[w][1];
        float corrected1 = outcur[w][1] - gridcorrection1;
        psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        if (sharpen && !dehalo)
          WienerFactor *= 1 + sfp.sharpen*wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) ); // sharpen factor - changed in v.1.1
        else if (!sharpen && dehalo)
          WienerFactor *= (psd + sfp.ht2n)/((psd + sfp.ht2n) + sfp.dehalo*wdehalo[w] * psd );
        else if (sharpen && dehalo)
          WienerFactor *= 1 + sfp.sharpen*wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) ) * (psd + sfp.ht2n)/((psd + sfp.ht2n) + sfp.dehalo*wdehalo[w] * psd );
        corrected0 *= WienerFactor; // apply filter on real part
        corrected1 *= WienerFactor; // apply filter on imaginary part
        outcur[w][0] = corrected0 + gridcorrection0;
        outcur[w][1] = corrected1 + gridcorrection1;
      }
      outcur += sfp.outpitch;
      wsharpen += sfp.outpitch;
      gridsample += sfp.outpitch;
      wdehalo += sfp.outpitch;
    }
  }
}

void ApplyWiener2D_degrid_C(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    ApplyWiener2D_degrid_C_impl<false, false>(outcur, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    ApplyWiener2D_degrid_C_impl<true, false>(outcur, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    ApplyWiener2D_degrid_C_impl<false, true>(outcur, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    ApplyWiener2D_degrid_C_impl<true, true>(outcur, sfp);
}
