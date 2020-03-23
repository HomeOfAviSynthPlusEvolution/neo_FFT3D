#include "code_impl.h"

void ApplyWiener2D_C(
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
  // float sigmaSquaredSharpenMinNormed
  // float sigmaSquaredSharpenMaxNormed
  // float *wsharpen
  // float dehalo
  // float *wdehalo
  // float ht2n

  float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
  int h,w, block;
  float psd;
  float WienerFactor;

  if (sfp.sharpen == 0 && sfp.dehalo == 0)// no sharpen, no dehalo
  {
    for (block =0; block <sfp.howmanyblocks; block++)
    {
      for (h=0; h<sfp.bh; h++) // middle
      {
        for (w=0; w<sfp.outwidth; w++) // not skip first v.1.2
        {
          psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
          WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
          outcur[w][0] *= WienerFactor; // apply filter on real part
          outcur[w][1] *= WienerFactor; // apply filter on imaginary part
        }
        outcur += sfp.outpitch;
      }
    }
  }
  else if (sfp.sharpen != 0 && sfp.dehalo==0) // sharpen
  {
    for (block =0; block <sfp.howmanyblocks; block++)
    {
      for (h=0; h<sfp.bh; h++) // middle
      {
        for (w=0; w<sfp.outwidth; w++) // not skip first
        {
          psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
          WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
          WienerFactor *= 1 + sfp.sharpen*sfp.wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) ); // sharpen factor - changed in v.1.1
          outcur[w][0] *= WienerFactor; // apply filter on real part
          outcur[w][1] *= WienerFactor; // apply filter on imaginary part
        }
        outcur += sfp.outpitch;
        sfp.wsharpen += sfp.outpitch;
      }
      sfp.wsharpen -= sfp.outpitch*sfp.bh;
    }
  }
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
  {
    for (block =0; block <sfp.howmanyblocks; block++)
    {
      for (h=0; h<sfp.bh; h++) // middle
      {
        for (w=0; w<sfp.outwidth; w++) // not skip first
        {
          psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
          WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
          WienerFactor *= (psd + sfp.ht2n)/((psd + sfp.ht2n) + sfp.dehalo*sfp.wdehalo[w] * psd );
          outcur[w][0] *= WienerFactor; // apply filter on real part
          outcur[w][1] *= WienerFactor; // apply filter on imaginary part
        }
        outcur += sfp.outpitch;
        sfp.wdehalo += sfp.outpitch;
      }
      sfp.wdehalo -= sfp.outpitch*sfp.bh;
    }
  }
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
  {
    for (block =0; block <sfp.howmanyblocks; block++)
    {
      for (h=0; h<sfp.bh; h++) // middle
      {
        for (w=0; w<sfp.outwidth; w++) // not skip first
        {
          psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
          WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
          WienerFactor *= 1 + sfp.sharpen*sfp.wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) ) *
            (psd + sfp.ht2n)/((psd + sfp.ht2n) + sfp.dehalo*sfp.wdehalo[w] * psd );
          outcur[w][0] *= WienerFactor; // apply filter on real part
          outcur[w][1] *= WienerFactor; // apply filter on imaginary part
        }
        outcur += sfp.outpitch;
        sfp.wsharpen += sfp.outpitch;
        sfp.wdehalo += sfp.outpitch;
      }
      sfp.wsharpen -= sfp.outpitch*sfp.bh;
      sfp.wdehalo -= sfp.outpitch*sfp.bh;
    }
  }
}
