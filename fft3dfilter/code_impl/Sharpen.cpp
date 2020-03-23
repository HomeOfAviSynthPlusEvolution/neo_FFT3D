#include "code_impl.h"

template <bool sharpen, bool dehalo>
void Sharpen_C_impl(
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
  // float dehalo
  // float *wdehalo
  // float ht2n

  int h,w, block;
  float psd;
  float sfact;

  float *wsharpen;
  float *wdehalo;

  for (block =0; block <sfp.howmanyblocks; block++)
  {
    wsharpen = sfp.wsharpen;
    wdehalo = sfp.wdehalo;
    for (h=0; h<sfp.bh; h++) // middle
    {
      for (w=0; w<sfp.outwidth; w++)
      {
        psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
        //improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
        if (sharpen && !dehalo)
          sfact = (1 + sfp.sharpen*wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) ) ) ;
        else if (!sharpen && dehalo)
          sfact = (psd + sfp.ht2n)/((psd + sfp.ht2n) + dehalo*wdehalo[w] * psd );
        else if (sharpen && dehalo)
          sfact = (1 + sfp.sharpen*wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) ) ) *
          (psd + sfp.ht2n) / ((psd + sfp.ht2n) + dehalo*wdehalo[w] * psd );
        outcur[w][0] *= sfact;
        outcur[w][1] *= sfact;
      }
      outcur += sfp.outpitch;
      wsharpen += sfp.outpitch;
      wdehalo += sfp.outpitch;
    }
  }
}

void Sharpen_C(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_C_impl<true, false>(outcur, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_C_impl<false, true>(outcur, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_C_impl<true, true>(outcur, sfp);
}

template <bool sharpen, bool dehalo>
void Sharpen_degrid_C_impl(
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

  int h,w, block;
  float psd;
  float sfact;

  float *wsharpen;
  float *wdehalo;
  fftwf_complex *gridsample;

  for (block =0; block <sfp.howmanyblocks; block++)
  {
    wsharpen = sfp.wsharpen;
    wdehalo = sfp.wdehalo;
    gridsample = sfp.gridsample;

    float gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
    for (h=0; h<sfp.bh; h++) // middle
    {
      for (w=0; w<sfp.outwidth; w++)
      {
        float gridcorrection0 = gridfraction*gridsample[w][0];
        float re = outcur[w][0] - gridcorrection0;
        float gridcorrection1 = gridfraction*gridsample[w][1];
        float im = outcur[w][1] - gridcorrection1;
        psd = (re*re + im*im) + 1e-15f;// power spectrum density
        //improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
        if (sharpen && !dehalo)
          sfact = (1 + sfp.sharpen*wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) )) ;
        else if (!sharpen && dehalo)
          sfact = (psd + sfp.ht2n) / ((psd + sfp.ht2n) + dehalo*wdehalo[w] * psd );
        else if (sharpen && dehalo)
          sfact = (1 + sfp.sharpen*wsharpen[w]*sqrt( psd*sfp.sigmaSquaredSharpenMaxNormed/((psd + sfp.sigmaSquaredSharpenMinNormed)*(psd + sfp.sigmaSquaredSharpenMaxNormed)) )) *
          (psd + sfp.ht2n)/((psd + sfp.ht2n) + dehalo*wdehalo[w] * psd );
        re *= sfact; // apply filter on real part
        im *= sfact; // apply filter on imaginary part
        outcur[w][0] = re + gridcorrection0;
        outcur[w][1] = im + gridcorrection1;
      }
      outcur += sfp.outpitch;
      wsharpen += sfp.outpitch;
      wdehalo += sfp.outpitch;
      gridsample += sfp.outpitch;
    }
  }
}

void Sharpen_degrid_C(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_degrid_C_impl<true, false>(outcur, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_degrid_C_impl<false, true>(outcur, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_degrid_C_impl<true, true>(outcur, sfp);
}
