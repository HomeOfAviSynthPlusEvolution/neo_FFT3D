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

void ApplyWiener3D2_C(
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

  // this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
  // return result in outprev
  float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
  float psd;
  float WienerFactor;
  float f3d0r, f3d1r, f3d0i, f3d1i;
  int block;
  int h,w;

  for (block=0; block <sfp.howmanyblocks; block++)
  {
    for (h=0; h<sfp.bh; h++)
    {
      for (w=0; w<sfp.outwidth; w++)
      {
        // dft 3d (very short - 2 points)
        f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
        f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
        f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
        f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
        psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        f3d0r *= WienerFactor; // apply filter on real part
        f3d0i *= WienerFactor; // apply filter on imaginary part
        psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        f3d1r *= WienerFactor; // apply filter on real part
        f3d1i *= WienerFactor; // apply filter on imaginary part
        // reverse dft for 2 points
        outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real part
        outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
        // Attention! return filtered "out" in "outprev" to preserve "out" for next step
      }
      outcur += sfp.outpitch;
      outprev += sfp.outpitch;
    }
  }
}

void ApplyWiener3D3_C(
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

  // this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
  // return result in outprev
  float fcr, fci, fpr, fpi, fnr, fni;
  float WienerFactor;
  float psd;
  float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
  float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

  int block;
  int h,w;

  for (block=0; block <sfp.howmanyblocks; block++)
  {
    for (h=0; h<sfp.bh; h++) // first half
    {
      for (w=0; w<sfp.outwidth; w++) //
      {
        // dft 3d (very short - 3 points)
        float pnr = outprev[w][0] + outnext[w][0];
        float pni = outprev[w][1] + outnext[w][1];
        fcr = outcur[w][0] + pnr; // real cur
        fci = outcur[w][1] + pni; // im cur
        float di = sin120*(outprev[w][1]-outnext[w][1]);
        float dr = sin120*(outnext[w][0]-outprev[w][0]);
        fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
        fnr = fpr - di - di; //v1.8.1
        fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
        fni = fpi - dr - dr; //v1.8.1
        psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fcr *= WienerFactor; // apply filter on real part
        fci *= WienerFactor; // apply filter on imaginary part
        psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fpr *= WienerFactor; // apply filter on real part
        fpi *= WienerFactor; // apply filter on imaginary part
        psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fnr *= WienerFactor; // apply filter on real part
        fni *= WienerFactor; // apply filter on imaginary part
        // reverse dft for 3 points
        outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real part
        outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
        // Attention! return filtered "out" in "outprev" to preserve "out" for next step
      }
      outcur += sfp.outpitch;
      outprev += sfp.outpitch;
      outnext += sfp.outpitch;
    }
  }
}

void ApplyWiener3D4_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev2,
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
  // dft with 4 points
  // this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
  // return result in outprev
  float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
  float WienerFactor;
  float psd;
  float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0

  int block;
  int h,w;

  for (block=0; block <sfp.howmanyblocks; block++)
  {
    for (h=0; h<sfp.bh; h++) // first half
    {
      for (w=0; w<sfp.outwidth; w++) //
      {
        // dft 3d (very short - 4 points)
        fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
        fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
        fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
        fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
        fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
        fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
        fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
        fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur

        psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fp2r *= WienerFactor; // apply filter on real part
        fp2i *= WienerFactor; // apply filter on imaginary part

        psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fpr *= WienerFactor; // apply filter on real part
        fpi *= WienerFactor; // apply filter on imaginary part

        psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fcr *= WienerFactor; // apply filter on real part
        fci *= WienerFactor; // apply filter on imaginary part

        psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fnr *= WienerFactor; // apply filter on real part
        fni *= WienerFactor; // apply filter on imaginary part

        // reverse dft for 4 points
        outprev2[w][0] = (fp2r + fpr + fcr + fnr)*0.25f; // get  real part
        outprev2[w][1] = (fp2i + fpi + fci + fni)*0.25f; // get imaginary part
        // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
      }
      outcur += sfp.outpitch;
      outprev2 += sfp.outpitch;
      outprev += sfp.outpitch;
      outnext += sfp.outpitch;
    }
  }
}

void ApplyWiener3D5_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev2,
  fftwf_complex *outprev,
  fftwf_complex *outnext,
  fftwf_complex *outnext2,
  SharedFunctionParams sfp)

{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float sigmaSquaredNoiseNormed
  // float beta

  // dft with 5 points
  // return result in outprev2
  float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i;
  float WienerFactor;
  float psd;
  float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
  float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
  float cos72 = 0.30901699437494742410229341718282f;
  float sin144 = 0.58778525229247312916870595463907f;
  float cos144 = -0.80901699437494742410229341718282f;

  int block;
  int h,w;

  for (block=0; block <sfp.howmanyblocks; block++)
  {
    for (h=0; h<sfp.bh; h++) // first half
    {
      for (w=0; w<sfp.outwidth; w++) //
      {
        // dft 3d (very short - 5 points)
        float sum = (outprev2[w][0] + outnext2[w][0])*cos72 + (outprev[w][0] + outnext[w][0])*cos144 + + outcur[w][0];
        float dif = (- outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1]  - outnext[w][1])*sin144;
        fp2r = sum + dif; // real prev2
        fn2r = sum - dif; // real next2
        sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
        dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (- outprev[w][0] + outnext[w][0])*sin144;
        fp2i = sum + dif; // im prev2
        fn2i = sum - dif; // im next2
        sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
        dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
        fpr = sum + dif; // real prev
        fnr = sum - dif; // real next
        sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
        dif =  (- outprev2[w][0] + outnext2[w][0])*sin144 + (- outprev[w][0] + outnext[w][0])*sin72;
        fpi = sum + dif; // im prev
        fni = sum - dif; // im next
        fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
        fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur

        psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fp2r *= WienerFactor; // apply filter on real part
        fp2i *= WienerFactor; // apply filter on imaginary part

        psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fpr *= WienerFactor; // apply filter on real part
        fpi *= WienerFactor; // apply filter on imaginary part

        psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fcr *= WienerFactor; // apply filter on real part
        fci *= WienerFactor; // apply filter on imaginary part

        psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fnr *= WienerFactor; // apply filter on real part
        fni *= WienerFactor; // apply filter on imaginary part

        psd = fn2r*fn2r + fn2i*fn2i + 1e-15f; // power spectrum density next2
        WienerFactor = MAX((psd - sfp.sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
        fn2r *= WienerFactor; // apply filter on real part
        fn2i *= WienerFactor; // apply filter on imaginary part

        // reverse dft for 5 points
        outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r)*0.2f ; // get  real part
        outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i)*0.2f; // get imaginary part
        // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
      }
      outcur += sfp.outpitch;
      outprev2 += sfp.outpitch;
      outprev += sfp.outpitch;
      outnext += sfp.outpitch;
      outnext2 += sfp.outpitch;
    }
  }
}
