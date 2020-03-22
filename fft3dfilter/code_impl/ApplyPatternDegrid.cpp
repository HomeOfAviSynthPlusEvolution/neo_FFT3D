#include "code_impl.h"

void ApplyPattern2D_degrid_C(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float pfactor
  // float *pattern2d
  // float beta,
  // float degrid
  // fftwf_complex *gridsample

	int h,w, block;
	float psd;
	float WienerFactor;
	float *pattern2d;
  fftwf_complex *gridsample;
	float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0

	if (sfp.pfactor == 0)
    return;

  for (block =0; block <sfp.howmanyblocks; block++)
  {
    pattern2d = sfp.pattern2d;
    gridsample = sfp.gridsample;
    float gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
    for (h=0; h<sfp.bh; h++) // middle
    {
      for (w=0; w<sfp.outwidth; w++) 
      {
        float gridcorrection0 = gridfraction*gridsample[w][0];
        float corrected0 = outcur[w][0] - gridcorrection0;
        float gridcorrection1 = gridfraction*gridsample[w][1];
        float corrected1 = outcur[w][1] - gridcorrection1;
        psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
        WienerFactor = MAX((psd - sfp.pfactor*pattern2d[w])/psd, lowlimit); // limited Wiener filter
        corrected0 *= WienerFactor; // apply filter on real part
        corrected1 *= WienerFactor; // apply filter on imaginary part
        outcur[w][0] = corrected0 + gridcorrection0;
        outcur[w][1] = corrected1 + gridcorrection1;
      }
      outcur += sfp.outpitch;
      pattern2d += sfp.outpitch;
      gridsample += sfp.outpitch;
    }
  }
}

void ApplyPattern3D2_degrid_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float *pattern3d
  // float beta,
  // float degrid
  // fftwf_complex *gridsample

	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float psd;
	float *pattern3d;
  fftwf_complex *gridsample;
	float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
	float WienerFactor;
	float f3d0r, f3d1r, f3d0i, f3d1i;
	int block;
	int h,w;

	for (block=0; block <sfp.howmanyblocks; block++)
	{
    pattern3d = sfp.pattern3d;
    gridsample = sfp.gridsample;
		float gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<sfp.bh; h++)  
		{
			for (w=0; w<sfp.outwidth; w++) 
			{
				float gridcorrection0_2 = gridfraction*gridsample[w][0]*2; // grid correction
				float gridcorrection1_2 = gridfraction*gridsample[w][1]*2;
				// dft 3d (very short - 2 points)
				f3d0r =  outcur[w][0] + outprev[w][0] - gridcorrection0_2; // real 0 (sum)
				f3d0i =  outcur[w][1] + outprev[w][1] - gridcorrection1_2; // im 0 (sum)
				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real part
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real part
				f3d1i *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r + gridcorrection0_2)*0.5f; // get real part
				outprev[w][1] = (f3d0i + f3d1i + gridcorrection1_2)*0.5f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += sfp.outpitch;
			outprev += sfp.outpitch;
			pattern3d += sfp.outpitch;
			gridsample += sfp.outpitch;
		}
	}
}

void ApplyPattern3D3_degrid_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev,
  fftwf_complex *outnext,
  SharedFunctionParams sfp)
{
  // int outwidth
  // int outpitch
  // int bh
  // int howmanyblocks
  // float *pattern3d
  // float beta,
  // float degrid
  // fftwf_complex *gridsample

	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni;
	float WienerFactor;
	float psd;
	float *pattern3d;
  fftwf_complex *gridsample;
	float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	int block;
	int h,w;

	for (block=0; block <sfp.howmanyblocks; block++)
	{
    pattern3d = sfp.pattern3d;
    gridsample = sfp.gridsample;
		float gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<sfp.bh; h++) // first half
		{
			for (w=0; w<sfp.outwidth; w++) // 
			{
				float gridcorrection0_3 = gridfraction*gridsample[w][0]*3;
				float gridcorrection1_3 = gridfraction*gridsample[w][1]*3;
				// dft 3d (very short - 3 points)
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fcr -= gridcorrection0_3;
				fci = outcur[w][1] + pni; // im cur
				fci -= gridcorrection1_3;
				float di = sin120*(outprev[w][1]-outnext[w][1]);
				float dr = sin120*(outnext[w][0]-outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = outcur[w][1] - 0.5f*pni - dr ; // im next
				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real part
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real part
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real part
				fni *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr + gridcorrection0_3)*0.33333333333f; // get real part
				outprev[w][1] = (fci + fpi + fni + gridcorrection1_3)*0.33333333333f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += sfp.outpitch;
			outprev += sfp.outpitch;
			outnext += sfp.outpitch;
			pattern3d += sfp.outpitch;
			gridsample += sfp.outpitch;
		}
	}
}

void ApplyPattern3D4_degrid_C(
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
  // float *pattern3d
  // float beta,
  // float degrid
  // fftwf_complex *gridsample

	// dft with 4 points
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
	float WienerFactor;
	float psd;
	float *pattern3d;
  fftwf_complex *gridsample;
	float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0

	int block;
	int h,w;

	for (block=0; block <sfp.howmanyblocks; block++)
	{
    pattern3d = sfp.pattern3d;
    gridsample = sfp.gridsample;
		float gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<sfp.bh; h++) // first half
		{
			for (w=0; w<sfp.outwidth; w++) // 
			{
				float gridcorrection0_4 = gridfraction*gridsample[w][0]*4;
				float gridcorrection1_4 = gridfraction*gridsample[w][1]*4;
				// dft 3d (very short - 4 points)
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fcr -= gridcorrection0_4;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fci -= gridcorrection1_4;
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real part
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real part
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real part
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real part
				fni *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0_4)*0.25f; // get real part
				outprev2[w][1] = (fp2i + fpi + fci + fni + gridcorrection1_4)*0.25f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += sfp.outpitch;
			outprev2 += sfp.outpitch;
			outprev += sfp.outpitch;
			outnext += sfp.outpitch;
			pattern3d += sfp.outpitch;
			gridsample += sfp.outpitch;
		}
	}
}

void ApplyPattern3D5_degrid_C(
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
  // float *pattern3d
  // float beta,
  // float degrid
  // fftwf_complex *gridsample

	// dft with 5 points
	// return result in outprev2
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i;
	float WienerFactor;
	float psd;
	float *pattern3d;
  fftwf_complex *gridsample;
	float lowlimit = (sfp.beta-1)/sfp.beta; //     (beta-1)/beta>=0
	float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	float cos72 = 0.30901699437494742410229341718282f;
	float sin144 = 0.58778525229247312916870595463907f; 
	float cos144 = -0.80901699437494742410229341718282f;

	int block;
	int h,w;

	for (block=0; block <sfp.howmanyblocks; block++)
	{
    pattern3d = sfp.pattern3d;
    gridsample = sfp.gridsample;
		float gridfraction = sfp.degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<sfp.bh; h++) // first half
		{
			for (w=0; w<sfp.outwidth; w++) // 
			{
				float gridcorrection0_5 = gridfraction*gridsample[w][0]*5;
				float gridcorrection1_5 = gridfraction*gridsample[w][1]*5;
				// dft 3d (very short - 5 points)
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72	+ (outprev[w][0] + outnext[w][0])*cos144 + + outcur[w][0];
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
				fcr -= gridcorrection0_5;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur
				fci -= gridcorrection1_5;

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real part
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real part
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real part
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real part
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r*fn2r + fn2i*fn2i + 1e-15f; // power spectrum density next
				WienerFactor = MAX((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real part
				fn2i *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r + gridcorrection0_5)*0.2f ; // get real part
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i + gridcorrection1_5)*0.2f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += sfp.outpitch;
			outprev2 += sfp.outpitch;
			outprev += sfp.outpitch;
			outnext += sfp.outpitch;
			outnext2 += sfp.outpitch;
			gridsample += sfp.outpitch;
			pattern3d += sfp.outpitch;
		}
	}
}
