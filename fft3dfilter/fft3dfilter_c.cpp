//
//	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter
//  pure C++ filtering functions
//  v1.9.2
//	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License version 2 as published by
//	the Free Software Foundation.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
//
//-----------------------------------------------------------------------------------------

#include "common.h"

// since v1.7 we use outpitch instead of outwidth
//-------------------------------------------------------------------------------------------
//

//
//-----------------------------------------------------------------------------------------
//
void ApplyKalmanPattern_C( fftwf_complex *outcur, fftwf_complex *outLast, 
						fftwf_complex *covar, fftwf_complex *covarProcess,
						int outwidth, int outpitch, int bh, int howmanyblocks,  
						float *covarNoiseNormed, float kratio2)
{
// return result in outLast
	float GainRe, GainIm;  // Kalman Gain 
//	float filteredRe, filteredIm;
	float sumre, sumim;
	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // 
		{
			for (w=0; w<outwidth; w++) 
			{
				// use one of possible method for motion detection:
				if ( (outcur[w][0]-outLast[w][0])*(outcur[w][0]-outLast[w][0]) > covarNoiseNormed[w]*kratio2 || 
				 	 (outcur[w][1]-outLast[w][1])*(outcur[w][1]-outLast[w][1]) > covarNoiseNormed[w]*kratio2 )
				{
					// big pixel variation due to motion etc
					// reset filter
					covar[w][0] = covarNoiseNormed[w]; 
					covar[w][1] = covarNoiseNormed[w]; 
					covarProcess[w][0] = covarNoiseNormed[w]; 
					covarProcess[w][1] = covarNoiseNormed[w]; 
					outLast[w][0] = outcur[w][0];
					outLast[w][1] = outcur[w][1];
					//return result in outLast
				}
				else
				{ // small variation
					// useful sum
					sumre = (covar[w][0] + covarProcess[w][0]);
					sumim = (covar[w][1] + covarProcess[w][1]);
					// real gain, imagine gain
					GainRe = sumre/(sumre + covarNoiseNormed[w]);
					GainIm = sumim/(sumim + covarNoiseNormed[w]);
					// update process
					covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed[w]);
					covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed[w]);
					// update variation
					covar[w][0] =  (1-GainRe)*sumre ;
					covar[w][1] =  (1-GainIm)*sumim ;
					outLast[w][0] = ( GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0] );
					outLast[w][1] = ( GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1] );
					//return filtered result in outLast
				}
			}
			outcur += outpitch;
			outLast += outpitch; 
			covar += outpitch; 
			covarProcess += outpitch; 
			covarNoiseNormed += outpitch;
		}
		covarNoiseNormed -= outpitch*bh;
	}
	
}
//-----------------------------------------------------------------------------------------
//
void ApplyKalman_C( fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, 
				 fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, 
				 int howmanyblocks,  float covarNoiseNormed, float kratio2)
{
// return result in outLast
	float GainRe, GainIm;  // Kalman Gain 
//	float filteredRe, filteredIm;
	float sumre, sumim;
	int block;
	int h,w;

	float sigmaSquaredMotionNormed = covarNoiseNormed*kratio2;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // 
		{
			for (w=0; w<outwidth; w++) 
			{
				// use one of possible method for motion detection:
				if ( (outcur[w][0]-outLast[w][0])*(outcur[w][0]-outLast[w][0]) > sigmaSquaredMotionNormed || 
				 	 (outcur[w][1]-outLast[w][1])*(outcur[w][1]-outLast[w][1]) > sigmaSquaredMotionNormed )
				{
					// big pixel variation due to motion etc
					// reset filter
					covar[w][0] = covarNoiseNormed; 
					covar[w][1] = covarNoiseNormed; 
					covarProcess[w][0] = covarNoiseNormed; 
					covarProcess[w][1] = covarNoiseNormed; 
					outLast[w][0] = outcur[w][0];
					outLast[w][1] = outcur[w][1];
					//return result in outLast
				}
				else
				{ // small variation
					// useful sum
					sumre = (covar[w][0] + covarProcess[w][0]);
					sumim = (covar[w][1] + covarProcess[w][1]);
					// real gain, imagine gain
					GainRe = sumre/(sumre + covarNoiseNormed);
					GainIm = sumim/(sumim + covarNoiseNormed);
					// update process
					covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed);
					covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed);
					// update variation
					covar[w][0] =  (1-GainRe)*sumre ;
					covar[w][1] =  (1-GainIm)*sumim ;
					outLast[w][0] = ( GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0] );
					outLast[w][1] = ( GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1] );
					//return filtered result in outLast
				}
			}
			outcur += outpitch;
			outLast += outpitch; 
			covar += outpitch; 
			covarProcess += outpitch; 
		}
	}
	
}

//-------------------------------------------------------------------------------------------
//
void Sharpen_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
			 int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, 
			 float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n)
{
	int h,w, block;
	float psd;
	float sfact;

	if (sharpen != 0 && dehalo==0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ) ;
					// sharpen factor - changed in v1.1c
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
				}
				outcur += outpitch;
				wsharpen += outpitch;
			}
			wsharpen -= outpitch*bh;
		}
	}
	else if (sharpen == 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
				}
				outcur += outpitch;
				wdehalo += outpitch;
			}
			wdehalo -= outpitch*bh;
		}
	}
	else if (sharpen != 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ) *
						(psd + ht2n) / ((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
		}
	}
}
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
// DEGRID
//-----------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//
void Sharpen_degrid_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
			 int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, 
			 float sigmaSquaredSharpenMax, float *wsharpen, 
			 float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n)
{
	int h,w, block;
	float psd;
	float sfact;

	if (sharpen != 0 && dehalo==0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im*im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) )) ; 
					// sharpen factor - changed in v1.1c
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
	if (sharpen == 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im*im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (psd + ht2n) / ((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
	if (sharpen != 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im*im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) )) * 
						(psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
}
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
