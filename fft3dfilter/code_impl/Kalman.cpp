#include "code_impl_C.h"

template <bool pattern>
void Kalman_C(fftwf_complex *outcur, fftwf_complex *outLast, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, outLast, outcur, 0, 0};
  loop_wrapper_C(dummy, outLast, sfp,
    [&](LambdaFunctionParams lfp) {
      float GainRe, GainIm;
      float SumRe, SumIm;
      float sigma;

      auto incur = dummy[2];
      auto inprev = dummy[1];
      auto out = outLast;

      if (pattern) {
        sigma = lfp.pattern2d[lfp.w];
        // Prevent bad blocks, maybe incorrect -- by XL
        if (sigma < 1e-15f)
          sigma = 1e-15f;
      }
      else
        sigma = sfp.sigmaSquaredNoiseNormed2D;
      // use one of possible method for motion detection:
      if ((incur[lfp.w][0] - inprev[lfp.w][0]) * (incur[lfp.w][0] - inprev[lfp.w][0]) > sigma * sfp.kratio2 ||
          (incur[lfp.w][1] - inprev[lfp.w][1]) * (incur[lfp.w][1] - inprev[lfp.w][1]) > sigma * sfp.kratio2 )
      {
        // big pixel variation due to motion etc
        // reset filter
        lfp.covar[lfp.w][0] = sigma;
        lfp.covar[lfp.w][1] = sigma;
        lfp.covarProcess[lfp.w][0] = sigma;
        lfp.covarProcess[lfp.w][1] = sigma;
        out[lfp.w][0] = incur[lfp.w][0];
        out[lfp.w][1] = incur[lfp.w][1];
      }
      else
      { // small variation
        // useful sum
        SumRe = lfp.covar[lfp.w][0] + lfp.covarProcess[lfp.w][0];
        SumIm = lfp.covar[lfp.w][1] + lfp.covarProcess[lfp.w][1];
        // real gain, imagine gain
        GainRe = SumRe / (SumRe + sigma);
        GainIm = SumIm / (SumIm + sigma);
        // update process
        lfp.covarProcess[lfp.w][0] = (GainRe * GainRe * sigma);
        lfp.covarProcess[lfp.w][1] = (GainIm * GainIm * sigma);
        // update variation
        lfp.covar[lfp.w][0] = (1 - GainRe) * SumRe;
        lfp.covar[lfp.w][1] = (1 - GainIm) * SumIm;
        out[lfp.w][0] = GainRe * incur[lfp.w][0] + (1 - GainRe) * inprev[lfp.w][0];
        out[lfp.w][1] = GainIm * incur[lfp.w][1] + (1 - GainIm) * inprev[lfp.w][1];
      }
    }
  );
}

template void Kalman_C<true>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
template void Kalman_C<false>(fftwf_complex *, fftwf_complex *, SharedFunctionParams);
