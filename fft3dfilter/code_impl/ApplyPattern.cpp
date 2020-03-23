#include "code_impl.h"
#include "code_impl_C.h"

template <bool degrid>
void ApplyPattern2D_C(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  float gridcorrection0 = 0.0f;
  float gridcorrection1 = 0.0f;

  loop_wrapper_C(
    [&](LambdaFunctionParams lfp) {
      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0];
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1];
      }

      float corrected0 = outcur[lfp.w][0] - gridcorrection0;
      float corrected1 = outcur[lfp.w][1] - gridcorrection1;
      float psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f; // power spectrum density
      float WienerFactor = MAX((psd - sfp.pfactor * lfp.pattern2d[lfp.w]) / psd, lfp.lowlimit); // limited Wiener filter
      corrected0 *= WienerFactor; // apply filter on real part
      corrected1 *= WienerFactor; // apply filter on imaginary part
      outcur[lfp.w][0] = corrected0 + gridcorrection0;
      outcur[lfp.w][1] = corrected1 + gridcorrection1;
    }, sfp, outcur
  );
}

#define DECLARE(degrid) \
  template void ApplyPattern2D_C<degrid>(fftwf_complex*, SharedFunctionParams);\

DECLARE(true)
DECLARE(false)
#undef DECLARE
