#include "code_impl_C.h"

template <bool degrid, bool sharpen, bool dehalo>
static void Sharpen_C_impl(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  loop_wrapper_C(
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0f;
      float gridcorrection1 = 0.0f;

      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0]; // grid correction
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1];
      }

      float cr = outcur[lfp.w][0] - gridcorrection0;
      float ci = outcur[lfp.w][1] - gridcorrection1;

      float psd = cr * cr + ci * ci + 1e-15f;

      float s_fact = 1 + sfp.sharpen * lfp.wsharpen[lfp.w] * sqrt(
        psd * sfp.sigmaSquaredSharpenMaxNormed / ((psd + sfp.sigmaSquaredSharpenMinNormed) * (psd + sfp.sigmaSquaredSharpenMaxNormed))
        );
      float d_fact = (psd + sfp.ht2n) / ((psd + sfp.ht2n) + sfp.dehalo * lfp.wdehalo[lfp.w] * psd);

      float factor;

      if (sharpen && !dehalo)
        factor = s_fact;
      else if (!sharpen && dehalo)
        factor = d_fact;
      else if (sharpen && dehalo)
        factor = s_fact * d_fact;

      outcur[lfp.w][0] = cr * factor + gridcorrection0;
      outcur[lfp.w][1] = ci * factor + gridcorrection1;
    }, sfp, outcur
  );
}

template <bool degrid>
void Sharpen_C(
  fftwf_complex *outcur,
  SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    return;
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Sharpen_C_impl<degrid, true, false>(outcur, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Sharpen_C_impl<degrid, false, true>(outcur, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Sharpen_C_impl<degrid, true, true>(outcur, sfp);
}

template void Sharpen_C<true>(fftwf_complex *outcur, SharedFunctionParams sfp);
template void Sharpen_C<false>(fftwf_complex *outcur, SharedFunctionParams sfp);
