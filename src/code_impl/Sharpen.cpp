/*
 * Copyright 2004-2007 A.G.Balakhnin aka Fizick
 * Copyright 2015 martin53
 * Copyright 2017-2019 Ferenc Pinter aka printerf
 * Copyright 2020 Xinyue Lu
 *
 * Sharpener implementation, Pure C code.
 *
 */

#include "code_impl_C.h"

#include <cmath>

template <bool degrid, bool sharpen, bool dehalo>
static inline void Sharpen_C_impl(fftwf_complex *out, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {nullptr, nullptr, out, nullptr, nullptr};
  loop_wrapper_C(dummy, out, sfp,
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0F;
      float gridcorrection1 = 0.0F;

      if constexpr (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0]; // grid correction
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1];
      }

      float cr = out[lfp.w][0] - gridcorrection0;
      float ci = out[lfp.w][1] - gridcorrection1;

      float psd = (cr * cr) + (ci * ci) + 1.0e-15F;

      float s_fact = 1.0F;
      if constexpr (sharpen) {
        s_fact += sfp.sharpen * lfp.wsharpen[lfp.w] * std::sqrt(
          psd * sfp.sigmaSquaredSharpenMaxNormed / ((psd + sfp.sigmaSquaredSharpenMinNormed) * (psd + sfp.sigmaSquaredSharpenMaxNormed))
          );
      }
      float d_fact = 1.0F;
      if constexpr (dehalo) {
        d_fact = (psd + sfp.ht2n) / ((psd + sfp.ht2n) + sfp.dehalo * lfp.wdehalo[lfp.w] * psd);
      }

      float factor = 1.0F;

      if constexpr (sharpen && !dehalo) {
        factor = s_fact;
      } else if constexpr (!sharpen && dehalo) {
        factor = d_fact;
      } else if constexpr (sharpen && dehalo) {
        factor = s_fact * d_fact;
      }

      out[lfp.w][0] = cr * factor + gridcorrection0;
      out[lfp.w][1] = ci * factor + gridcorrection1;
    }
  );
}

template <bool degrid>
void Sharpen_C(fftwf_complex *out, SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0) {
    return;
  }
  if (sfp.sharpen != 0 && sfp.dehalo == 0) {
    Sharpen_C_impl<degrid, true, false>(out, sfp);
  } else if (sfp.sharpen == 0 && sfp.dehalo != 0) {
    Sharpen_C_impl<degrid, false, true>(out, sfp);
  } else if (sfp.sharpen != 0 && sfp.dehalo != 0) {
    Sharpen_C_impl<degrid, true, true>(out, sfp);
  }
}

template void Sharpen_C<true>(fftwf_complex *, SharedFunctionParams);
template void Sharpen_C<false>(fftwf_complex *, SharedFunctionParams);
