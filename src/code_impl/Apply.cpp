/*
 * Copyright 2004-2007 A.G.Balakhnin aka Fizick
 * Copyright 2015 martin53
 * Copyright 2017-2019 Ferenc Pinter aka printerf
 * Copyright 2020 Xinyue Lu
 *
 * 2D and 3D filter implementation, Pure C code.
 *
 */

#include "code_impl_C.h"

template <bool pattern, bool degrid, bool sharpen, bool dehalo>
static inline void Apply2D_C_impl(fftwf_complex *out, SharedFunctionParams sfp)
{
  fftwf_complex * dummy[5] = {0, 0, out, 0, 0};
  loop_wrapper_C(dummy, out, sfp,
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0f;
      float gridcorrection1 = 0.0f;

      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0]; // grid correction
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1];
      }

      float cr = out[lfp.w][0] - gridcorrection0;
      float ci = out[lfp.w][1] - gridcorrection1;

      float psd = cr * cr + ci * ci + 1e-15f;
      float factor = MAX((psd - (pattern ? lfp.pattern2d[lfp.w] : sfp.sigmaSquaredNoiseNormed) ) / psd, lfp.lowlimit); // limited Wiener filter

      if (!pattern) {
        // Skip sharpen and dehalo for ApplyPattern family
        float s_fact = 1 + sfp.sharpen * lfp.wsharpen[lfp.w] * sqrt(
          psd * sfp.sigmaSquaredSharpenMaxNormed / ((psd + sfp.sigmaSquaredSharpenMinNormed) * (psd + sfp.sigmaSquaredSharpenMaxNormed))
          );
        float d_fact = (psd + sfp.ht2n) / ((psd + sfp.ht2n) + sfp.dehalo * lfp.wdehalo[lfp.w] * psd);

        if (sharpen && !dehalo)
          factor *= s_fact;
        else if (!sharpen && dehalo)
          factor *= d_fact;
        else if (sharpen && dehalo)
          factor *= s_fact * d_fact;
      }

      out[lfp.w][0] = cr * factor + gridcorrection0;
      out[lfp.w][1] = ci * factor + gridcorrection1;
    }
  );
}

template <bool pattern, bool degrid>
void Apply2D_C(fftwf_complex *out, SharedFunctionParams sfp)
{
  if (sfp.sharpen == 0 && sfp.dehalo == 0)
    Apply2D_C_impl<pattern, degrid, true, false>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo == 0)
    Apply2D_C_impl<pattern, degrid, true, false>(out, sfp);
  else if (sfp.sharpen == 0 && sfp.dehalo != 0)
    Apply2D_C_impl<pattern, degrid, false, true>(out, sfp);
  else if (sfp.sharpen != 0 && sfp.dehalo != 0)
    Apply2D_C_impl<pattern, degrid, true, true>(out, sfp);
}

template <bool pattern, bool degrid>
void Apply3D2_C(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  loop_wrapper_C(in, out, sfp,
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0f;
      float gridcorrection1 = 0.0f;
      auto incur = in[2];
      auto inprev = in[1];

      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 2; // grid correction
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 2;
      }

      // dft 3d (very short - 2 points)
      float f3d0r = incur[lfp.w][0] + inprev[lfp.w][0] - gridcorrection0; // real 0 (sum)
      float f3d0i = incur[lfp.w][1] + inprev[lfp.w][1] - gridcorrection1; // im 0 (sum)
      float f3d1r = incur[lfp.w][0] - inprev[lfp.w][0]; // real 1 (dif)
      float f3d1i = incur[lfp.w][1] - inprev[lfp.w][1]; // im 1 (dif)

      lfp.wiener_factor_3d<pattern>(f3d0r, f3d0i);
      lfp.wiener_factor_3d<pattern>(f3d1r, f3d1i);

      // reverse dft for 2 points
      out[lfp.w][0] = (f3d0r + f3d1r + gridcorrection0) * 0.5f; // get real part
      out[lfp.w][1] = (f3d0i + f3d1i + gridcorrection1) * 0.5f; // get imaginary part
    }
  );
}

template <bool pattern, bool degrid>
void Apply3D3_C(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  constexpr float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;
  constexpr float athird = 1.0f/3.0f;

  loop_wrapper_C(in, out, sfp,
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0f;
      float gridcorrection1 = 0.0f;
      auto incur = in[2];
      auto inprev = in[1];
      auto innext = in[3];

      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 3;
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 3;
      }

      // dft 3d (very short - 3 points)
      float pnr = inprev[lfp.w][0] + innext[lfp.w][0];
      float pni = inprev[lfp.w][1] + innext[lfp.w][1];
      float fcr = incur[lfp.w][0] + pnr - gridcorrection0;
      float fci = incur[lfp.w][1] + pni - gridcorrection1;
      float di = sin120*(inprev[lfp.w][1]-innext[lfp.w][1]);
      float dr = sin120*(innext[lfp.w][0]-inprev[lfp.w][0]);
      float fpr, fpi, fnr, fni;
      fpr = incur[lfp.w][0] - 0.5f*pnr + di; // real prev
      fnr = incur[lfp.w][0] - 0.5f*pnr - di; // real next
      fpi = incur[lfp.w][1] - 0.5f*pni + dr; // im prev
      fni = incur[lfp.w][1] - 0.5f*pni - dr; // im next

      lfp.wiener_factor_3d<pattern>(fcr, fci);
      lfp.wiener_factor_3d<pattern>(fpr, fpi);
      lfp.wiener_factor_3d<pattern>(fnr, fni);

      // reverse dft for 3 points
      out[lfp.w][0] = (fcr + fpr + fnr + gridcorrection0) * athird; // get real part
      out[lfp.w][1] = (fci + fpi + fni + gridcorrection1) * athird; // get imaginary part
    }
  );
}

template <bool pattern, bool degrid>
void Apply3D4_C(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  loop_wrapper_C(in, out, sfp,
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0f;
      float gridcorrection1 = 0.0f;
      auto incur = in[2];
      auto inprev = in[1];
      auto innext = in[3];
      auto inprev2 = in[0];
      float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;

      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 4;
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 4;
      }

      // dft 3d (very short - 4 points)
      fp2r = (incur[lfp.w][0] + inprev2[lfp.w][0]) - (inprev[lfp.w][0] + innext[lfp.w][0]); // real prev2
      fp2i = (incur[lfp.w][1] + inprev2[lfp.w][1]) - (inprev[lfp.w][1] + innext[lfp.w][1]); // im cur
      fpr  = (incur[lfp.w][0] - inprev2[lfp.w][0]) + (inprev[lfp.w][1] - innext[lfp.w][1]); // real prev
      fpi  = (incur[lfp.w][1] - inprev2[lfp.w][1]) - (inprev[lfp.w][0] - innext[lfp.w][0]); // im cur
      fcr  = (incur[lfp.w][0] + inprev2[lfp.w][0]) + (inprev[lfp.w][0] + innext[lfp.w][0]) - gridcorrection0;
      fci  = (incur[lfp.w][1] + inprev2[lfp.w][1]) + (inprev[lfp.w][1] + innext[lfp.w][1]) - gridcorrection1;
      fnr  = (incur[lfp.w][0] - inprev2[lfp.w][0]) - (inprev[lfp.w][1] - innext[lfp.w][1]); // real next
      fni  = (incur[lfp.w][1] - inprev2[lfp.w][1]) + (inprev[lfp.w][0] - innext[lfp.w][0]); // im next

      lfp.wiener_factor_3d<pattern>(fp2r, fp2i);
      lfp.wiener_factor_3d<pattern>(fpr, fpi);
      lfp.wiener_factor_3d<pattern>(fcr, fci);
      lfp.wiener_factor_3d<pattern>(fnr, fni);

      // reverse dft for 4 points
      out[lfp.w][0] = ((fp2r + fpr) + (fcr + fnr) + gridcorrection0) * 0.25f; // get real part
      out[lfp.w][1] = ((fp2i + fpi) + (fci + fni) + gridcorrection1) * 0.25f; // get imaginary part
    }
  );
}

template <bool pattern, bool degrid>
void Apply3D5_C(fftwf_complex **in, fftwf_complex *out, SharedFunctionParams sfp)
{
  constexpr float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
  constexpr float cos72 = 0.30901699437494742410229341718282f;
  constexpr float sin144 = 0.58778525229247312916870595463907f;
  constexpr float cos144 = -0.80901699437494742410229341718282f;

  loop_wrapper_C(in, out, sfp,
    [&](LambdaFunctionParams lfp) {
      float gridcorrection0 = 0.0f;
      float gridcorrection1 = 0.0f;
      auto incur = in[2];
      auto inprev = in[1];
      auto innext = in[3];
      auto inprev2 = in[0];
      auto innext2 = in[4];

      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 5;
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 5;
      }

      // dft 3d (very short - 5 points)
      float sum = (inprev2[lfp.w][0] + innext2[lfp.w][0])*cos72 + (inprev[lfp.w][0] + innext[lfp.w][0])*cos144 + incur[lfp.w][0];
      float dif = (- inprev2[lfp.w][1] + innext2[lfp.w][1])*sin72 + (inprev[lfp.w][1]  - innext[lfp.w][1])*sin144;
      float fp2r = sum + dif; // real prev2
      float fn2r = sum - dif; // real next2
      sum = (inprev2[lfp.w][1] + innext2[lfp.w][1])*cos72 + (inprev[lfp.w][1] + innext[lfp.w][1])*cos144 + incur[lfp.w][1];
      dif = (inprev2[lfp.w][0] - innext2[lfp.w][0])*sin72 + (- inprev[lfp.w][0] + innext[lfp.w][0])*sin144;
      float fp2i = sum + dif; // im prev2
      float fn2i = sum - dif; // im next2
      sum = (inprev2[lfp.w][0] + innext2[lfp.w][0])*cos144 + (inprev[lfp.w][0] + innext[lfp.w][0])*cos72 + incur[lfp.w][0];
      dif = (inprev2[lfp.w][1] - innext2[lfp.w][1])*sin144 + (inprev[lfp.w][1] - innext[lfp.w][1])*sin72;
      float fpr = sum + dif; // real prev
      float fnr = sum - dif; // real next
      sum = (inprev2[lfp.w][1] + innext2[lfp.w][1])*cos144 + (inprev[lfp.w][1] + innext[lfp.w][1])*cos72 + incur[lfp.w][1];
      dif =  (- inprev2[lfp.w][0] + innext2[lfp.w][0])*sin144 + (- inprev[lfp.w][0] + innext[lfp.w][0])*sin72;
      float fpi = sum + dif; // im prev
      float fni = sum - dif; // im next
      float fcr = inprev2[lfp.w][0] + inprev[lfp.w][0] + incur[lfp.w][0] + innext[lfp.w][0] + innext2[lfp.w][0] - gridcorrection0;
      float fci = inprev2[lfp.w][1] + inprev[lfp.w][1] + incur[lfp.w][1] + innext[lfp.w][1] + innext2[lfp.w][1] - gridcorrection1;

      lfp.wiener_factor_3d<pattern>(fp2r, fp2i);
      lfp.wiener_factor_3d<pattern>(fpr, fpi);
      lfp.wiener_factor_3d<pattern>(fcr, fci);
      lfp.wiener_factor_3d<pattern>(fnr, fni);
      lfp.wiener_factor_3d<pattern>(fn2r, fn2i);

      // reverse dft for 5 points
      out[lfp.w][0] = (fp2r + fpr + fcr + fnr + fn2r + gridcorrection0) * 0.2f; // get real part
      out[lfp.w][1] = (fp2i + fpi + fci + fni + fn2i + gridcorrection1) * 0.2f; // get imaginary part
    }
  );
}


#define DECLARE(pattern, degrid) \
  template void Apply2D_C<pattern, degrid>(fftwf_complex*, SharedFunctionParams);\
  template void Apply3D2_C<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D3_C<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D4_C<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D5_C<pattern, degrid>(fftwf_complex**, fftwf_complex*, SharedFunctionParams);\

DECLARE(true, true)
DECLARE(false, true)
DECLARE(true, false)
DECLARE(false, false)
#undef DECLARE
