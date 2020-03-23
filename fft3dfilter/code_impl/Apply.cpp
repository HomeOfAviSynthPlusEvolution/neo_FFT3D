#include "code_impl_C.h"

template <bool pattern, bool degrid>
void Apply3D2_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev,
  SharedFunctionParams sfp)
{
  float gridcorrection0 = 0.0f;
  float gridcorrection1 = 0.0f;

  loop_wrapper_C(
    [&](LambdaFunctionParams lfp) {
      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 2; // grid correction
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 2;
      }

      // dft 3d (very short - 2 points)
      float f3d0r = outcur[lfp.w][0] + outprev[lfp.w][0] - gridcorrection0; // real 0 (sum)
      float f3d0i = outcur[lfp.w][1] + outprev[lfp.w][1] - gridcorrection1; // im 0 (sum)
      float f3d1r = outcur[lfp.w][0] - outprev[lfp.w][0]; // real 1 (dif)
      float f3d1i = outcur[lfp.w][1] - outprev[lfp.w][1]; // im 1 (dif)

      lfp.wiener_factor_3d<pattern>(f3d0r, f3d0i);
      lfp.wiener_factor_3d<pattern>(f3d1r, f3d1i);

      // reverse dft for 2 points
      outprev[lfp.w][0] = (f3d0r + f3d1r + gridcorrection0) * 0.5f; // get real part
      outprev[lfp.w][1] = (f3d0i + f3d1i + gridcorrection1) * 0.5f; // get imaginary part
      // Attention! return filtered "out" in "outprev" to preserve "out" for next step
    }, sfp, outcur, outprev
  );
}

template <bool pattern, bool degrid>
void Apply3D3_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev,
  fftwf_complex *outnext,
  SharedFunctionParams sfp)
{
  float gridcorrection0 = 0.0f;
  float gridcorrection1 = 0.0f;
  const float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

  loop_wrapper_C(
    [&](LambdaFunctionParams lfp) {
      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 3;
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 3;
      }

      // dft 3d (very short - 3 points)
      float pnr = outprev[lfp.w][0] + outnext[lfp.w][0];
      float pni = outprev[lfp.w][1] + outnext[lfp.w][1];
      float fcr = outcur[lfp.w][0] + pnr - gridcorrection0;
      float fci = outcur[lfp.w][1] + pni - gridcorrection1;
      float di = sin120*(outprev[lfp.w][1]-outnext[lfp.w][1]);
      float dr = sin120*(outnext[lfp.w][0]-outprev[lfp.w][0]);
      float fpr, fpi, fnr, fni;
      fpr = outcur[lfp.w][0] - 0.5f*pnr + di; // real prev
      fnr = outcur[lfp.w][0] - 0.5f*pnr - di; // real next
      fpi = outcur[lfp.w][1] - 0.5f*pni + dr; // im prev
      fni = outcur[lfp.w][1] - 0.5f*pni - dr; // im next

      lfp.wiener_factor_3d<pattern>(fcr, fci);
      lfp.wiener_factor_3d<pattern>(fpr, fpi);
      lfp.wiener_factor_3d<pattern>(fnr, fni);

      // reverse dft for 3 points
      outprev[lfp.w][0] = (fcr + fpr + fnr + gridcorrection0) * 0.33333333333f; // get real part
      outprev[lfp.w][1] = (fci + fpi + fni + gridcorrection1) * 0.33333333333f; // get imaginary part
      // Attention! return filtered "out" in "outprev" to preserve "out" for next step
    }, sfp, outcur, outprev, outnext
  );
}

template <bool pattern, bool degrid>
void Apply3D4_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev2,
  fftwf_complex *outprev,
  fftwf_complex *outnext,
  SharedFunctionParams sfp)
{
  float gridcorrection0 = 0.0f;
  float gridcorrection1 = 0.0f;

  float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;

  loop_wrapper_C(
    [&](LambdaFunctionParams lfp) {
      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 4;
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 4;
      }

      // dft 3d (very short - 4 points)
      fp2r = outprev2[lfp.w][0] - outprev[lfp.w][0] + outcur[lfp.w][0] - outnext[lfp.w][0]; // real prev2
      fp2i = outprev2[lfp.w][1] - outprev[lfp.w][1] + outcur[lfp.w][1] - outnext[lfp.w][1]; // im cur
      fpr = -outprev2[lfp.w][0] + outprev[lfp.w][1] + outcur[lfp.w][0] - outnext[lfp.w][1]; // real prev
      fpi = -outprev2[lfp.w][1] - outprev[lfp.w][0] + outcur[lfp.w][1] + outnext[lfp.w][0]; // im cur
      fcr = outprev2[lfp.w][0] + outprev[lfp.w][0] + outcur[lfp.w][0] + outnext[lfp.w][0] - gridcorrection0;
      fci = outprev2[lfp.w][1] + outprev[lfp.w][1] + outcur[lfp.w][1] + outnext[lfp.w][1] - gridcorrection1;
      fnr = -outprev2[lfp.w][0] - outprev[lfp.w][1] + outcur[lfp.w][0] + outnext[lfp.w][1]; // real next
      fni = -outprev2[lfp.w][1] + outprev[lfp.w][0] + outcur[lfp.w][1] - outnext[lfp.w][0]; // im next

      lfp.wiener_factor_3d<pattern>(fp2r, fp2i);
      lfp.wiener_factor_3d<pattern>(fpr, fpi);
      lfp.wiener_factor_3d<pattern>(fcr, fci);
      lfp.wiener_factor_3d<pattern>(fnr, fni);

      // reverse dft for 4 points
      outprev2[lfp.w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0) * 0.25f; // get real part
      outprev2[lfp.w][1] = (fp2i + fpi + fci + fni + gridcorrection1) * 0.25f; // get imaginary part
      // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
    }, sfp, outcur, outprev2, outprev, outnext
  );
}

template <bool pattern, bool degrid>
void Apply3D5_C(
  fftwf_complex *outcur,
  fftwf_complex *outprev2,
  fftwf_complex *outprev,
  fftwf_complex *outnext,
  fftwf_complex *outnext2,
  SharedFunctionParams sfp)
{
  float gridcorrection0 = 0.0f;
  float gridcorrection1 = 0.0f;

  float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
  float cos72 = 0.30901699437494742410229341718282f;
  float sin144 = 0.58778525229247312916870595463907f;
  float cos144 = -0.80901699437494742410229341718282f;

  loop_wrapper_C(
    [&](LambdaFunctionParams lfp) {
      if (degrid) {
        gridcorrection0 = lfp.gridfraction * lfp.gridsample[lfp.w][0] * 5;
        gridcorrection1 = lfp.gridfraction * lfp.gridsample[lfp.w][1] * 5;
      }

      // dft 3d (very short - 5 points)
      float sum = (outprev2[lfp.w][0] + outnext2[lfp.w][0])*cos72 + (outprev[lfp.w][0] + outnext[lfp.w][0])*cos144 + outcur[lfp.w][0];
      float dif = (- outprev2[lfp.w][1] + outnext2[lfp.w][1])*sin72 + (outprev[lfp.w][1]  - outnext[lfp.w][1])*sin144;
      float fp2r = sum + dif; // real prev2
      float fn2r = sum - dif; // real next2
      sum = (outprev2[lfp.w][1] + outnext2[lfp.w][1])*cos72 + (outprev[lfp.w][1] + outnext[lfp.w][1])*cos144 + outcur[lfp.w][1];
      dif = (outprev2[lfp.w][0] - outnext2[lfp.w][0])*sin72 + (- outprev[lfp.w][0] + outnext[lfp.w][0])*sin144;
      float fp2i = sum + dif; // im prev2
      float fn2i = sum - dif; // im next2
      sum = (outprev2[lfp.w][0] + outnext2[lfp.w][0])*cos144 + (outprev[lfp.w][0] + outnext[lfp.w][0])*cos72 + outcur[lfp.w][0];
      dif = (outprev2[lfp.w][1] - outnext2[lfp.w][1])*sin144 + (outprev[lfp.w][1] - outnext[lfp.w][1])*sin72;
      float fpr = sum + dif; // real prev
      float fnr = sum - dif; // real next
      sum = (outprev2[lfp.w][1] + outnext2[lfp.w][1])*cos144 + (outprev[lfp.w][1] + outnext[lfp.w][1])*cos72 + outcur[lfp.w][1];
      dif =  (- outprev2[lfp.w][0] + outnext2[lfp.w][0])*sin144 + (- outprev[lfp.w][0] + outnext[lfp.w][0])*sin72;
      float fpi = sum + dif; // im prev
      float fni = sum - dif; // im next
      float fcr = outprev2[lfp.w][0] + outprev[lfp.w][0] + outcur[lfp.w][0] + outnext[lfp.w][0] + outnext2[lfp.w][0] - gridcorrection0;
      float fci = outprev2[lfp.w][1] + outprev[lfp.w][1] + outcur[lfp.w][1] + outnext[lfp.w][1] + outnext2[lfp.w][1] - gridcorrection1;

      lfp.wiener_factor_3d<pattern>(fp2r, fp2i);
      lfp.wiener_factor_3d<pattern>(fpr, fpi);
      lfp.wiener_factor_3d<pattern>(fcr, fci);
      lfp.wiener_factor_3d<pattern>(fnr, fni);
      lfp.wiener_factor_3d<pattern>(fn2r, fn2i);

      // reverse dft for 5 points
      outprev2[lfp.w][0] = (fp2r + fpr + fcr + fnr + fn2r + gridcorrection0) * 0.2f; // get real part
      outprev2[lfp.w][1] = (fp2i + fpi + fci + fni + fn2i + gridcorrection1) * 0.2f; // get imaginary part
      // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
    }, sfp, outcur, outprev2, outprev, outnext, outnext2
  );
}

#define DECLARE(pattern, degrid) \
  template void Apply3D2_C<pattern, degrid>(fftwf_complex*, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D3_C<pattern, degrid>(fftwf_complex*, fftwf_complex*, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D4_C<pattern, degrid>(fftwf_complex*, fftwf_complex*, fftwf_complex*, fftwf_complex*, SharedFunctionParams);\
  template void Apply3D5_C<pattern, degrid>(fftwf_complex*, fftwf_complex*, fftwf_complex*, fftwf_complex*, fftwf_complex*, SharedFunctionParams);\

DECLARE(true, true)
DECLARE(false, true)
DECLARE(true, false)
DECLARE(false, false)
#undef DECLARE
