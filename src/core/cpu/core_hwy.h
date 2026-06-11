#pragma once
#include "fft3d_common.h"

namespace neo_fft3d::cpu {

void Apply2D_Hwy(fftwf_complex* out, SharedFunctionParams sfp);
void Apply3D_Hwy(fftwf_complex** in, fftwf_complex* out, SharedFunctionParams sfp);
void Sharpen_Hwy(fftwf_complex* out, SharedFunctionParams sfp);
void Kalman_Hwy(fftwf_complex* curr, fftwf_complex* prev, SharedFunctionParams sfp);

} // namespace neo_fft3d::cpu
