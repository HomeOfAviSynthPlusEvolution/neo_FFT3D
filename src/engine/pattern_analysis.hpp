#pragma once
#ifndef NEO_FFT3D_ENGINE_PATTERN_ANALYSIS_HPP
#define NEO_FFT3D_ENGINE_PATTERN_ANALYSIS_HPP

#include "common/fft3d_views.hpp"

namespace neo_fft3d {

void FindPatternBlock(ComplexBlockView spectrum, int outwidth, int nox, int noy, int &px, int &py, ConstFloatPlaneView pwin, float degrid, ComplexBlockView gridsample);
void SetPattern(ComplexBlockView spectrum, int outwidth, int nox, int noy, int px, int py, ConstFloatPlaneView pwin, FloatPlaneView pattern2d, float &psigma, float degrid, ComplexBlockView gridsample);
void PutPatternOnly(ComplexBlockView spectrum, int outwidth, int nox, int noy, int px, int py);
void Pattern2Dto3D(ConstFloatPlaneView pattern2d, float mult, FloatPlaneView pattern3d);
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int outwidth, float norm, FloatPlaneView pattern2d);

} // namespace neo_fft3d

#endif
