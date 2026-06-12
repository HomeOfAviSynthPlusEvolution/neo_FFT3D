#pragma once
#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "fft3d_common.h"
#include "common/fft3d_views.hpp"
#include <type_traits>

void FrameToCover(EngineParams * ep, int plane, neo_fft3d::BytePlaneView src, neo_fft3d::MutableBytePlaneView cover, int mirw, int mirh);
void CoverToFrame(EngineParams * ep, int plane, neo_fft3d::BytePlaneView cover, neo_fft3d::MutableBytePlaneView dst, int mirw, int mirh);
void CoverToOverlap(EngineParams * ep, IOParams * iop, neo_fft3d::FloatSpan dst, neo_fft3d::BytePlaneView src, bool chroma);
void OverlapToCover(EngineParams * ep, IOParams * iop, neo_fft3d::FloatSpan src, float norm, neo_fft3d::MutableBytePlaneView dst, bool chroma);

#endif
