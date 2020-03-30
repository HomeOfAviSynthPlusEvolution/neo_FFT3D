#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "common.h"
#include <type_traits>

void FrameToCover(EngineParams * ep, int plane, const byte *src_ptr, byte *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh);
void CoverToFrame(EngineParams * ep, int plane, const byte *coverbuf, int coverwidth, int coverheight, int coverpitch, byte *dst_ptr, int mirw, int mirh);
void CoverToOverlap(EngineParams * ep, IOParams * iop, float *dst_ptr, const byte *src_ptr, int src_width, int src_pitch, bool chroma);
void OverlapToCover(EngineParams * ep, IOParams * iop, float *src_ptr, float norm, byte *dst_ptr, int dst_width, int dst_pitch, bool chroma);

#endif
