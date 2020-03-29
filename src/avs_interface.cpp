/*
 * Copyright 2020 Xinyue Lu
 *
 * AviSynth+ Public API.
 *
 */

#include "wrapper/avs_filter.hpp"
#include "neo_fft3d.hpp"
#include "version.hpp"

AVSValue __cdecl CreateAVSFilter(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  auto filter = new FFT3D<AVSFilter>(args, env);
  try {
    filter->initialize();
  }
  catch (const char *err) {
    env->ThrowError("%s %s: %s", filter->name(), PLUGIN_VERSION, err);
  }
  return filter;
}

const AVS_Linkage *AVS_linkage = NULL;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, AVS_Linkage* vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("neo_fft3d", "c[sigma]f[beta]f[unused1]i[bw]i[bh]i[bt]i[ow]i[oh]i[kratio]f[sharpen]f[scutoff]f[svr]f[smin]f[smax]f[measure]b[interlaced]b[wintype]i[pframe]i[px]i[py]i[pshow]b[pcutoff]f[pfactor]f[sigma2]f[sigma3]f[sigma4]f[degrid]f[dehalo]f[hr]f[ht]f[ncpu]i[y]i[u]i[v]i[l]i[t]i[r]i[b]i", CreateAVSFilter, 0);

  return "Neo FFT3D Filter";
}
