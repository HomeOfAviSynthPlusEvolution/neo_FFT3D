/*
 * Copyright 2020 Xinyue Lu
 *
 * DualSynth bridge - plugin.
 *
 */

#pragma once

#include "version.hpp"
#include "fft3d.hpp"

namespace Plugin {
  const char* Identifier = "in.7086.neo_fft3d";
  const char* Namespace = "neo_fft3d";
  const char* Description = "Neo FFT3D Filter " PLUGIN_VERSION;
}

std::vector<register_vsfilter_proc> RegisterVSFilters()
{
  return std::vector<register_vsfilter_proc> { VSInterface::RegisterFilter<FFT3D> };
}

std::vector<register_avsfilter_proc> RegisterAVSFilters()
{
  return std::vector<register_avsfilter_proc> { AVSInterface::RegisterFilter<FFT3D> };
}
