#include <avisynth.h>
#include <vapoursynth/VapourSynth4.h>

#include <dualsynth/avisynth/video_bridge.hpp>
#include <dualsynth/vapoursynth/video_bridge.hpp>
#include <dualsynth/video_bridge.hpp>

#include "plugin/fft3d_filter.hpp"

#include <string>

#if defined(_WIN32)
#define NEO_FFT3D_AVS_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#elif defined(__clang__) || defined(__GNUC__)
#define NEO_FFT3D_AVS_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define NEO_FFT3D_AVS_PLUGIN_EXPORT extern "C"
#endif

const AVS_Linkage* AVS_linkage = nullptr;

namespace {

using FFT3DBridge = neo_fft3d::FFT3DBridge;

const char* vs_signature() {
  static const std::string signature = [] {
    const auto generated = ds::make_vapoursynth_signature(FFT3DBridge::descriptor());
    if (!generated.has_value()) {
      return std::string{"clip:vnode;"};
    }
    return generated.value();
  }();
  return signature.c_str();
}

const char* avs_signature() {
  static const std::string signature = [] {
    const auto generated = ds::make_avisynth_signature(FFT3DBridge::descriptor());
    if (!generated.has_value()) {
      return std::string{"c"};
    }
    return generated.value();
  }();
  return signature.c_str();
}

void VS_CC create_vapoursynth_fft3d(
  const VSMap* in,
  VSMap* out,
  void*,
  VSCore* core,
  const VSAPI* vsapi
) {
  ds::vapoursynth::create_video_filter_bridge<FFT3DBridge>(in, out, core, vsapi);
}

// AVS AddFunction requires this callback signature.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
AVSValue __cdecl create_avisynth_fft3d(AVSValue args, void*, IScriptEnvironment* env) {
  return ds::avisynth::create_video_filter_bridge<FFT3DBridge>(args, env);
}

const char* register_avisynth_fft3d(IScriptEnvironment* env, bool register_mt_mode) {
  env->AddFunction(
    FFT3DBridge::avs_name,
    avs_signature(),
    create_avisynth_fft3d,
    nullptr
  );
  if (register_mt_mode) {
    ds::avisynth::set_video_filter_mt_mode<FFT3DBridge>(env);
  }
  return neo_fft3d::Plugin::Description;
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
  vspapi->configPlugin(
    neo_fft3d::Plugin::Identifier,
    neo_fft3d::Plugin::Namespace,
    neo_fft3d::Plugin::Description,
    VS_MAKE_VERSION(1, 0),
    VAPOURSYNTH_API_VERSION,
    0,
    plugin
  );

  vspapi->registerFunction(
    FFT3DBridge::vs_name,
    vs_signature(),
    "clip:vnode;",
    create_vapoursynth_fft3d,
    nullptr,
    plugin
  );
}

NEO_FFT3D_AVS_PLUGIN_EXPORT const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env) {
  AVS_linkage = env->GetAVSLinkage();
  return register_avisynth_fft3d(env, false);
}
