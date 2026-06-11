#pragma once

#include <dualsynth/avisynth/video_bridge.hpp>
#include <dualsynth/format.hpp>
#include <dualsynth/param.hpp>
#include <dualsynth/video_bridge.hpp>
#include <dualsynth/video_filter.hpp>

#include "version.hpp"
#include "dualsynth_compat.hpp"
#include "fft/fft_backend.hpp"

#include <memory>

class FFT3DEngine;
struct EngineParams;
struct FFTFunctionPointers;

namespace neo_fft3d {

struct FFT3DCore {
  static constexpr const char* name = "FFT3D";
  static constexpr int input_count = 1;
  static constexpr ds::OutputOrigin output_origin = ds::OutputOrigin::fresh();

  struct State {
    State();
    ~State();

    State(State&&) noexcept;
    State& operator=(State&&) noexcept;
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    int process[4] {2, 2, 2, 2};
    FFT3DEngine* engine[4] {nullptr, nullptr, nullptr, nullptr};
    int plane_index[4] {0, 0, 0, 0};
    int engine_count {0};
    int copy_count {0};
    EngineParams* ep {nullptr};
    std::shared_ptr<neo_fft3d::fft::FFTBackend> fft_backend;
    std::unique_ptr<FetchFrameFunctor> fetch_frame_func;
    int fft_threads {2};
    bool mt {false};
    bool crop {false};
  };

  static ds::Result<ds::VideoInitStateResult<State>> init(ds::VideoInitContext& context);
  static ds::Result<ds::VideoRequestResult> request(ds::VideoRequestContext& context);
  static ds::Result<ds::VideoProcessResult> process(ds::VideoProcessContext& context);
  static int cache_hints(ds::VideoCacheHintsContext& context);
};

struct FFT3DBridge : ds::SingleInputVideoBridgeDefaults<FFT3DCore> {
  static constexpr const char* vs_name = "FFT3D";
  static constexpr const char* avs_name = "neo_fft3d";
  static constexpr const char* vs_signature = "";
  static constexpr const char* avs_signature = "";
  static constexpr const char* missing_input_error = "neo_fft3d: missing required video clip";
  static constexpr const char* vs_format_error =
    "neo_fft3d: only constant 8-16 bit integer and 32 bit float video is supported";
  static constexpr const char* avs_format_error =
    "neo_fft3d: only constant 8-16 bit integer and 32 bit float video is supported";
  static constexpr ds::avisynth::MtMode avs_mt_mode = ds::avisynth::MtMode::NiceFilter;

  static bool accepts_video_format(ds::VideoFormat format);
  static ds::FilterDescriptor descriptor();
};

namespace Plugin {
inline constexpr const char* Identifier = "in.7086.neo_fft3d";
inline constexpr const char* Namespace = "neo_fft3d";
inline constexpr const char* Description = "Neo FFT3D Filter " PLUGIN_VERSION;
} // namespace Plugin

} // namespace neo_fft3d
