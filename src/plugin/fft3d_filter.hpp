#pragma once

#include <dualsynth/avisynth/video_bridge.hpp>
#include <dualsynth/format.hpp>
#include <dualsynth/param.hpp>
#include <dualsynth/video_bridge.hpp>
#include <dualsynth/video_filter.hpp>

#include "version.hpp"
#include "engine/engine_params.hpp"
#include "fft/fft_backend.hpp"

#include <array>
#include <memory>

class FFT3DEngine;

namespace neo_fft3d {

enum class PlaneAction : int {
  None = 1,
  Copy = 2,
  Process = 3
};

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

    std::array<PlaneAction, 4> process {
      PlaneAction::Copy,
      PlaneAction::Copy,
      PlaneAction::Copy,
      PlaneAction::Copy
    };
    std::array<std::unique_ptr<FFT3DEngine>, 4> engine {};
    int engine_count {0};
    std::unique_ptr<EngineParams> ep;
    std::shared_ptr<neo_fft3d::fft::FFTBackend> fft_backend;
    int fft_threads {2};
    bool mt {false};
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
