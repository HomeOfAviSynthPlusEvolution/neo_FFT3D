#include "plugin/fft3d_filter.hpp"

#include "engine/fft3d_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace neo_fft3d {

namespace {
ds::Error invalid_argument(std::string message) {
  return ds::Error{ds::ErrorCode::InvalidArgument, std::move(message)};
}

template <class T>
T get_param_val(const ds::Result<T>& res) {
  if (!res.has_value()) {
    throw std::runtime_error("Parameter parsing error: " + res.error().message);
  }
  return res.value();
}

float get_param_val(const ds::Result<double>& res, float /* fallback */) {
  if (!res.has_value()) {
    throw std::runtime_error("Parameter parsing error: " + res.error().message);
  }
  return static_cast<float>(res.value());
}

EngineVideoInfo make_engine_video_info(const ds::VideoInputInfo& input) {
  EngineVideoInfo vi;
  vi.Width = input.width;
  vi.Height = input.height;
  vi.Frames = input.num_frames;
  vi.FPSNum = static_cast<int>(input.fps.numerator);
  vi.FPSDen = static_cast<int>(input.fps.denominator);
  vi.Format.IsFamilyYUV = (input.format.color_family == ds::ColorFamily::Yuv);
  vi.Format.IsFamilyRGB = (input.format.color_family == ds::ColorFamily::Rgb);
  vi.Format.IsFamilyGray = (input.format.color_family == ds::ColorFamily::Gray);
  vi.Format.Planes = input.format.plane_count;
  vi.Format.BytesPerSample = ds::bytes_per_sample(input.format.sample_format);
  vi.Format.BitsPerSample = ds::bits_per_sample(input.format.sample_format);
  vi.Format.SSW = input.format.subsampling_w;
  vi.Format.SSH = input.format.subsampling_h;
  return vi;
}

EngineParams make_default_engine_params(float default_sigma, EngineVideoInfo vi) {
  EngineParams ep {};
  ep.sigma = default_sigma;
  ep.beta = 1.0f;
  ep.bw = 32;
  ep.bh = 32;
  ep.bt = 3;
  ep.ow = -1;
  ep.oh = -1;
  ep.kratio = 2.0f;
  ep.sharpen = 0.0f;
  ep.scutoff = 0.3f;
  ep.svr = 1.0f;
  ep.smin = 4.0f;
  ep.smax = 20.0f;
  ep.measure = true;
  ep.interlaced = false;
  ep.wintype = 0;
  ep.pframe = 0;
  ep.px = 0;
  ep.py = 0;
  ep.pshow = false;
  ep.pcutoff = 0.1f;
  ep.pfactor = 0.0f;
  ep.sigma2 = default_sigma;
  ep.sigma3 = default_sigma;
  ep.sigma4 = default_sigma;
  ep.degrid = 1.0f;
  ep.dehalo = 0.0f;
  ep.hr = 2.0f;
  ep.ht = 50.0f;
  ep.l = 0;
  ep.t = 0;
  ep.r = 0;
  ep.b = 0;
  ep.opt = 0;
  ep.vi = vi;
  return ep;
}

PlaneAction plane_action_from_legacy(int value) {
  return static_cast<PlaneAction>(value);
}
} // namespace

FFT3DCore::State::State() = default;
FFT3DCore::State::~State() = default;
FFT3DCore::State::State(State&& old) noexcept = default;
FFT3DCore::State& FFT3DCore::State::operator=(State&& old) noexcept = default;

ds::Result<ds::VideoInitStateResult<FFT3DCore::State>> FFT3DCore::init(
  ds::VideoInitContext& context
) {
  try {
    auto has_param = [](const ds::ParamValues& params, const char* name) {
      return std::any_of(params.entries.begin(), params.entries.end(), [name](const ds::ParamEntry& entry) {
        return entry.name == name;
      });
    };

    const auto inputs = ds::collect_video_input_infos<FFT3DCore>(context.inputs);
    if (!inputs.has_value()) {
      return ds::Result<ds::VideoInitStateResult<State>>::failure(inputs.error());
    }
    const ds::VideoInputInfo& input = inputs.value()[0];

    State state;

    EngineVideoInfo in_vi = make_engine_video_info(input);

    float default_sigma = get_param_val(context.params->get_double("sigma", 2.0));

    state.ep = std::make_unique<EngineParams>(make_default_engine_params(default_sigma, in_vi));

    state.ep->beta = get_param_val(context.params->get_double("beta", state.ep->beta), state.ep->beta);
    state.ep->bw = get_param_val(context.params->get_int("bw", state.ep->bw));
    state.ep->bh = get_param_val(context.params->get_int("bh", state.ep->bh));
    state.ep->bt = get_param_val(context.params->get_int("bt", state.ep->bt));
    state.ep->ow = get_param_val(context.params->get_int("ow", state.ep->ow));
    state.ep->oh = get_param_val(context.params->get_int("oh", state.ep->oh));
    state.ep->kratio = get_param_val(context.params->get_double("kratio", state.ep->kratio), state.ep->kratio);
    state.ep->sharpen = get_param_val(context.params->get_double("sharpen", state.ep->sharpen), state.ep->sharpen);
    state.ep->scutoff = get_param_val(context.params->get_double("scutoff", state.ep->scutoff), state.ep->scutoff);
    state.ep->svr = get_param_val(context.params->get_double("svr", state.ep->svr), state.ep->svr);
    state.ep->smin = get_param_val(context.params->get_double("smin", state.ep->smin), state.ep->smin);
    state.ep->smax = get_param_val(context.params->get_double("smax", state.ep->smax), state.ep->smax);
    state.ep->measure = get_param_val(context.params->get_bool("measure", state.ep->measure));
    state.ep->interlaced = get_param_val(context.params->get_bool("interlaced", state.ep->interlaced));
    state.ep->wintype = get_param_val(context.params->get_int("wintype", state.ep->wintype));

    int temp_pframe = get_param_val(context.params->get_int("pframe", state.ep->pframe));
    if (in_vi.Frames > 0) {
      if (temp_pframe < 0) {
        temp_pframe = 0;
      }
      if (temp_pframe >= in_vi.Frames) {
        temp_pframe = in_vi.Frames - 1;
      }
    } else {
      temp_pframe = 0;
    }
    state.ep->pframe = temp_pframe;

    state.ep->px = get_param_val(context.params->get_int("px", state.ep->px));
    state.ep->py = get_param_val(context.params->get_int("py", state.ep->py));
    state.ep->pshow = get_param_val(context.params->get_bool("pshow", state.ep->pshow));
    state.ep->pcutoff = get_param_val(context.params->get_double("pcutoff", state.ep->pcutoff), state.ep->pcutoff);
    state.ep->pfactor = get_param_val(context.params->get_double("pfactor", state.ep->pfactor), state.ep->pfactor);
    state.ep->sigma2 = get_param_val(context.params->get_double("sigma2", state.ep->sigma2), state.ep->sigma2);
    state.ep->sigma3 = get_param_val(context.params->get_double("sigma3", state.ep->sigma3), state.ep->sigma3);
    state.ep->sigma4 = get_param_val(context.params->get_double("sigma4", state.ep->sigma4), state.ep->sigma4);
    state.ep->degrid = get_param_val(context.params->get_double("degrid", state.ep->degrid), state.ep->degrid);
    state.ep->dehalo = get_param_val(context.params->get_double("dehalo", state.ep->dehalo), state.ep->dehalo);
    state.ep->hr = get_param_val(context.params->get_double("hr", state.ep->hr), state.ep->hr);
    state.ep->ht = get_param_val(context.params->get_double("ht", state.ep->ht), state.ep->ht);

    state.ep->l = (std::max)(get_param_val(context.params->get_int("l", state.ep->l)), 0);
    state.ep->t = (std::max)(get_param_val(context.params->get_int("t", state.ep->t)), 0);
    state.ep->r = (std::max)(get_param_val(context.params->get_int("r", state.ep->r)), 0);
    state.ep->b = (std::max)(get_param_val(context.params->get_int("b", state.ep->b)), 0);
    state.ep->opt = get_param_val(context.params->get_int("opt", state.ep->opt));

    std::string fft_backend_name = "fftw";
    bool explicit_backend = false;
    if (has_param(*context.params, "fft_backend")) {
      fft_backend_name = get_param_val(context.params->get_string("fft_backend", "fftw"));
      explicit_backend = true;
    }

    if (state.ep->l + state.ep->r >= state.ep->vi.Width) {
      throw "Width cannot be cropped to zero or below";
    }
    if (state.ep->t + state.ep->b >= state.ep->vi.Height) {
      throw "Height cannot be cropped to zero or below";
    }

    state.process.fill(PlaneAction::Copy);

    auto read_vs_planes = [&]() {
      std::vector<std::int64_t> user_planes;
      auto val_planes = get_param_val(context.params->get_int_array("planes", user_planes));
      if (val_planes.empty()) {
        state.process[0] = state.process[1] = state.process[2] = PlaneAction::Process;
      } else {
        for (auto&& p : val_planes) {
          if (p >= 0 && p < state.ep->vi.Format.Planes) {
            state.process[p] = PlaneAction::Process;
          } else {
            throw "planes: plane index out of bounds";
          }
        }
      }
    };
    auto read_avs_yuv = [&]() {
      state.process[0] = plane_action_from_legacy(get_param_val(context.params->get_int("y", 3)));
      state.process[1] = plane_action_from_legacy(get_param_val(context.params->get_int("u", 3)));
      state.process[2] = plane_action_from_legacy(get_param_val(context.params->get_int("v", 3)));
    };

    switch (context.host) {
    case ds::HostKind::VapourSynth:
      read_vs_planes();
      break;
    case ds::HostKind::AviSynth:
      read_avs_yuv();
      break;
    case ds::HostKind::Unknown:
      if (has_param(*context.params, "planes")) {
        read_vs_planes();
      } else {
        read_avs_yuv();
      }
      break;
    }

    state.mt = get_param_val(context.params->get_bool("mt", state.mt));
    state.fft_threads = get_param_val(context.params->get_int("ncpu", state.fft_threads));
    if (state.fft_threads < 1) {
      state.fft_threads = 1;
    }

    if (fft_backend_name == "fftw") {
      state.fft_backend = fft::CreateFFTWBackend();
      if (!state.fft_backend->Load()) {
        if (explicit_backend) {
          throw std::runtime_error("fft_backend: failed to load fftw library");
        } else {
          state.fft_backend = fft::CreatePocketFFTBackend();
          state.fft_backend->Load();
        }
      }
    } else if (fft_backend_name == "pocketfft") {
      state.fft_backend = fft::CreatePocketFFTBackend();
      state.fft_backend->Load();
    } else {
      throw std::runtime_error("fft_backend must be 'fftw' or 'pocketfft'");
    }

    if (state.fft_threads > 1 && state.fft_backend->HasThreading()) {
      state.fft_backend->SetThreadCount(state.fft_threads);
    }

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      if (state.process[i] == PlaneAction::Process) {
        state.ep->IsChroma = state.ep->vi.Format.IsFamilyYUV && i != 0;
        state.ep->framewidth = state.ep->IsChroma ? state.ep->vi.Width >> state.ep->vi.Format.SSW : state.ep->vi.Width;
        state.ep->frameheight = state.ep->IsChroma ? state.ep->vi.Height >> state.ep->vi.Format.SSH : state.ep->vi.Height;
        state.engine[i] = std::make_unique<FFT3DEngine>(*state.ep, i, state.fft_backend);
        state.engine_count++;
      }
    }

    return ds::Result<ds::VideoInitStateResult<State>>::success(
      ds::VideoInitStateResult<State>{
        ds::VideoOutputInfo{input.width, input.height, input.num_frames, input.format, input.fps},
        std::move(state)
      }
    );
  } catch (const char* error) {
    return ds::Result<ds::VideoInitStateResult<State>>::failure(invalid_argument(error));
  } catch (const std::exception& error) {
    return ds::Result<ds::VideoInitStateResult<State>>::failure(invalid_argument(error.what()));
  } catch (...) {
    return ds::Result<ds::VideoInitStateResult<State>>::failure(
      invalid_argument("neo_fft3d: unhandled initialization error")
    );
  }
}

ds::Result<ds::VideoRequestResult> FFT3DCore::request(ds::VideoRequestContext& context) {
  try {
    auto& state = context.state<State>();
    int n = context.output_frame;

    if (state.ep->bt > 1 && state.engine_count > 0) {
      int from = (std::max)(n - state.ep->bt / 2, 0);
      int to = (std::min)(n + (state.ep->bt - 1) / 2, state.ep->vi.Frames - 1);
      for (int i = from; i <= to; i++) {
        context.request_frame(0, i);
      }
    } else {
      context.request_frame(0, n);
    }

    if (state.ep->pfactor != 0.0f) {
      context.request_frame(0, state.ep->pframe);
    }

    return ds::Result<ds::VideoRequestResult>::success(ds::VideoRequestResult{});
  } catch (const std::exception& error) {
    return ds::Result<ds::VideoRequestResult>::failure(invalid_argument(error.what()));
  }
}

static void copy_plane_pixels(
  const ds::VideoFrameView& src,
  ds::MutableVideoFrameView& dst,
  int plane,
  int bytes_per_sample
) {
  const auto& src_plane = src.plane(plane);
  auto& dst_plane = dst.plane(plane);
  const auto row_bytes = static_cast<std::size_t>(src_plane.width) * static_cast<std::size_t>(bytes_per_sample);
  auto src_ptr = static_cast<const std::uint8_t*>(src_plane.data);
  auto dst_ptr = static_cast<std::uint8_t*>(dst_plane.data);

  for (int y = 0; y < src_plane.height; ++y) {
    std::memcpy(dst_ptr, src_ptr, row_bytes);
    src_ptr += src_plane.stride_bytes;
    dst_ptr += dst_plane.stride_bytes;
  }
}

static bool has_crop(const EngineParams& ep) {
  return ep.l > 0 || ep.r > 0 || ep.t > 0 || ep.b > 0;
}

ds::Result<ds::VideoProcessResult> FFT3DCore::process(ds::VideoProcessContext& context) {
  try {
    auto& state = context.state<State>();
    int n = context.output_frame;

    auto src_res = context.frames.get(0, n);
    if (!src_res.has_value()) {
      return ds::Result<ds::VideoProcessResult>::failure(invalid_argument("Failed to get source frame"));
    }
    const ds::VideoFrameView& src = src_res.value().frame;

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      if (state.process[i] == PlaneAction::Process && state.engine[i]) {
        if (state.ep->bt > 1) {
          int from = (std::max)(n - state.ep->bt / 2, 0);
          int to = (std::min)(n + (state.ep->bt - 1) / 2, state.ep->vi.Frames - 1);
          for (int f = from; f <= to; f++) {
            state.engine[i]->CacheRefresh(f);
          }
        }
      }
    }

    auto core_process = [&](int i) {
      if (state.process[i] == PlaneAction::Process) {
        if (has_crop(*state.ep)) {
          copy_plane_pixels(src, context.dst, i, state.ep->vi.Format.BytesPerSample);
        }
        state.engine[i]->ProcessFrame(n, context.frames, context.dst);
      } else if (state.process[i] == PlaneAction::Copy) {
        copy_plane_pixels(src, context.dst, i, state.ep->vi.Format.BytesPerSample);
      }
    };

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      core_process(i);
    }

    return ds::Result<ds::VideoProcessResult>::success(ds::VideoProcessResult{});
  } catch (const char* error) {
    return ds::Result<ds::VideoProcessResult>::failure(invalid_argument(error));
  } catch (const std::exception& error) {
    return ds::Result<ds::VideoProcessResult>::failure(invalid_argument(error.what()));
  } catch (...) {
    return ds::Result<ds::VideoProcessResult>::failure(
      invalid_argument("neo_fft3d: unhandled processing error")
    );
  }
}

int FFT3DCore::cache_hints(ds::VideoCacheHintsContext& context) {
  auto& state = context.state<State>();
  if (context.cachehints == CACHE_GET_MTMODE) {
    return static_cast<int>(state.ep->bt == 0 ? ds::avisynth::MtMode::Serialized : ds::avisynth::MtMode::NiceFilter);
  }
  return 0;
}

bool FFT3DBridge::accepts_video_format(ds::VideoFormat format) {
  return format.plane_count >= 1 &&
    format.plane_count <= 4 &&
    (format.sample_format == ds::SampleFormat::UInt8 ||
     format.sample_format == ds::SampleFormat::UInt10 ||
     format.sample_format == ds::SampleFormat::UInt12 ||
     format.sample_format == ds::SampleFormat::UInt14 ||
     format.sample_format == ds::SampleFormat::UInt16 ||
     format.sample_format == ds::SampleFormat::Float32);
}

ds::FilterDescriptor FFT3DBridge::descriptor() {
  return ds::FilterDescriptor{
    .name = "FFT3D",
    .params = {
      ds::ParamSpec{.name = "clip", .type = ds::ParamType::Clip, .required = true},
      ds::ParamSpec{.name = "sigma", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "beta", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "fft_backend", .type = ds::ParamType::String},
      ds::ParamSpec{.name = "planes", .type = ds::ParamType::Integer, .is_array = true, .avs_enabled = false},
      ds::ParamSpec{.name = "bw", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "bh", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "bt", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "ow", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "oh", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "kratio", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "sharpen", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "scutoff", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "svr", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "smin", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "smax", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "measure", .type = ds::ParamType::Boolean},
      ds::ParamSpec{.name = "interlaced", .type = ds::ParamType::Boolean},
      ds::ParamSpec{.name = "wintype", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "pframe", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "px", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "py", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "pshow", .type = ds::ParamType::Boolean},
      ds::ParamSpec{.name = "pcutoff", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "pfactor", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "sigma2", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "sigma3", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "sigma4", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "degrid", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "dehalo", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "hr", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "ht", .type = ds::ParamType::Float},
      ds::ParamSpec{.name = "y", .type = ds::ParamType::Integer, .vs_enabled = false},
      ds::ParamSpec{.name = "u", .type = ds::ParamType::Integer, .vs_enabled = false},
      ds::ParamSpec{.name = "v", .type = ds::ParamType::Integer, .vs_enabled = false},
      ds::ParamSpec{.name = "l", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "t", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "r", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "b", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "opt", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "ncpu", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "mt", .type = ds::ParamType::Boolean}
    }
  };
}

} // namespace neo_fft3d
