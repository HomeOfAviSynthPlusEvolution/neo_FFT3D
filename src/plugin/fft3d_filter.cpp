#include "plugin/fft3d_filter.hpp"

#include "dualsynth_compat.hpp"
#include "fft3d_engine.h"
#include "fftwlite.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

DSFrame::DSFrame(const ds::VideoFrameView& vsView) {
  Format.IsFamilyYUV = (vsView.format.color_family == ds::ColorFamily::Yuv);
  Format.IsFamilyRGB = (vsView.format.color_family == ds::ColorFamily::Rgb);
  Format.IsFamilyGray = (vsView.format.color_family == ds::ColorFamily::Gray);
  Format.Planes = vsView.plane_count;
  Format.BytesPerSample = ds::bytes_per_sample(vsView.format.sample_format);
  Format.BitsPerSample = ds::bits_per_sample(vsView.format.sample_format);
  Format.SSW = vsView.format.subsampling_w;
  Format.SSH = vsView.format.subsampling_h;

  FrameWidth = vsView.plane(0).width;
  FrameHeight = vsView.plane(0).height;

  SrcPointers = new const unsigned char*[Format.Planes];
  StrideBytes = new int[Format.Planes];
  for (int i = 0; i < Format.Planes; i++) {
    SrcPointers[i] = static_cast<const unsigned char*>(vsView.plane(i).data);
    StrideBytes[i] = static_cast<int>(vsView.plane(i).stride_bytes);
  }
}

DSFrame::DSFrame(const ds::MutableVideoFrameView& vsView) {
  Format.IsFamilyYUV = (vsView.format.color_family == ds::ColorFamily::Yuv);
  Format.IsFamilyRGB = (vsView.format.color_family == ds::ColorFamily::Rgb);
  Format.IsFamilyGray = (vsView.format.color_family == ds::ColorFamily::Gray);
  Format.Planes = vsView.plane_count;
  Format.BytesPerSample = ds::bytes_per_sample(vsView.format.sample_format);
  Format.BitsPerSample = ds::bits_per_sample(vsView.format.sample_format);
  Format.SSW = vsView.format.subsampling_w;
  Format.SSH = vsView.format.subsampling_h;

  FrameWidth = vsView.plane(0).width;
  FrameHeight = vsView.plane(0).height;

  SrcPointers = new const unsigned char*[Format.Planes];
  DstPointers = new unsigned char*[Format.Planes];
  StrideBytes = new int[Format.Planes];
  for (int i = 0; i < Format.Planes; i++) {
    SrcPointers[i] = static_cast<const unsigned char*>(vsView.plane(i).data);
    DstPointers[i] = static_cast<unsigned char*>(vsView.plane(i).data);
    StrideBytes[i] = static_cast<int>(vsView.plane(i).stride_bytes);
  }
}

namespace neo_fft3d {

namespace {
ds::Error invalid_argument(std::string message) {
  return ds::Error{ds::ErrorCode::InvalidArgument, std::move(message)};
}

template <class T>
T get_param_val(const ds::Result<T>& res, T fallback) {
  if (res.has_value()) return res.value();
  return fallback;
}

float get_param_val(const ds::Result<double>& res, float fallback) {
  if (res.has_value()) return static_cast<float>(res.value());
  return fallback;
}
} // namespace

FFT3DCore::State::State() {
  fftfp = std::make_shared<FFTFunctionPointers>();
  fetch_frame_func = std::make_unique<FetchFrameFunctor>();
}

FFT3DCore::State::~State() {
  for (int i = 0; i < 4; i++) {
    delete engine[i];
  }
  delete ep;
  if (fftfp) {
    fftfp->free();
  }
}

FFT3DCore::State::State(State&& old) noexcept {
  *this = std::move(old);
}

FFT3DCore::State& FFT3DCore::State::operator=(State&& old) noexcept {
  if (this != &old) {
    for (int i = 0; i < 4; i++) {
      delete engine[i];
      engine[i] = old.engine[i];
      old.engine[i] = nullptr;
    }
    delete ep;
    ep = old.ep;
    old.ep = nullptr;

    std::memcpy(process, old.process, sizeof(process));
    std::memcpy(plane_index, old.plane_index, sizeof(plane_index));
    engine_count = old.engine_count;
    copy_count = old.copy_count;
    fftfp = std::move(old.fftfp);
    fetch_frame_func = std::move(old.fetch_frame_func);
    fft_threads = old.fft_threads;
    mt = old.mt;
    crop = old.crop;
  }
  return *this;
}

ds::Result<ds::VideoInitStateResult<FFT3DCore::State>> FFT3DCore::init(
  ds::VideoInitContext& context
) {
  try {
    const auto inputs = ds::collect_video_input_infos<FFT3DCore>(context.inputs);
    if (!inputs.has_value()) {
      return ds::Result<ds::VideoInitStateResult<State>>::failure(inputs.error());
    }
    const ds::VideoInputInfo& input = inputs.value()[0];

    State state;

    DSVideoInfo in_vi;
    in_vi.Width = input.width;
    in_vi.Height = input.height;
    in_vi.Frames = input.num_frames;
    in_vi.FPSNum = input.fps.numerator;
    in_vi.FPSDen = input.fps.denominator;
    in_vi.Format.IsFamilyYUV = (input.format.color_family == ds::ColorFamily::Yuv);
    in_vi.Format.IsFamilyRGB = (input.format.color_family == ds::ColorFamily::Rgb);
    in_vi.Format.IsFamilyGray = (input.format.color_family == ds::ColorFamily::Gray);
    in_vi.Format.Planes = input.format.plane_count;
    in_vi.Format.BytesPerSample = ds::bytes_per_sample(input.format.sample_format);
    in_vi.Format.BitsPerSample = ds::bits_per_sample(input.format.sample_format);
    in_vi.Format.SSW = input.format.subsampling_w;
    in_vi.Format.SSH = input.format.subsampling_h;

    float default_sigma = get_param_val(context.params->get_double("sigma", 2.0), 2.0f);

    state.ep = new EngineParams {
      default_sigma, 1.0f,
      32, 32,
      3,
      -1, -1,
      2.0f,
      0.0f,
      0.3f,
      1.0f,
      4.0f, 20.0f,
      true,
      false,
      0,
      0,
      0, 0,
      false,
      0.1f,
      0.0f,
      default_sigma, default_sigma, default_sigma,
      1.0f,
      0.0f,
      2.0f, 50.0f,
      0, 0, 0, 0,
      0,
      in_vi,
      nullptr, // avs_env
      false // IsAVS12
    };

    state.ep->beta = get_param_val(context.params->get_double("beta", state.ep->beta), state.ep->beta);
    state.ep->bw = get_param_val(context.params->get_int("bw", state.ep->bw), state.ep->bw);
    state.ep->bh = get_param_val(context.params->get_int("bh", state.ep->bh), state.ep->bh);
    state.ep->bt = get_param_val(context.params->get_int("bt", state.ep->bt), state.ep->bt);
    state.ep->ow = get_param_val(context.params->get_int("ow", state.ep->ow), state.ep->ow);
    state.ep->oh = get_param_val(context.params->get_int("oh", state.ep->oh), state.ep->oh);
    state.ep->kratio = get_param_val(context.params->get_double("kratio", state.ep->kratio), state.ep->kratio);
    state.ep->sharpen = get_param_val(context.params->get_double("sharpen", state.ep->sharpen), state.ep->sharpen);
    state.ep->scutoff = get_param_val(context.params->get_double("scutoff", state.ep->scutoff), state.ep->scutoff);
    state.ep->svr = get_param_val(context.params->get_double("svr", state.ep->svr), state.ep->svr);
    state.ep->smin = get_param_val(context.params->get_double("smin", state.ep->smin), state.ep->smin);
    state.ep->smax = get_param_val(context.params->get_double("smax", state.ep->smax), state.ep->smax);
    state.ep->measure = get_param_val(context.params->get_bool("measure", state.ep->measure), state.ep->measure);
    state.ep->interlaced = get_param_val(context.params->get_bool("interlaced", state.ep->interlaced), state.ep->interlaced);
    state.ep->wintype = get_param_val(context.params->get_int("wintype", state.ep->wintype), state.ep->wintype);
    state.ep->pframe = get_param_val(context.params->get_int("pframe", state.ep->pframe), state.ep->pframe);
    state.ep->px = get_param_val(context.params->get_int("px", state.ep->px), state.ep->px);
    state.ep->py = get_param_val(context.params->get_int("py", state.ep->py), state.ep->py);
    state.ep->pshow = get_param_val(context.params->get_bool("pshow", state.ep->pshow), state.ep->pshow);
    state.ep->pcutoff = get_param_val(context.params->get_double("pcutoff", state.ep->pcutoff), state.ep->pcutoff);
    state.ep->pfactor = get_param_val(context.params->get_double("pfactor", state.ep->pfactor), state.ep->pfactor);
    state.ep->sigma2 = get_param_val(context.params->get_double("sigma2", state.ep->sigma2), state.ep->sigma2);
    state.ep->sigma3 = get_param_val(context.params->get_double("sigma3", state.ep->sigma3), state.ep->sigma3);
    state.ep->sigma4 = get_param_val(context.params->get_double("sigma4", state.ep->sigma4), state.ep->sigma4);
    state.ep->degrid = get_param_val(context.params->get_double("degrid", state.ep->degrid), state.ep->degrid);
    state.ep->dehalo = get_param_val(context.params->get_double("dehalo", state.ep->dehalo), state.ep->dehalo);
    state.ep->hr = get_param_val(context.params->get_double("hr", state.ep->hr), state.ep->hr);
    state.ep->ht = get_param_val(context.params->get_double("ht", state.ep->ht), state.ep->ht);

    state.ep->l = (std::max)(get_param_val(context.params->get_int("l", state.ep->l), state.ep->l), 0);
    state.ep->t = (std::max)(get_param_val(context.params->get_int("t", state.ep->t), state.ep->t), 0);
    state.ep->r = (std::max)(get_param_val(context.params->get_int("r", state.ep->r), state.ep->r), 0);
    state.ep->b = (std::max)(get_param_val(context.params->get_int("b", state.ep->b), state.ep->b), 0);
    state.ep->opt = get_param_val(context.params->get_int("opt", state.ep->opt), state.ep->opt);

    state.crop = state.ep->l > 0 || state.ep->r > 0 || state.ep->t > 0 || state.ep->b > 0;

    if (state.ep->l + state.ep->r >= state.ep->vi.Width) {
      throw "Width cannot be cropped to zero or below";
    }
    if (state.ep->t + state.ep->b >= state.ep->vi.Height) {
      throw "Height cannot be cropped to zero or below";
    }

    state.process[0] = state.process[1] = state.process[2] = state.process[3] = 2;
    std::vector<std::int64_t> user_planes;
    auto res_planes = context.params->get_int_array("planes", user_planes);
    if (res_planes.has_value() && !res_planes.value().empty()) {
      state.process[0] = state.process[1] = state.process[2] = state.process[3] = 2;
      for (auto&& p : res_planes.value()) {
        if (p < state.ep->vi.Format.Planes) {
          state.process[p] = 3;
        }
      }
    } else {
      state.process[0] = get_param_val(context.params->get_int("y", 3), 3);
      state.process[1] = get_param_val(context.params->get_int("u", 3), 3);
      state.process[2] = get_param_val(context.params->get_int("v", 3), 3);
    }

    state.mt = get_param_val(context.params->get_bool("mt", state.mt), state.mt);
    state.fft_threads = get_param_val(context.params->get_int("ncpu", state.fft_threads), state.fft_threads);
    if (state.fft_threads < 1) {
      state.fft_threads = 1;
    }

    state.fftfp->load();

    if (state.fft_threads > 1 && state.fftfp->has_threading()) {
      state.fftfp->fftwf_init_threads();
      state.fftfp->fftwf_plan_with_nthreads(state.fft_threads);
    }

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A };
    int* planes = (state.ep->vi.Format.IsFamilyYUV) ? planes_y : planes_r;

    auto fetch_frame_bridge = [](void* opaque, int frame_num) -> DSFrame {
      auto* ctx = static_cast<ds::VideoProcessContext*>(opaque);
      auto res = ctx->frames.get(0, frame_num);
      if (!res.has_value()) {
        return DSFrame();
      }
      return DSFrame(res.value().frame);
    };
    state.fetch_frame_func->Fn = fetch_frame_bridge;
    state.fetch_frame_func->Opaque = nullptr;

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      state.plane_index[i] = planes[i];
      if (state.process[i] == 3) {
        state.ep->IsChroma = state.ep->vi.Format.IsFamilyYUV && i != 0;
        state.ep->framewidth = state.ep->IsChroma ? state.ep->vi.Width >> state.ep->vi.Format.SSW : state.ep->vi.Width;
        state.ep->frameheight = state.ep->IsChroma ? state.ep->vi.Height >> state.ep->vi.Format.SSH : state.ep->vi.Height;
        state.engine[i] = new FFT3DEngine(*state.ep, i, state.fetch_frame_func.get(), *state.fftfp);
        state.engine_count++;
      } else if (state.process[i] == 2) {
        state.copy_count++;
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
    return ds::Result<ds::VideoRequestResult>::success(ds::VideoRequestResult{});
  } catch (const std::exception& error) {
    return ds::Result<ds::VideoRequestResult>::failure(invalid_argument(error.what()));
  }
}

template<bool slow>
static void copy_frame(DSFrame& dst, DSFrame& src, int plane, bool chroma, EngineParams* ep) {
  auto height = ep->vi.Height;
  auto src_stride = src.StrideBytes[plane];
  auto dst_stride = dst.StrideBytes[plane];
  auto src_ptr = src.SrcPointers[plane];
  auto dst_ptr = dst.DstPointers[plane];
  if (chroma) {
    height >>= ep->vi.Format.SSH;
  }
  if constexpr (slow) {
    for (int i = 0; i < height; i++) {
      std::memcpy(dst_ptr, src_ptr, dst_stride);
      src_ptr += src_stride;
      dst_ptr += dst_stride;
    }
  } else {
    std::memcpy(dst_ptr, src_ptr, dst_stride * height);
  }
}

template<bool slow>
static void copy_frame(DSFrame& dst, DSFrame& src, DSFrame& processed, int plane, bool chroma, int l, int t, int r, int b, EngineParams* ep) {
  auto width = ep->vi.Width * ep->vi.Format.BytesPerSample;
  auto height = ep->vi.Height;
  auto src_stride = src.StrideBytes[plane];
  auto dst_stride = dst.StrideBytes[plane];
  auto pcs_stride = processed.StrideBytes[plane];
  auto src_ptr = src.SrcPointers[plane];
  auto dst_ptr = dst.DstPointers[plane];
  auto pcs_ptr = processed.SrcPointers[plane];
  if (chroma) {
    width >>= ep->vi.Format.SSW;
    height >>= ep->vi.Format.SSH;
  }
  auto left_b = l * ep->vi.Format.BytesPerSample;
  auto right_b = r * ep->vi.Format.BytesPerSample;
  if constexpr (slow) {
    for (int i = 0; i < height; i++) {
      if (i < t || i >= height - b) {
        std::memcpy(dst_ptr, src_ptr, dst_stride);
      } else {
        if (l > 0) {
          std::memcpy(dst_ptr, src_ptr, left_b);
        }
        std::memcpy(dst_ptr + left_b, pcs_ptr + left_b, width - left_b - right_b);
        if (r > 0) {
          std::memcpy(dst_ptr + width - right_b, src_ptr + width - right_b, right_b);
        }
      }
      src_ptr += src_stride;
      dst_ptr += dst_stride;
      pcs_ptr += pcs_stride;
    }
  } else {
    if (l > 0 || r > 0 || t > 0 || b > 0) {
      for (int i = 0; i < height; i++) {
        if (i < t || i >= height - b) {
          std::memcpy(dst_ptr, src_ptr, dst_stride);
        } else {
          if (l > 0) {
            std::memcpy(dst_ptr, src_ptr, left_b);
          }
          std::memcpy(dst_ptr + left_b, pcs_ptr + left_b, width - left_b - right_b);
          if (r > 0) {
            std::memcpy(dst_ptr + width - right_b, src_ptr + width - right_b, right_b);
          }
        }
        src_ptr += src_stride;
        dst_ptr += dst_stride;
        pcs_ptr += pcs_stride;
      }
    } else {
      std::memcpy(dst_ptr, pcs_ptr, dst_stride * height);
    }
  }
}

ds::Result<ds::VideoProcessResult> FFT3DCore::process(ds::VideoProcessContext& context) {
  try {
    auto& state = context.state<State>();
    int n = context.output_frame;

    auto src_res = context.frames.get(0, n);
    if (!src_res.has_value()) {
      return ds::Result<ds::VideoProcessResult>::failure(invalid_argument("Failed to get source frame"));
    }
    DSFrame src = DSFrame(src_res.value().frame);
    DSFrame dst = DSFrame(context.dst);

    state.fetch_frame_func->Opaque = &context;

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      if (state.process[i] == 3 && state.engine[i]) {
        if (state.ep->bt > 1) {
          int from = (std::max)(n - state.ep->bt / 2, 0);
          int to = (std::min)(n + (state.ep->bt - 1) / 2, state.ep->vi.Frames - 1);
          for (int f = from; f <= to; f++) {
            state.engine[i]->CacheRefresh(f);
          }
        }
      }
    }

    std::unordered_map<int, DSFrame> empty_map;

    if (state.engine_count == 1 && state.copy_count == 0 && !state.crop) {
      for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
        if (state.process[i] == 3) {
          DSFrame frame = state.engine[i]->GetFrame(n, empty_map);
          copy_frame<false>(dst, src, frame, i, false, 0, 0, 0, 0, state.ep);
          return ds::Result<ds::VideoProcessResult>::success(ds::VideoProcessResult{});
        }
      }
    }

    auto core_process = [&](int i) {
      bool chroma = state.ep->vi.Format.IsFamilyYUV && i > 0 && i < 3;

      if (state.process[i] == 3) {
        auto l = chroma ? (state.ep->l >> state.ep->vi.Format.SSW) : state.ep->l;
        auto r = chroma ? (state.ep->r >> state.ep->vi.Format.SSW) : state.ep->r;
        auto t = chroma ? (state.ep->t >> state.ep->vi.Format.SSH) : state.ep->t;
        auto b = chroma ? (state.ep->b >> state.ep->vi.Format.SSH) : state.ep->b;
        DSFrame frame = state.engine[i]->GetFrame(n, empty_map);
        if (src.StrideBytes[i] == dst.StrideBytes[i]) {
          copy_frame<false>(dst, src, frame, i, chroma, l, t, r, b, state.ep);
        } else {
          copy_frame<true>(dst, src, frame, i, chroma, l, t, r, b, state.ep);
        }
      } else if (state.process[i] == 2) {
        if (src.StrideBytes[i] == dst.StrideBytes[i]) {
          copy_frame<false>(dst, src, i, chroma, state.ep);
        } else {
          copy_frame<true>(dst, src, i, chroma, state.ep);
        }
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
      ds::ParamSpec{.name = "planes", .type = ds::ParamType::Integer, .is_array = true},
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
      ds::ParamSpec{.name = "y", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "u", .type = ds::ParamType::Integer},
      ds::ParamSpec{.name = "v", .type = ds::ParamType::Integer},
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
