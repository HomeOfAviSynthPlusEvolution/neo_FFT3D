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
} // namespace

FFT3DCore::State::State() {
  fftfp = std::make_shared<FFTFunctionPointers>();
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

FFT3DCore::State::State(State&&) noexcept = default;
FFT3DCore::State& FFT3DCore::State::operator=(State&&) noexcept = default;

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

    float sigma1 = 2.0f;
    context.params->get_double("sigma", sigma1);

    state.ep = new EngineParams {
      sigma1, 1.0f,
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
      sigma1, sigma1, sigma1,
      1.0f,
      0.0f,
      2.0f, 50.0f,
      0, 0, 0, 0,
      0,
      in_vi,
      nullptr, // avs_env
      false // IsAVS12
    };

    double temp_beta = state.ep->beta;
    context.params->get_double("beta", temp_beta);
    state.ep->beta = static_cast<float>(temp_beta);

    int temp_bw = state.ep->bw;
    context.params->get_int("bw", temp_bw);
    state.ep->bw = temp_bw;

    int temp_bh = state.ep->bh;
    context.params->get_int("bh", temp_bh);
    state.ep->bh = temp_bh;

    int temp_bt = state.ep->bt;
    context.params->get_int("bt", temp_bt);
    state.ep->bt = temp_bt;

    int temp_ow = state.ep->ow;
    context.params->get_int("ow", temp_ow);
    state.ep->ow = temp_ow;

    int temp_oh = state.ep->oh;
    context.params->get_int("oh", temp_oh);
    state.ep->oh = temp_oh;

    double temp_kratio = state.ep->kratio;
    context.params->get_double("kratio", temp_kratio);
    state.ep->kratio = static_cast<float>(temp_kratio);

    double temp_sharpen = state.ep->sharpen;
    context.params->get_double("sharpen", temp_sharpen);
    state.ep->sharpen = static_cast<float>(temp_sharpen);

    double temp_scutoff = state.ep->scutoff;
    context.params->get_double("scutoff", temp_scutoff);
    state.ep->scutoff = static_cast<float>(temp_scutoff);

    double temp_svr = state.ep->svr;
    context.params->get_double("svr", temp_svr);
    state.ep->svr = static_cast<float>(temp_svr);

    double temp_smin = state.ep->smin;
    context.params->get_double("smin", temp_smin);
    state.ep->smin = static_cast<float>(temp_smin);

    double temp_smax = state.ep->smax;
    context.params->get_double("smax", temp_smax);
    state.ep->smax = static_cast<float>(temp_smax);

    bool temp_measure = state.ep->measure;
    context.params->get_bool("measure", temp_measure);
    state.ep->measure = temp_measure;

    bool temp_interlaced = state.ep->interlaced;
    context.params->get_bool("interlaced", temp_interlaced);
    state.ep->interlaced = temp_interlaced;

    int temp_wintype = state.ep->wintype;
    context.params->get_int("wintype", temp_wintype);
    state.ep->wintype = temp_wintype;

    int temp_pframe = state.ep->pframe;
    context.params->get_int("pframe", temp_pframe);
    state.ep->pframe = temp_pframe;

    int temp_px = state.ep->px;
    context.params->get_int("px", temp_px);
    state.ep->px = temp_px;

    int temp_py = state.ep->py;
    context.params->get_int("py", temp_py);
    state.ep->py = temp_py;

    bool temp_pshow = state.ep->pshow;
    context.params->get_bool("pshow", temp_pshow);
    state.ep->pshow = temp_pshow;

    double temp_pcutoff = state.ep->pcutoff;
    context.params->get_double("pcutoff", temp_pcutoff);
    state.ep->pcutoff = static_cast<float>(temp_pcutoff);

    double temp_pfactor = state.ep->pfactor;
    context.params->get_double("pfactor", temp_pfactor);
    state.ep->pfactor = static_cast<float>(temp_pfactor);

    double temp_sigma2 = state.ep->sigma2;
    context.params->get_double("sigma2", temp_sigma2);
    state.ep->sigma2 = static_cast<float>(temp_sigma2);

    double temp_sigma3 = state.ep->sigma3;
    context.params->get_double("sigma3", temp_sigma3);
    state.ep->sigma3 = static_cast<float>(temp_sigma3);

    double temp_sigma4 = state.ep->sigma4;
    context.params->get_double("sigma4", temp_sigma4);
    state.ep->sigma4 = static_cast<float>(temp_sigma4);

    double temp_degrid = state.ep->degrid;
    context.params->get_double("degrid", temp_degrid);
    state.ep->degrid = static_cast<float>(temp_degrid);

    double temp_dehalo = state.ep->dehalo;
    context.params->get_double("dehalo", temp_dehalo);
    state.ep->dehalo = static_cast<float>(temp_dehalo);

    double temp_hr = state.ep->hr;
    context.params->get_double("hr", temp_hr);
    state.ep->hr = static_cast<float>(temp_hr);

    double temp_ht = state.ep->ht;
    context.params->get_double("ht", temp_ht);
    state.ep->ht = static_cast<float>(temp_ht);

    int temp_l = state.ep->l;
    context.params->get_int("l", temp_l);
    state.ep->l = (std::max)(temp_l, 0);

    int temp_t = state.ep->t;
    context.params->get_int("t", temp_t);
    state.ep->t = (std::max)(temp_t, 0);

    int temp_r = state.ep->r;
    context.params->get_int("r", temp_r);
    state.ep->r = (std::max)(temp_r, 0);

    int temp_b = state.ep->b;
    context.params->get_int("b", temp_b);
    state.ep->b = (std::max)(temp_b, 0);

    int temp_opt = state.ep->opt;
    context.params->get_int("opt", temp_opt);
    state.ep->opt = temp_opt;

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
    if (res_planes.has_value()) {
      for (auto&& p : res_planes.value()) {
        if (p < state.ep->vi.Format.Planes) {
          state.process[p] = 3;
        }
      }
    } else {
      context.params->get_int("y", state.process[0]);
      context.params->get_int("u", state.process[1]);
      context.params->get_int("v", state.process[2]);
    }

    context.params->get_bool("mt", state.mt);
    context.params->get_int("ncpu", state.fft_threads);
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
    FetchFrameFunctor fetch_frame_func { fetch_frame_bridge, &context };

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      state.plane_index[i] = planes[i];
      if (state.process[i] == 3) {
        state.ep->IsChroma = state.ep->vi.Format.IsFamilyYUV && i != 0;
        state.ep->framewidth = state.ep->IsChroma ? state.ep->vi.Width >> state.ep->vi.Format.SSW : state.ep->vi.Width;
        state.ep->frameheight = state.ep->IsChroma ? state.ep->vi.Height >> state.ep->vi.Format.SSH : state.ep->vi.Height;
        state.engine[i] = new FFT3DEngine(*state.ep, i, &fetch_frame_func, *state.fftfp);
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
    std::memcpy(dst_ptr, pcs_ptr, dst_stride * height);
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

    for (int i = 0; i < state.ep->vi.Format.Planes; i++) {
      if (state.process[i] == 3 && state.engine[i]) {
        state.engine[i]->fetch_frame->Opaque = &context;
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
