/*
 * Copyright 2004-2007 A.G.Balakhnin aka Fizick
 * Copyright 2015 martin53
 * Copyright 2017-2019 Ferenc Pinter aka printerf
 * Copyright 2020 Xinyue Lu
 *
 * Main file.
 *
 */

#pragma once

#include "fft3d_common.h"
#include "cache.hpp"
#include "engine/engine_workspace.hpp"
#include "engine/filter_backend.hpp"
#include "engine/frame_views.hpp"
#include "fft/fft_backend.hpp"

#include <dualsynth/video_filter.hpp>

#include <complex>
#include <cstddef>
#include <memory>
#include <mutex>

class FFT3DEngine {
public:
  std::unique_ptr<EngineParams> ep;
  std::unique_ptr<IOParams> iop;
  int plane; // color plane
  std::shared_ptr<neo_fft3d::fft::FFTBackend> fft_backend;
  std::unique_ptr<neo_fft3d::engine::FilterBackend> backend;

  std::unique_ptr<neo_fft3d::fft::FFTPlan> plan, planinv, plan1;

  AlignedVector<std::complex<float>> gridsample; //v1.8
  AlignedVector<std::complex<float>> outLast;
  AlignedVector<std::complex<float>> covar;
  AlignedVector<std::complex<float>> covarProcess;

  AlignedVector<float> wsharpen;
  AlignedVector<float> wdehalo;
  AlignedVector<float> pwin;
  AlignedVector<float> pattern2d;
  AlignedVector<float> pattern3d;

  neo_fft3d::EngineWorkspace workspace;

  std::unique_ptr<cache<std::complex<float>>> fftcache;

  char messagebuf[80] {0};

  int outwidth;
  int outpitch; //v.1.7
  int insize;
  int outsize;
  int howmanyblocks;

  float sigmaSquaredNoiseNormed2D {0};
  float sigmaNoiseNormed2D {0};
  float sigmaMotionNormed {0};
  float sigmaSquaredSharpenMinNormed {0};
  float sigmaSquaredSharpenMaxNormed {0};
  float ht2n {0}; // halo threshold squared normed
  float norm {0}; // normalization factor

  int coverwidth;
  int coverheight;
  int coverpitch;

  int mirw; // mirror width for padding
  int mirh; // mirror height for padding

  bool isPatternSet;
  float psigma {0};

  std::mutex cache_mutex;

  bool pattern3d_initialized {false};
  std::mutex init3d_mutex;

private:
  neo_fft3d::engine::CoverPlaneGeometry cover_geometry() const;

  neo_fft3d::engine::FrameViewAdapter frame_views() const;

  neo_fft3d::FloatPlaneView kernel_float_view(AlignedVector<float>& data) const;

  neo_fft3d::ConstFloatPlaneView kernel_float_view(const AlignedVector<float>& data) const;

  neo_fft3d::ComplexBlockView kernel_complex_blocks(AlignedVector<std::complex<float>>& data, int block_count) const;

  neo_fft3d::ComplexBlockView kernel_complex_blocks(std::complex<float>* data, int block_count) const;

  void validate_backend() const;

  void normalize_params_for_sample_format();

  void validate_overlap_params_and_apply_defaults();

  void validate_temporal_params() const;

  void configure_geometry();

  void configure_storage();

  void create_main_plans(const neo_fft3d::fft::PlanOptions& plan_options, const neo_fft3d::fft::PlanBuffers& plan_buffers);

  void resize_window_buffers();

  void initialize_overlap_windows();

  void initialize_sharpen_window();

  void initialize_dehalo_window();

  void initialize_filter_windows();

  void initialize_normalization_params();

  void initialize_kalman_buffers();

  void initialize_pattern_buffers();

  void create_grid_plan(const neo_fft3d::fft::PlanOptions& plan_options, const neo_fft3d::fft::PlanBuffers& plan_buffers);

  void initialize_grid_sample(float* in, const neo_fft3d::fft::PlanOptions& plan_options,
                              const neo_fft3d::fft::PlanBuffers& plan_buffers);

  void ensure_fft_cache_capacity(std::size_t workspace_slot_count);

  void transform_source_plane_to_spectrum(
    const ds::PlaneView& src_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* spectrum,
    bool chroma
  );

  void write_spectrum_to_destination(
    std::complex<float>* spectrum,
    float* in,
    byte* coverbuf,
    const neo_fft3d::engine::FrameViewAdapter& views,
    const ds::MutablePlaneView& dst_plane,
    bool chroma
  );

  void show_pattern_frame(
    int n,
    int pxf,
    int pyf,
    ds::MutablePlaneView& dst_plane,
    const ds::PlaneView& src_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez
  );

  bool handle_pattern_frame(
    int n,
    ds::VideoFrameProvider& provider,
    ds::MutableVideoFrameView& dst,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez
  );

  int current_bt_for_frame(int n) const;

  SharedFunctionParams make_shared_function_params(int btcur);

  void ensure_3d_pattern_initialized(int btcur);

  void process_wiener_3d_frame(
    int n,
    int btcur,
    ds::VideoFrameProvider& provider,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  );

  void process_wiener_frame(
    int n,
    int btcur,
    ds::VideoFrameProvider& provider,
    const ds::PlaneView& src_plane,
    const ds::MutablePlaneView& dst_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  );

  void process_kalman_frame(
    int n,
    const ds::PlaneView& src_plane,
    const ds::MutablePlaneView& dst_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  );

  void process_sharpen_frame(
    const ds::PlaneView& src_plane,
    const ds::MutablePlaneView& dst_plane,
    const neo_fft3d::engine::FrameViewAdapter& views,
    byte* coverbuf,
    float* in,
    std::complex<float>* outrez,
    SharedFunctionParams sfp
  );

public:
  FFT3DEngine(EngineParams _ep, int _plane, std::shared_ptr<neo_fft3d::fft::FFTBackend> _fft_backend,
              std::unique_ptr<neo_fft3d::engine::FilterBackend> _backend);
  ~FFT3DEngine();

  void ProcessFrame(int n, ds::VideoFrameProvider& provider, ds::MutableVideoFrameView dst);

  bool CacheRefresh(int n);
};
