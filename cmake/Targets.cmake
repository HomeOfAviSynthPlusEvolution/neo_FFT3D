set(NEO_FFT3D_PLUGIN_SOURCES
  src/plugin/plugin_entry.cpp
  src/plugin/fft3d_filter.cpp
  src/plugin/fft3d_filter.hpp
)

set(NEO_FFT3D_ENGINE_SOURCES
  src/cache.hpp
  src/common/aligned_vector.hpp
  src/common/fft3d_views.hpp
  src/engine/engine_params.hpp
  src/engine/engine_workspace.cpp
  src/engine/engine_workspace.hpp
  src/engine/cpu_filter_backend.cpp
  src/engine/filter_backend.hpp
  src/engine/frame_cover.cpp
  src/engine/frame_buffer.hpp
  src/engine/frame_views.hpp
  src/engine/fft3d_engine.cpp
  src/engine/fft3d_engine.hpp
  src/engine/overlap_transform.cpp
  src/engine/pattern_analysis.cpp
  src/engine/pattern_analysis.hpp
  src/fft3d_common.h
  src/neo_fft3d.hpp
)

set(NEO_FFT3D_FFT_SOURCES
  src/fft/dynamic_library.cpp
  src/fft/dynamic_library.hpp
  src/fft/fft_backend.hpp
  src/fft/fftw_abi.hpp
  src/fft/fftw_backend.cpp
  src/fft/fftw_lock.hpp
  src/fft/fftw_runtime.cpp
  src/fft/fftw_runtime.hpp
  src/fft/pocketfft_backend.cpp
)

set(NEO_FFT3D_CPU_SOURCES
  src/core/cpu/apply_hwy.cpp
  src/core/cpu/cpu_dispatch.cpp
  src/core/cpu/cpu_dispatch.hpp
  src/core/cpu/core_hwy.h
  src/core/cpu/kalman_hwy.cpp
  src/core/cpu/sharpen_hwy.cpp
)

set(NEO_FFT3D_CODE_IMPL_SOURCES
  src/code_impl/Apply.cpp
  src/code_impl/Kalman.cpp
  src/code_impl/Sharpen.cpp
  src/code_impl/code_impl.h
  src/code_impl/code_impl_C.h
)

add_library(neo-fft3d SHARED
  ${NEO_FFT3D_PLUGIN_SOURCES}
  ${NEO_FFT3D_ENGINE_SOURCES}
  ${NEO_FFT3D_FFT_SOURCES}
  ${NEO_FFT3D_CPU_SOURCES}
  ${NEO_FFT3D_CODE_IMPL_SOURCES}
)

target_link_libraries(neo-fft3d PRIVATE
  DualSynth::dualsynth
  hwy
  avisynth_headers
  vapoursynth_headers
)

target_include_directories(neo-fft3d PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  src
  ${pocketfft_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_compile_definitions(neo-fft3d PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)

target_sources(neo-fft3d PRIVATE "${GENERATED_DIR}/version.rc")
target_include_directories(neo-fft3d PRIVATE "${GENERATED_DIR}")

if (NOT WIN32)
  target_link_libraries(neo-fft3d PRIVATE dl)
endif()

if(NEO_FFT3D_CLANG_TIDY)
  set_target_properties(neo-fft3d PROPERTIES CXX_CLANG_TIDY "${NEO_FFT3D_CLANG_TIDY}")
endif()

add_custom_command(
  TARGET neo-fft3d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:neo-fft3d> "../Release_${VERSION}/${_DIR}/$<TARGET_FILE_NAME:neo-fft3d>"
)
