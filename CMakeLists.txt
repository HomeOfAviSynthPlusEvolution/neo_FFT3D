cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(neo_fft3d CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC AND NOT CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(AVISYNTHPLUS_TAG "751ce2021cab1c8405d04825ac2f7e8ac93e8fa0" CACHE STRING "AviSynthPlus tag used when pkg-config cannot find AviSynth headers")
set(VAPOURSYNTH_TAG "R72" CACHE STRING "VapourSynth tag used when pkg-config cannot find VapourSynth headers")

include(FetchContent)

# 1. Import Google Highway (for future SIMD operators)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.4.0
)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

# 2. Import dualsynth2
set(DS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DS_BUILD_ACCEPTANCE_PLUGIN OFF CACHE BOOL "" FORCE)
if(MSVC)
  set(DS_MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}" CACHE STRING "" FORCE)
endif()
FetchContent_Declare(
  dualsynth2
  GIT_REPOSITORY https://github.com/HomeOfAviSynthPlusEvolution/dualsynth2.git
  GIT_TAG master
)
FetchContent_MakeAvailable(dualsynth2)

# 3. Import PocketFFT (header-only C++ transform library)
FetchContent_Declare(
  pocketfft
  GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
  GIT_TAG 5f27d5a8f51c5c25030cb22abf434decc9faf0ff
)
FetchContent_MakeAvailable(pocketfft)

# 4. Dynamically pull official AviSynth+ and VapourSynth headers and perform compatibility adaptation (align with DelogoHD)
function(fetch_avisynth_headers out_var)
  FetchContent_Declare(
    avisynthplus_headers
    GIT_REPOSITORY https://github.com/AviSynth/AviSynthPlus.git
    GIT_TAG "${AVISYNTHPLUS_TAG}"
    SOURCE_SUBDIR no-cmake
  )
  FetchContent_MakeAvailable(avisynthplus_headers)
  FetchContent_GetProperties(avisynthplus_headers)
  set(${out_var} "${avisynthplus_headers_SOURCE_DIR}/avs_core/include" PARENT_SCOPE)
endfunction()

function(fetch_vapoursynth_headers out_var)
  FetchContent_Declare(
    vapoursynth_headers
    GIT_REPOSITORY https://github.com/vapoursynth/vapoursynth.git
    GIT_TAG "${VAPOURSYNTH_TAG}"
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR no-cmake
  )
  FetchContent_MakeAvailable(vapoursynth_headers)
  FetchContent_GetProperties(vapoursynth_headers)
  set(${out_var} "${vapoursynth_headers_SOURCE_DIR}/include" PARENT_SCOPE)
endfunction()

function(add_vapoursynth_compat_headers target)
  set(compat_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/vapoursynth-compat")
  file(MAKE_DIRECTORY "${compat_dir}/vapoursynth")
  foreach(header IN ITEMS
    VSConstants4.h
    VSHelper.h
    VSHelper4.h
    VSScript.h
    VSScript4.h
    VapourSynth.h
    VapourSynth4.h
  )
    file(WRITE
      "${compat_dir}/vapoursynth/${header}"
      "#pragma once\n#include <${header}>\n"
    )
  endforeach()
  target_include_directories(${target} INTERFACE "${compat_dir}")
endfunction()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(AVISYNTH QUIET avisynth)
  pkg_check_modules(VAPOURSYNTH QUIET vapoursynth)
endif()

add_library(avisynth_headers INTERFACE)
if(AVISYNTH_FOUND)
  target_include_directories(avisynth_headers INTERFACE ${AVISYNTH_INCLUDE_DIRS})
else()
  fetch_avisynth_headers(AVS_INCLUDE_DIRS)
  target_include_directories(avisynth_headers INTERFACE ${AVS_INCLUDE_DIRS})
endif()

add_library(vapoursynth_headers INTERFACE)
if(VAPOURSYNTH_FOUND)
  target_include_directories(vapoursynth_headers INTERFACE ${VAPOURSYNTH_INCLUDE_DIRS})
else()
  fetch_vapoursynth_headers(VS_INCLUDE_DIRS)
  target_include_directories(vapoursynth_headers INTERFACE ${VS_INCLUDE_DIRS})
endif()
add_vapoursynth_compat_headers(vapoursynth_headers)

# 5. Compile targets definition
set(NEO_FFT3D_PLUGIN_SOURCES
  src/plugin/plugin_entry.cpp
  src/plugin/fft3d_filter.cpp
  src/plugin/fft3d_filter.hpp
)

set(NEO_FFT3D_ENGINE_SOURCES
  src/cache.hpp
  src/common/aligned_vector.hpp
  src/common/fft3d_views.hpp
  src/cpuid.cpp
  src/engine/engine_params.hpp
  src/engine/frame_buffer.cpp
  src/engine/frame_buffer.hpp
  src/engine/fft3d_engine.hpp
  src/engine/pattern_analysis.cpp
  src/engine/pattern_analysis.hpp
  src/fft3d_common.h
  src/fftwlite.h
  src/functions.h
  src/neo_fft3d.hpp
)

set(NEO_FFT3D_FFT_SOURCES
  src/fft/fft_backend.hpp
  src/fft/fftw_abi.hpp
  src/fft/fftw_backend.cpp
  src/fft/fftw_lock.hpp
  src/fft/pocketfft_backend.cpp
)

set(NEO_FFT3D_CPU_SOURCES
  src/core/cpu/apply_hwy.cpp
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

# Link dependencies, automatically bringing in dualsynth2, hwy, and pulled AVS/VS official headers
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

# 6. Git version control injection
find_package(Git REQUIRED)
execute_process(
  COMMAND ${GIT_EXECUTABLE} describe --first-parent --tags --always
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_REPO_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "^(r[0-9]+).*$" "\\1" VERSION "${GIT_REPO_VERSION}")
execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_COUNT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/version.hpp.in"
  "${GENERATED_DIR}/version.hpp"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/src/version.rc.in"
  "${GENERATED_DIR}/version.rc"
)

# Add generated rc file as source file to compilation
target_sources(neo-fft3d PRIVATE "${GENERATED_DIR}/version.rc")
target_include_directories(neo-fft3d PRIVATE "${GENERATED_DIR}")

# 7. Test tools build definition
option(NEO_FFT3D_BUILD_TEST_TOOLS "Build Neo_FFT3D baseline test tools" ON)

if (NEO_FFT3D_BUILD_TEST_TOOLS)
  add_executable(neo-fft3d_avs_frame_dump
    tests/tools/avs_frame_dump.cpp
  )
  set_property(TARGET neo-fft3d_avs_frame_dump PROPERTY CXX_STANDARD 17)
  target_link_libraries(neo-fft3d_avs_frame_dump PRIVATE avisynth_headers)

  if (NOT WIN32)
    target_link_libraries(neo-fft3d_avs_frame_dump PRIVATE dl)
  endif()
endif()

if (NOT WIN32)
  target_link_libraries(neo-fft3d PRIVATE dl)
endif()

# 8. CTest unit tests and baseline verification integration
if (NEO_FFT3D_BUILD_TEST_TOOLS)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  add_test(
    NAME neo-fft3d_baseline_unit_tests
    COMMAND "${Python3_EXECUTABLE}" -m unittest tests.baseline.test_baseline_runner -v
  )
  set_tests_properties(neo-fft3d_baseline_unit_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")

  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(neo-fft3d_cache_test tests/unit/cache_test.cpp)
  target_include_directories(neo-fft3d_cache_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(neo-fft3d_cache_test PRIVATE Catch2::Catch2WithMain)
  add_test(NAME neo-fft3d_cache_test COMMAND "$<TARGET_FILE:neo-fft3d_cache_test>")

  add_executable(neo-fft3d_c_apply_null_slots_test
    tests/unit/c_apply_null_slots_test.cpp
    src/code_impl/Apply.cpp
  )
  target_include_directories(neo-fft3d_c_apply_null_slots_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(neo-fft3d_c_apply_null_slots_test PRIVATE DualSynth::dualsynth)

  add_test(
    NAME neo-fft3d_c_apply_null_slots_test
    COMMAND "$<TARGET_FILE:neo-fft3d_c_apply_null_slots_test>"
  )

  add_test(
    NAME neo-fft3d_baseline_smoke_verification
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/baseline_runner.py" verify
      --plugin "$<TARGET_FILE:neo-fft3d>"
      --avs-dump "$<TARGET_FILE:neo-fft3d_avs_frame_dump>"
      --golden "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/golden/smoke.json"
      --tier smoke
      --hosts vs avs
  )
  set_tests_properties(neo-fft3d_baseline_smoke_verification PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")

  add_test(
    NAME neo-fft3d_baseline_compat_verification
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/baseline_runner.py" verify
      --plugin "$<TARGET_FILE:neo-fft3d>"
      --avs-dump "$<TARGET_FILE:neo-fft3d_avs_frame_dump>"
      --golden "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/golden/compat.json"
      --tier compat
      --hosts vs avs
  )
  set_tests_properties(neo-fft3d_baseline_compat_verification PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

add_custom_command(
  TARGET neo-fft3d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:neo-fft3d> "../Release_${VERSION}/${_DIR}/$<TARGET_FILE_NAME:neo-fft3d>"
)
