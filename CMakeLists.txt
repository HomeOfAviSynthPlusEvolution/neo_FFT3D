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

# 1. 引入 Google Highway (后续 SIMD 算子使用)
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

# 2. 引入 dualsynth2
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

# 3. 引入 PocketFFT (纯头文件 C++ 变换库)
FetchContent_Declare(
  pocketfft
  GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
  GIT_TAG 5f27d5a8f51c5c25030cb22abf434decc9faf0ff
)
FetchContent_MakeAvailable(pocketfft)

# 3. 动态拉取 AviSynth+ 与 VapourSynth 官方头文件并进行兼容适配 (对齐 DelogoHD)
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

# 4. 编译目标定义
file(GLOB CODE "src/*.cpp")
file(GLOB CODE_FFT "src/fft/*.cpp")
file(GLOB CODE_IMPL "src/code_impl/*.cpp")
file(GLOB AVX_CODE_IMPL "src/code_impl/*AVX.cpp")
file(GLOB AVX512_CODE_IMPL "src/code_impl/*AVX512.cpp")

add_library(neo-fft3d SHARED
  src/plugin/plugin_entry.cpp
  src/plugin/fft3d_filter.cpp
  src/plugin/fft3d_filter.hpp
  ${CODE}
  ${CODE_FFT}
  ${CODE_IMPL}
)

# 链接依赖，自动带入 dualsynth2、hwy、以及拉取的 AVS/VS 官方头文件
target_link_libraries(neo-fft3d PRIVATE
  DualSynth::dualsynth
  hwy
  avisynth_headers
  vapoursynth_headers
)

target_include_directories(neo-fft3d PRIVATE
  src
  ${pocketfft_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# 临时保留旧指令集编译规则 (在后续阶段使用 Highway 重构时将删除)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set_source_files_properties(${AVX_CODE_IMPL} PROPERTIES COMPILE_FLAGS "/arch:AVX")
  set_source_files_properties(${AVX512_CODE_IMPL} PROPERTIES COMPILE_FLAGS "/arch:AVX512")
else()
  set_source_files_properties(${AVX_CODE_IMPL} PROPERTIES COMPILE_FLAGS "-mavx")
  set_source_files_properties(${AVX512_CODE_IMPL} PROPERTIES COMPILE_FLAGS "-mavx512f")
endif()

target_compile_definitions(neo-fft3d PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)

# 5. Git 版本控制注入
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

# 将生成的 rc 文件作为源文件加入编译
target_sources(neo-fft3d PRIVATE "${GENERATED_DIR}/version.rc")
target_include_directories(neo-fft3d PRIVATE "${GENERATED_DIR}")

# 6. 测试工具构建定义
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

# 7. CTest 单元测试与基准校验集成
if (NEO_FFT3D_BUILD_TEST_TOOLS)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  add_test(
    NAME neo-fft3d_baseline_unit_tests
    COMMAND "${Python3_EXECUTABLE}" -m unittest tests.baseline.test_baseline_runner -v
  )
  set_tests_properties(neo-fft3d_baseline_unit_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")

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
