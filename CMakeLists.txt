cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(neo_fft3d CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NEO_FFT3D_ENABLE_CLANG_TIDY "Enable clang-tidy static analysis during compilation" OFF)

if(NEO_FFT3D_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
    set(NEO_FFT3D_CLANG_TIDY "${CLANG_TIDY_EXE};--extra-arg=-Wno-unknown-argument;--extra-arg=-Wno-unused-command-line-argument")
  else()
    message(WARNING "clang-tidy requested but not found in PATH")
  endif()
endif()

if(MSVC AND NOT CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(AVISYNTHPLUS_TAG "751ce2021cab1c8405d04825ac2f7e8ac93e8fa0" CACHE STRING "AviSynthPlus tag used when pkg-config cannot find AviSynth headers")
set(VAPOURSYNTH_TAG "R72" CACHE STRING "VapourSynth tag used when pkg-config cannot find VapourSynth headers")

include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Targets.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Tests.cmake")
