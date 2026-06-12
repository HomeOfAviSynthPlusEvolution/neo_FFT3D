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

FetchContent_Declare(
  pocketfft
  GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
  GIT_TAG 5f27d5a8f51c5c25030cb22abf434decc9faf0ff
)
FetchContent_MakeAvailable(pocketfft)

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
