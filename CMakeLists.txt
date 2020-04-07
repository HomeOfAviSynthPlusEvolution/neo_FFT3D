cmake_minimum_required (VERSION 3.1)
project (neo_fft3d CXX)
file(GLOB CODE "src/*.cpp")
file(GLOB CODE_IMPL "src/code_impl/*.cpp")
file(GLOB AVX_CODE_IMPL "src/code_impl/*AVX.cpp")
file(GLOB AVX512_CODE_IMPL "src/code_impl/*AVX512.cpp")
add_library(neo-fft3d SHARED main.cpp src/version.rc ${CODE} ${CODE_IMPL})
set_property(TARGET neo-fft3d PROPERTY CXX_STANDARD 17)

find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} describe --first-parent --tags --always OUTPUT_VARIABLE GIT_REPO_VERSION)
string(APPEND GIT_REPO_VERSION "r0")
string(REGEX REPLACE "(r[0-9]+).*" "\\1" VERSION ${GIT_REPO_VERSION})

configure_file (
  "${PROJECT_SOURCE_DIR}/src/version.hpp.in"
  "${PROJECT_SOURCE_DIR}/src/version.hpp"
)
configure_file (
  "${PROJECT_SOURCE_DIR}/src/version.rc.in"
  "${PROJECT_SOURCE_DIR}/src/version.rc"
)

find_package(PkgConfig REQUIRED)

pkg_check_modules(AVISYNTH avisynth)
if(AVISYNTH_FOUND)
  include_directories(${AVISYNTH_INCLUDE_DIRS})
else()
  include_directories(include/avisynth)
endif()

pkg_check_modules(VAPOURSYNTH vapoursynth)
if(VAPOURSYNTH_FOUND)
  include_directories(${VAPOURSYNTH_INCLUDE_DIRS})
else()
  include_directories(include/vapoursynth)
endif()

include_directories(.)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set_source_files_properties(${AVX_CODE_IMPL} PROPERTIES COMPILE_FLAGS "/arch:AVX")
  set_source_files_properties(${AVX512_CODE_IMPL} PROPERTIES COMPILE_FLAGS "/arch:AVX512")

  if (CMAKE_GENERATOR_TOOLSET MATCHES "v[0-9]*_xp")
    target_compile_definitions(neo-fft3d PRIVATE WINVER=0x502 _WIN32_WINNT=0x502)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:threadSafeInit-")
  endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set_source_files_properties(${AVX_CODE_IMPL} PROPERTIES COMPILE_FLAGS "/arch:AVX")
  set_source_files_properties(${AVX512_CODE_IMPL} PROPERTIES COMPILE_FLAGS "/QxCOMMON-AVX512")

else()
  set_source_files_properties(${AVX_CODE_IMPL} PROPERTIES COMPILE_FLAGS "-mavx")
  set_source_files_properties(${AVX512_CODE_IMPL} PROPERTIES COMPILE_FLAGS "-mavx512f")

endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(neo-fft3d tbb)
endif()

add_custom_command(
  TARGET neo-fft3d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:neo-fft3d> "../Release_${VERSION}/${_DIR}/$<TARGET_FILE_NAME:neo-fft3d>"
)
