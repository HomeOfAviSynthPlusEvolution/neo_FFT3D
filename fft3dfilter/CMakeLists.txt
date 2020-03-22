cmake_minimum_required (VERSION 3.1)
project (neo_fft3d)
file(GLOB CODE_IMPL "code_impl/*.cpp")
add_library(neo_fft3d SHARED FFT3DFilter.cpp fft3dfilter_c.cpp fft3dfilter_sse.cpp info.cpp FFT3DFilter.rc ${CODE_IMPL})
include_directories(.)
add_custom_command(
  TARGET neo_fft3d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:neo_fft3d> "../Release_${VERSION}/${_DIR}/$<TARGET_FILE_NAME:neo_fft3d>"
)
