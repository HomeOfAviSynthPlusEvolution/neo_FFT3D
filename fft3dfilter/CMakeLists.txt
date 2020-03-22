cmake_minimum_required (VERSION 3.1)
project (FFT3DFilter)
add_library(FFT3DFilter SHARED FFT3DFilter.cpp fft3dfilter_c.cpp fft3dfilter_sse.cpp info.cpp FFT3DFilter.rc)
include_directories(.)
add_custom_command(
  TARGET FFT3DFilter POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FFT3DFilter> "../Release_${VERSION}/${_DIR}/$<TARGET_FILE_NAME:FFT3DFilter>"
)
