#pragma once

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <memory>
#include <new>

#include "common/aligned_vector.hpp"
#include "engine/engine_params.hpp"
#include "fft/fftw_abi.hpp"

#ifdef HAS_EXECUTION
  #include <execution>
#endif

#ifndef __cpp_lib_execution
  #undef ENABLE_PAR
#endif

#ifdef ENABLE_PAR
  #define PAR_POLICY std::execution::par
  #define SEQ_POLICY std::execution::seq
#else
  #define PAR_POLICY nullptr
  #define SEQ_POLICY nullptr
#endif

#ifndef _WIN32
  #define wsprintf sprintf
#endif

#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif
