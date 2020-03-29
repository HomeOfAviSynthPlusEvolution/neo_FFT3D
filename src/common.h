#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cmath>

#include "fftwlite.h"

typedef unsigned char byte;

#ifndef _WIN32
  #define wsprintf sprintf
#endif

#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

int GetCPUFlags();

#endif
