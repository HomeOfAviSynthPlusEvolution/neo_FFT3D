/*
 * Copyright 2020 Xinyue Lu
 *
 * DualSynth wrapper - Common header+.
 *
 */
#pragma once

#include <avisynth.h>
#include <VapourSynth.h>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include "ds_format.hpp"
#include "ds_videoinfo.hpp"
#include "ds_frame.hpp"

inline std::mutex& GetFFTWMutex() {
    static std::mutex m;
    return m;
}

class GlobalLockGuard
{
    IScriptEnvironment* env;
    const char* name;
    bool acquired;
    bool is_legacy;

public:
    GlobalLockGuard(IScriptEnvironment* _env, const char* _name, bool use_avs_lock)
        : env(_env), name(_name), acquired(false), is_legacy(false)
    {
        if (!name)
            return;
        if (env && use_avs_lock)
        {
            acquired = env->AcquireGlobalLock(name);
            if (acquired)
                return;
        }
        if (strcmp(name, "fftw") == 0) {
            GetFFTWMutex().lock();
            acquired = true;
            is_legacy = true;
        }
    }

    ~GlobalLockGuard()
    {
        if (acquired)
        {
            if (is_legacy)
                GetFFTWMutex().unlock();
            else
                env->ReleaseGlobalLock(name);
        }
    }

    GlobalLockGuard(const GlobalLockGuard&) = delete;
    GlobalLockGuard& operator=(const GlobalLockGuard&) = delete;
};

typedef void (*register_vsfilter_proc)(VSRegisterFunction, VSPlugin*);
typedef void (*register_avsfilter_proc)(IScriptEnvironment* env);
std::vector<register_vsfilter_proc> RegisterVSFilters();
std::vector<register_avsfilter_proc> RegisterAVSFilters();

enum ParamType
{
  Clip, Integer, Float, Boolean, String
};

struct Param
{
  const char* Name;
  const ParamType Type;
  const bool IsArray {false};
  bool AVSEnabled {true};
  bool VSEnabled {true};
  const bool IsOptional {true};
};

struct InDelegator
{
  virtual void* GetEnv() { return nullptr; }
  virtual bool IsAVS12() { return false; }
  virtual void Read(const char* name, int& output) = 0;
  virtual void Read(const char* name, int64_t& output) = 0;
  virtual void Read(const char* name, float& output) = 0;
  virtual void Read(const char* name, double& output) = 0;
  virtual void Read(const char* name, bool& output) = 0;
  virtual void Read(const char* name, std::string& output) = 0;
  virtual void Read(const char* name, std::vector<int>& output) = 0;
  virtual void Read(const char* name, std::vector<int64_t>& output) = 0;
  virtual void Read(const char* name, std::vector<float>& output) = 0;
  virtual void Read(const char* name, std::vector<double>& output) = 0;
  virtual void Read(const char* name, std::vector<bool>& output) = 0;
  virtual void Read(const char* name, void*& output) = 0;
  virtual void Free(void*& clip) = 0;
};

struct FetchFrameFunctor
{
  virtual DSFrame operator()(int n) = 0;
  virtual ~FetchFrameFunctor() {}
};
