#include "wrapper/vs_filter.hpp"
#include "neo_fft3d.hpp"
#include "version.hpp"

void VS_CC
pluginInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
  VSFilter *d = *reinterpret_cast<VSFilter**>(instanceData);
  vsapi->setVideoInfo(&d->vi._vi, 1, node);
}

void VS_CC
pluginFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
  VSFilter *d = static_cast<VSFilter*>(instanceData);
  vsapi->freeNode(d->child->_clip);
  delete d;
}

const VSFrameRef *VS_CC
pluginGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
  FFT3D<VSFilter> *d = *reinterpret_cast<FFT3D<VSFilter>**>(instanceData);
  static int to = (d->bt - 1) / 2;
  if (activationReason == arInitial) {
    d->GetFramePre(frameCtx, core, vsapi, n + to);
    return nullptr;
  }
  if (activationReason != arAllFramesReady)
    return nullptr;

  return d->GetFrame(frameCtx, core, vsapi, n);
}

static void VS_CC
pluginCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
  FFT3D<VSFilter> *data = nullptr;

  try {
    data = new FFT3D<VSFilter>(in, out, core, vsapi);
    data->initialize();
    vsapi->createFilter(in, out, "FFT3D", pluginInit, pluginGetFrame, pluginFree, data->bt == 0 ? fmSerial : fmParallelRequests, 0, data, core);
  }
  catch(const char *err){
    char msg_buff[256];
    snprintf(msg_buff, 256, "%s(" PLUGIN_VERSION "): %s", data ? data->name() : "Neo_FFT3D", err);
    vsapi->setError(out, msg_buff);
  }
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
  configFunc("in.7086.neo_fft3d", "neo_fft3d",
    "VapourSynth DelogoHD Filter " PLUGIN_VERSION,
    VAPOURSYNTH_API_VERSION, 1, plugin);
  const char * options =
    "clip:clip;sigma:float:opt;beta:float:opt;planes:int[]:opt;bw:int:opt;bh:int:opt;bt:int:opt;ow:int:opt;oh:int:opt;"
    "kratio:float:opt;sharpen:float:opt;scutoff:float:opt;svr:float:opt;smin:float:opt;smax:float:opt;"
    "measure:int:opt;interlaced:int:opt;wintype:int:opt;"
    "pframe:int:opt;px:int:opt;py:int:opt;pshow:int:opt;pcutoff:float:opt;pfactor:float:opt;"
    "sigma2:float:opt;sigma3:float:opt;sigma4:float:opt;degrid:float:opt;"
    "dehalo:float:opt;hr:float:opt;ht:float:opt;ncpu:int:opt;"
    "l:int:opt;t:int:opt;r:int:opt;b:int:opt;"
    ;
  registerFunc("DelogoHD", options, pluginCreate, nullptr, plugin);
}
