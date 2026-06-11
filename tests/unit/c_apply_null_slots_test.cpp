#include "src/code_impl/code_impl.h"

#include <cmath>
#include <iostream>
#include <memory>

namespace {

void fill_complex(fftwf_complex* data, int count, float real_base) {
  for (int i = 0; i < count; ++i) {
    data[i][0] = real_base + static_cast<float>(i) * 0.25f;
    data[i][1] = -0.5f + static_cast<float>(i) * 0.125f;
  }
}

void fill_float(float* data, int count, float value) {
  for (int i = 0; i < count; ++i) {
    data[i] = value;
  }
}

SharedFunctionParams make_params(
    fftwf_complex* gridsample,
    float* pattern2d,
    float* pattern3d,
    float* wsharpen,
    float* wdehalo,
    fftwf_complex* covar,
    fftwf_complex* covar_process) {
  SharedFunctionParams sfp {};
  sfp.outwidth = 3;
  sfp.outpitch = 4;
  sfp.bh = 2;
  sfp.howmanyblocks = 1;
  sfp.sigmaSquaredNoiseNormed = 0.1f;
  sfp.pfactor = 0.0f;
  sfp.pattern2d = pattern2d;
  sfp.pattern3d = pattern3d;
  sfp.beta = 2.0f;
  sfp.degrid = 0.0f;
  sfp.gridsample = gridsample;
  sfp.sharpen = 0.0f;
  sfp.sigmaSquaredSharpenMinNormed = 0.0f;
  sfp.sigmaSquaredSharpenMaxNormed = 0.0f;
  sfp.wsharpen = wsharpen;
  sfp.dehalo = 0.0f;
  sfp.wdehalo = wdehalo;
  sfp.ht2n = 0.0f;
  sfp.covar = covar;
  sfp.covarProcess = covar_process;
  sfp.sigmaSquaredNoiseNormed2D = 0.1f;
  sfp.kratio2 = 1.0f;
  return sfp;
}

bool output_is_finite(const fftwf_complex* out, int count) {
  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(out[i][0]) || !std::isfinite(out[i][1])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int total = 8;
  auto prev = std::make_unique<fftwf_complex[]>(total);
  auto curr = std::make_unique<fftwf_complex[]>(total);
  auto out = std::make_unique<fftwf_complex[]>(total);
  auto gridsample = std::make_unique<fftwf_complex[]>(total);
  auto covar = std::make_unique<fftwf_complex[]>(total);
  auto covar_process = std::make_unique<fftwf_complex[]>(total);
  auto pattern2d = std::make_unique<float[]>(total);
  auto pattern3d = std::make_unique<float[]>(total);
  auto wsharpen = std::make_unique<float[]>(total);
  auto wdehalo = std::make_unique<float[]>(total);

  fill_complex(prev.get(), total, 0.25f);
  fill_complex(curr.get(), total, 1.0f);
  fill_complex(out.get(), total, 0.0f);
  fill_complex(gridsample.get(), total, 1.0f);
  fill_complex(covar.get(), total, 0.0f);
  fill_complex(covar_process.get(), total, 0.0f);
  fill_float(pattern2d.get(), total, 0.1f);
  fill_float(pattern3d.get(), total, 0.1f);
  fill_float(wsharpen.get(), total, 0.0f);
  fill_float(wdehalo.get(), total, 0.0f);

  fftwf_complex* in[5] {};
  in[1] = prev.get();
  in[2] = curr.get();

  auto sfp = make_params(
      gridsample.get(),
      pattern2d.get(),
      pattern3d.get(),
      wsharpen.get(),
      wdehalo.get(),
      covar.get(),
      covar_process.get());

  Apply3D2_C<false, false>(in, out.get(), sfp);

  if (in[0] != nullptr || in[3] != nullptr || in[4] != nullptr) {
    std::cerr << "Apply3D2_C advanced unused null input slots\n";
    return 1;
  }

  if (!output_is_finite(out.get(), total)) {
    std::cerr << "Apply3D2_C produced non-finite output\n";
    return 1;
  }
  return 0;
}
