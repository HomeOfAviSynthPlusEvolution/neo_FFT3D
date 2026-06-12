#pragma once

#include "common/aligned_vector.hpp"

#include <complex>
#include <cstddef>
#include <mutex>
#include <vector>

namespace neo_fft3d {

struct EngineWorkspaceSlot {
  unsigned int id {0};
  byte* cover {nullptr};
  float* overlap {nullptr};
  std::complex<float>* spectrum {nullptr};
  std::size_t slot_count {0};
};

class EngineWorkspace {
public:
  class Lease {
  public:
    Lease() = default;
    Lease(EngineWorkspace& owner, EngineWorkspaceSlot slot) noexcept;
    ~Lease();

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;

    const EngineWorkspaceSlot& get() const noexcept {
      return slot_;
    }

  private:
    void reset() noexcept;

    EngineWorkspace* owner_ {nullptr};
    EngineWorkspaceSlot slot_;
  };

  void configure(std::size_t cover_size, std::size_t overlap_size, std::size_t spectrum_size);
  float* initial_overlap();
  std::complex<float>* initial_spectrum();
  Lease acquire();

private:
  void ensure_slot(std::size_t slot);
  void release(unsigned int id) noexcept;

  std::mutex mutex_;
  std::vector<int> slot_in_use_;
  std::vector<AlignedVector<byte>> cover_buffers_;
  std::vector<AlignedVector<float>> overlap_buffers_;
  std::vector<AlignedVector<std::complex<float>>> spectrum_buffers_;
  std::size_t cover_size_ {0};
  std::size_t overlap_size_ {0};
  std::size_t spectrum_size_ {0};
};

} // namespace neo_fft3d
