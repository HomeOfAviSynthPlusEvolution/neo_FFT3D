#include "engine/engine_workspace.hpp"

#include <algorithm>
#include <utility>

namespace neo_fft3d {

EngineWorkspace::Lease::Lease(EngineWorkspace& owner, EngineWorkspaceSlot slot) noexcept
  : owner_(&owner), slot_(slot) {}

EngineWorkspace::Lease::~Lease() {
  reset();
}

EngineWorkspace::Lease::Lease(Lease&& other) noexcept
  : owner_(std::exchange(other.owner_, nullptr)), slot_(other.slot_) {}

EngineWorkspace::Lease& EngineWorkspace::Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    reset();
    owner_ = std::exchange(other.owner_, nullptr);
    slot_ = other.slot_;
  }
  return *this;
}

void EngineWorkspace::Lease::reset() noexcept {
  if (owner_) {
    owner_->release(slot_.id);
    owner_ = nullptr;
  }
}

void EngineWorkspace::configure(std::size_t cover_size, std::size_t overlap_size, std::size_t spectrum_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  cover_size_ = cover_size;
  overlap_size_ = overlap_size;
  spectrum_size_ = spectrum_size;
  ensure_slot(0);
}

float* EngineWorkspace::initial_overlap() {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_slot(0);
  return overlap_buffers_[0].data();
}

std::complex<float>* EngineWorkspace::initial_spectrum() {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_slot(0);
  return spectrum_buffers_[0].data();
}

EngineWorkspace::Lease EngineWorkspace::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = std::ranges::find(slot_in_use_, 0);
  const auto slot = static_cast<unsigned int>(std::distance(slot_in_use_.begin(), it));
  if (it == slot_in_use_.end()) {
    slot_in_use_.push_back(1);
  }
  else {
    *it = 1;
  }

  ensure_slot(slot);
  return Lease(*this, EngineWorkspaceSlot{
    .id = slot,
    .cover = cover_buffers_[slot].data(),
    .overlap = overlap_buffers_[slot].data(),
    .spectrum = spectrum_buffers_[slot].data(),
    .slot_count = slot_in_use_.size()
  });
}

void EngineWorkspace::ensure_slot(std::size_t slot) {
  while (cover_buffers_.size() <= slot) {
    cover_buffers_.emplace_back(cover_size_);
  }
  while (overlap_buffers_.size() <= slot) {
    overlap_buffers_.emplace_back(overlap_size_);
  }
  while (spectrum_buffers_.size() <= slot) {
    spectrum_buffers_.emplace_back(spectrum_size_);
  }
}

void EngineWorkspace::release(unsigned int id) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (id < slot_in_use_.size()) {
    slot_in_use_[id] = 0;
  }
}

} // namespace neo_fft3d
