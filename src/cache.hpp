/*
 * Copyright 2020 Xinyue Lu
 * Modernized 2026: std::vector flat LRU cache with Lease-Based concurrency protection
 */

#pragma once
#ifndef __CACHE_HPP__
#define __CACHE_HPP__

#include "fft3d_common.h"
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

template<typename value_t>
class cache {
  private:
    struct CacheNode {
        std::int32_t key;
        std::shared_ptr<AlignedVector<value_t>> data;
    };
    std::vector<CacheNode> _vault;
    size_t _data_size {0};

  public:
    cache(const cache&) = delete;
    cache& operator=(const cache&) = delete;
    cache(cache&&) = delete;
    cache& operator=(cache&&) = delete;

    cache(size_t vault_size, size_t data_size) {
      _data_size = data_size;
      resize(vault_size);
    }

    ~cache() = default;

    std::shared_ptr<AlignedVector<value_t>> get_read(const std::int32_t key) {
      auto it = std::find_if(_vault.begin(), _vault.end(), [key](const CacheNode& n) { return n.key == key; });
      if (it == _vault.end()) return nullptr;

      std::rotate(it, it + 1, _vault.end());
      return _vault.back().data;
    }

    std::shared_ptr<AlignedVector<value_t>> get_write(const std::int32_t key) {
      auto it = std::find_if(_vault.begin(), _vault.end(),
                             [](const CacheNode& n) { return n.data.use_count() == 1; });

      if (it == _vault.end()) {
        auto buf = std::make_shared<AlignedVector<value_t>>(_data_size);
        _vault.insert(_vault.begin(), CacheNode{-1, std::move(buf)});
        it = _vault.begin();
      }

      // Invalidate the old key immediately. The new key is only visible after publish().
      it->key = -1;
      std::rotate(it, it + 1, _vault.end());
      return _vault.back().data;
    }

    // New API to publish the key after the data is fully written
    void publish(const std::shared_ptr<AlignedVector<value_t>>& data, const std::int32_t key) {
        auto it = std::find_if(_vault.begin(), _vault.end(),
                               [&](const CacheNode& n) { return n.data == data; });
        if (it != _vault.end()) {
            it->key = key;
        }
    }

    bool refresh(const std::int32_t key) {
      auto it = std::find_if(_vault.begin(), _vault.end(), [key](const CacheNode& n) { return n.key == key; });
      if (it != _vault.end()) {
        std::rotate(it, it + 1, _vault.end());
        return true;
      }
      return false;
    }

    void resize(size_t new_vault_size) {
      if (new_vault_size <= _vault.size()) return;

      _vault.reserve(new_vault_size);
      size_t to_add = new_vault_size - _vault.size();
      for (size_t i = 0; i < to_add; ++i) {
        _vault.insert(_vault.begin(), CacheNode{-1, std::make_shared<AlignedVector<value_t>>(_data_size)});
      }
    }
};

#endif
