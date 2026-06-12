/*
 * Copyright 2020 Xinyue Lu
 * Modernized 2026: std::vector flat LRU cache
 */

#pragma once
#ifndef __CACHE_HPP__
#define __CACHE_HPP__

#include "fft3d_common.h"
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>

template<typename value_t>
class cache {
  private:
    struct CacheNode {
        int32_t key;
        value_t* data;
    };
    std::vector<CacheNode> _vault;
    size_t _data_size {0};

  public:
    // Rule of Five: Move-Only Resource
    cache(const cache&) = delete;
    cache& operator=(const cache&) = delete;
    cache(cache&&) = delete;
    cache& operator=(cache&&) = delete;

    cache(size_t vault_size, size_t data_size) {
      static_assert(FRAME_ALIGN % sizeof(value_t) == 0, "value_t size must divide FRAME_ALIGN evenly.");
      auto alignment_size = FRAME_ALIGN / sizeof(value_t);
      _data_size = (data_size + alignment_size - 1) & ~(alignment_size - 1);
      resize(vault_size);
    }

    ~cache() {
      for (auto& node : _vault) {
        if (node.data) {
          _aligned_free(node.data);
        }
      }
    }

    value_t* get_read(const int32_t key) {
      auto it = std::find_if(_vault.begin(), _vault.end(), [key](const CacheNode& n) { return n.key == key; });
      if (it == _vault.end()) return nullptr;

      // LRU logic: rotate to end (MRU)
      std::rotate(it, it + 1, _vault.end());
      return _vault.back().data;
    }

    value_t* get_write(const int32_t key) {
      // LRU node is always at the front
      auto it = _vault.begin();
      value_t* data = it->data;
      it->key = key;

      // Rotate to end (MRU)
      std::rotate(it, it + 1, _vault.end());
      return data;
    }

    bool refresh(const int32_t key) {
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
        auto mem = (value_t*)_aligned_malloc(sizeof(value_t) * _data_size, FRAME_ALIGN);
        // Insert empty frames (-1) at the FRONT, ensuring they are LRU and immediately ready for get_write
        _vault.insert(_vault.begin(), CacheNode{-1, mem});
      }
    }
};

#endif
