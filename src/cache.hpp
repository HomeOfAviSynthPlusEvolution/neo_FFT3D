/*
 * Copyright 2020 Xinyue Lu
 *
 * LRU key-value cache.
 *
 */

#pragma once
#ifndef __CACHE_HPP__
#define __CACHE_HPP__

#include "fft3d_common.h"
#include <unordered_map>
#include <list>
#include <cstddef>

template<typename value_t>
class cache {
  public:
    typedef typename std::pair<int32_t, value_t*> key_value_pair_t;
    typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

    cache(size_t vault_size, size_t data_size) {
      auto alignment_size = 64 / sizeof(value_t);
      _data_size = (data_size + alignment_size - 1) & (~alignment_size);
      resize(vault_size);
    }

    ~cache() {
      for (auto &&i : _internal_vault)
        _aligned_free(i);
    }

    value_t* get_write(const int32_t key) {
      auto last = _cache_items_list.end();
      last--;
      auto data = last->second;
      _cache_items_map.erase(last->first);
      _cache_items_list.pop_back();
      _cache_items_list.push_front(key_value_pair_t(key, data));
      _cache_items_map[key] = _cache_items_list.begin();
      return data;
    }

    value_t* get_read(const int32_t key) {
      auto it = _cache_items_map.find(key);
      if (it == _cache_items_map.end())
        return NULL;

      _cache_items_list.splice(_cache_items_list.begin(), _cache_items_list, it->second);
      return it->second->second;
    }

    bool exists(const int32_t key) const {
      return _cache_items_map.find(key) != _cache_items_map.end();
    }

    bool refresh(const int32_t key) {
      bool key_exists = exists(key);
      if (key_exists)
        _cache_items_list.splice(_cache_items_list.begin(), _cache_items_list, _cache_items_map[key]);
      return key_exists;
    }

    void resize(size_t new_vault_size) {
      while (_internal_vault.size() < new_vault_size) {
        auto mem = (value_t*)_aligned_malloc(sizeof(value_t) * _data_size, FRAME_ALIGN);
        _internal_vault.push_back(mem);
        _cache_items_list.push_back(key_value_pair_t(-1, mem));
      }
    }

  private:
    std::list<key_value_pair_t> _cache_items_list;
    std::unordered_map<int32_t, list_iterator_t> _cache_items_map;
    std::vector<value_t*> _internal_vault;
    size_t _data_size {0};
};

#endif
