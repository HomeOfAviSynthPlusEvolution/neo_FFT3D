/*
 * Copyright 2020 Xinyue Lu
 *
 * LRU key-value cache.
 *
 */

#ifndef __CACHE_HPP__
#define __CACHE_HPP__

#include "common.h"
#include <unordered_map>
#include <list>
#include <cstddef>

template<typename value_t>
class cache {
  public:
    typedef typename std::pair<int32_t, value_t*> key_value_pair_t;
    typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

    cache(size_t vault_size, size_t data_size) :
      _vault_size(vault_size) {
      auto alignment_size = 64 / sizeof(value_t);
      data_size = (data_size + alignment_size - 1) & (~alignment_size);
      _internal_vault = new value_t*[vault_size];
      for (size_t i = 0; i < vault_size; i++) {
        _internal_vault[i] = (value_t*)_aligned_malloc(sizeof(value_t) * data_size, 64);
        _cache_items_list.push_front(key_value_pair_t(-1, _internal_vault[i]));
      }
    }

    ~cache() {
      for (size_t i = 0; i < _vault_size; i++) {
        _aligned_free(_internal_vault[i]);
      }
      delete[] _internal_vault;
      _internal_vault = NULL;
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

  private:
    std::list<key_value_pair_t> _cache_items_list;
    std::unordered_map<int32_t, list_iterator_t> _cache_items_map;
    value_t** _internal_vault;
    size_t _vault_size;
};

#endif
