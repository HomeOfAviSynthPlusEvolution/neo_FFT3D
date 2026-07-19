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
#include <memory>
#include <mutex>
#include <type_traits>

template<typename value_t>
class cache {
  public:
    using element_t = typename std::remove_extent<value_t>::type;
    struct Item {
      std::shared_ptr<element_t> data;
      std::shared_ptr<std::mutex> mtx;
    };
    typedef typename std::pair<int32_t, Item> key_value_pair_t;
    typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

    cache(size_t vault_size, size_t data_size) {
      auto alignment_size = 64 / sizeof(value_t);
      _data_size = (data_size + alignment_size - 1) & (~alignment_size);
      _max_size = vault_size;
    }

    ~cache() {}

    Item get_write(const int32_t key) {
      if (_cache_items_list.size() >= _max_size) {
        auto last = _cache_items_list.end();
        last--;
        _cache_items_map.erase(last->first);
        _cache_items_list.pop_back();
      }
      auto deleter = [](element_t* ptr) { _aligned_free(ptr); };
      std::shared_ptr<element_t> mem((element_t*)_aligned_malloc(sizeof(value_t) * _data_size, FRAME_ALIGN), deleter);
      Item item{mem, std::make_shared<std::mutex>()};
        
      _cache_items_list.push_front(key_value_pair_t(key, item));
      _cache_items_map[key] = _cache_items_list.begin();
      return item;
    }

    Item get_read(const int32_t key) {
      auto it = _cache_items_map.find(key);
      if (it == _cache_items_map.end())
        return {nullptr, nullptr};

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
      _max_size = new_vault_size;
    }

  private:
    std::list<key_value_pair_t> _cache_items_list;
    std::unordered_map<int32_t, list_iterator_t> _cache_items_map;
    size_t _data_size{0};
    size_t _max_size{0};
};

#endif
