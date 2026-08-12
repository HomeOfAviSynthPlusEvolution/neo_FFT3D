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

#ifndef FRAME_ALIGN
#define FRAME_ALIGN 64
#endif

template<typename value_t>
class cache {
public:
  using element_t = std::remove_extent_t<value_t>;

  struct Item {
    std::shared_ptr<element_t> data;
    std::shared_ptr<std::mutex> mtx;
  };

private:
  static std::size_t normalize_alignment(std::size_t alignment) noexcept
  {
    if (alignment == 0)
      alignment = alignof(std::max_align_t);

    std::size_t power_of_two = 1;
    while (power_of_two < alignment)
      power_of_two <<= 1;

    const std::size_t minimum = alignof(std::max_align_t);
    return (power_of_two < minimum) ? minimum : power_of_two;
  }

  static void* allocate_aligned(std::size_t bytes, std::size_t alignment)
  {
    alignment = normalize_alignment(alignment);
    if (bytes == 0)
      bytes = alignment;

    return ::operator new(bytes, std::align_val_t{ alignment });
  }

  static void deallocate_aligned(void* p, std::size_t bytes, std::size_t alignment) noexcept
  {
    if (!p)
      return;

    alignment = normalize_alignment(alignment);
    (void)bytes;

    ::operator delete(p, std::align_val_t{ alignment });
  }

  // One pooled block contains the aligned FFT buffer and the mutex used to
  // serialize first-time computation of that cached frame/block.
  struct Block {
    element_t* mem;
    std::mutex mtx;
    Block* next_free;

    Block() noexcept
      : mem(nullptr)
      , next_free(nullptr)
    {
    }
  };

  struct PoolState {
    std::mutex mtx;
    Block* free_head;
    std::vector<Block*> all_blocks;
    std::size_t alloc_bytes;
    std::size_t alignment;

    PoolState() noexcept
      : free_head(nullptr)
      , alloc_bytes(0)
      , alignment(0)
    {
    }

    ~PoolState()
    {
      for (Block* block : all_blocks) {
        if (!block)
          continue;

        void* mem = block->mem;
        delete block;
        deallocate_aligned(mem, alloc_bytes, alignment);
      }
    }
  };

  struct BlockDeleter {
    std::shared_ptr<PoolState> state;

    void operator()(Block* block) const noexcept
    {
      if (!block || !state)
        return;

      std::lock_guard<std::mutex> lock(state->mtx);
      block->next_free = state->free_head;
      state->free_head = block;
    }
  };

  using key_value_pair_t = std::pair<int32_t, Item>;
  using list_iterator_t = typename std::list<key_value_pair_t>::iterator;

  std::list<key_value_pair_t> _cache_items_list;
  std::unordered_map<int32_t, list_iterator_t> _cache_items_map;

  std::size_t _alloc_bytes{ 0 };
  std::size_t _alignment{ 0 };
  std::atomic<std::size_t> _max_size{ 0 };
  std::shared_ptr<PoolState> _state;

  Block* allocate_block()
  {
    void* mem = allocate_aligned(_alloc_bytes, _alignment);

    Block* block = nullptr;
    try {
      block = new Block();
    }
    catch (...) {
      deallocate_aligned(mem, _alloc_bytes, _alignment);
      throw;
    }

    block->mem = static_cast<element_t*>(mem);

    {
      std::lock_guard<std::mutex> lock(_state->mtx);
      try {
        _state->all_blocks.push_back(block);
      }
      catch (...) {
        delete block;
        deallocate_aligned(mem, _alloc_bytes, _alignment);
        throw;
      }
    }

    return block;
  }

  Block* acquire_block()
  {
    {
      std::lock_guard<std::mutex> lock(_state->mtx);

      if (_state->free_head) {
        Block* block = _state->free_head;
        _state->free_head = block->next_free;
        block->next_free = nullptr;
        return block;
      }
    }

    return allocate_block();
  }

  void release_block(Block* block) noexcept
  {
    if (!block)
      return;

    std::lock_guard<std::mutex> lock(_state->mtx);
    block->next_free = _state->free_head;
    _state->free_head = block;
  }

  Item make_item(Block* block)
  {
    std::shared_ptr<Block> holder;
    try {
      holder = std::shared_ptr<Block>(block, BlockDeleter{ _state });
    }
    catch (...) {
      release_block(block);
      throw;
    }

    return Item{
      std::shared_ptr<element_t>(holder, block->mem),
      std::shared_ptr<std::mutex>(holder, &block->mtx)
    };
  }

public:
  cache(std::size_t vault_size, std::size_t data_size)
    : _max_size(vault_size)
  {
    _alignment = normalize_alignment(FRAME_ALIGN);

    std::size_t bytes = data_size * sizeof(value_t);
    if (bytes == 0)
      bytes = 64;

    const std::size_t min_bytes = (_alignment > 64) ? _alignment : 64;
    _alloc_bytes = (bytes + min_bytes - 1) & ~(min_bytes - 1);

    _state = std::make_shared<PoolState>();
    _state->alloc_bytes = _alloc_bytes;
    _state->alignment = _alignment;
    _state->all_blocks.reserve(vault_size);

    for (std::size_t i = 0; i < vault_size; ++i) {
      Block* block = allocate_block();
      release_block(block);
    }
  }

  ~cache() = default;

  cache(const cache&) = delete;
  cache& operator=(const cache&) = delete;

  Item get_write(const int32_t key)
  {
    const std::size_t max_size = _max_size.load(std::memory_order_relaxed);
    if (max_size == 0)
      return Item{ nullptr, nullptr };

    auto existing = _cache_items_map.find(key);
    if (existing != _cache_items_map.end()) {
      _cache_items_list.splice(
        _cache_items_list.begin(),
        _cache_items_list,
        existing->second
      );
      return existing->second->second;
    }

    while (_cache_items_list.size() >= max_size) {
      auto last = _cache_items_list.end();
      --last;

      _cache_items_map.erase(last->first);
      _cache_items_list.pop_back();
    }

    Block* block = acquire_block();
    Item item = make_item(block);

    _cache_items_list.push_front(key_value_pair_t(key, item));

    try {
      auto inserted = _cache_items_map.emplace(
        key,
        _cache_items_list.begin()
      );

      if (!inserted.second) {
        _cache_items_list.pop_front();

        _cache_items_list.splice(
            _cache_items_list.begin(),
            _cache_items_list,
            inserted.first->second
        );

        return inserted.first->second->second;
      }
    }
    catch (...) {
      _cache_items_list.pop_front();
      throw;
    }

    return _cache_items_list.front().second;
  }

  Item get_read(const int32_t key)
  {
    auto it = _cache_items_map.find(key);
    if (it == _cache_items_map.end())
      return Item{ nullptr, nullptr };

    _cache_items_list.splice(
      _cache_items_list.begin(),
      _cache_items_list,
      it->second
    );

    return it->second->second;
  }

  bool exists(const int32_t key) const
  {
    return _cache_items_map.find(key) != _cache_items_map.end();
  }

  bool refresh(const int32_t key)
  {
    auto it = _cache_items_map.find(key);
    if (it == _cache_items_map.end())
      return false;

    _cache_items_list.splice(
      _cache_items_list.begin(),
      _cache_items_list,
      it->second
    );

    return true;
  }

  void resize(std::size_t new_vault_size)
  {
    _max_size.store(new_vault_size, std::memory_order_relaxed);
  }
};

#endif
