#include <catch2/catch_test_macros.hpp>
#include "cache.hpp"

TEST_CASE("Cache alignment logic conforms to power-of-two padding", "[cache]") {
    // float has size 4. alignment_size = 64 / 4 = 16.
    // _data_size should be rounded up to nearest multiple of 16.
    // If requested 1, should be 16.
    cache<float> c(1, 1);

    // Test write returns aligned memory.
    float* ptr = c.get_write(100);
    REQUIRE(ptr != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ptr) % 64 == 0); // FRAME_ALIGN is 64
}

TEST_CASE("Cache LRU eviction and access ordering", "[cache]") {
    // capacity: 3, data_size: 1 float
    cache<float> c(3, 1);

    // Fill capacity
    float* ptr1 = c.get_write(1);
    *ptr1 = 1.0f;
    float* ptr2 = c.get_write(2);
    *ptr2 = 2.0f;
    float* ptr3 = c.get_write(3);
    *ptr3 = 3.0f;

    // Hit order: 1 is oldest (LRU), 3 is newest (MRU)
    // Read 1 to make it MRU
    float* r1 = c.get_read(1);
    REQUIRE(r1 != nullptr);
    REQUIRE(*r1 == 1.0f);

    // Now 2 is LRU. Write 4 should evict 2.
    float* ptr4 = c.get_write(4);
    *ptr4 = 4.0f;

    // Verify 2 is gone, 1, 3, 4 are present
    REQUIRE(c.get_read(2) == nullptr);
    REQUIRE(c.get_read(1) != nullptr);
    REQUIRE(c.get_read(3) != nullptr);
    REQUIRE(c.get_read(4) != nullptr);
}

TEST_CASE("Cache refresh explicitly updates MRU", "[cache]") {
    cache<float> c(2, 1);
    c.get_write(10);
    c.get_write(20);

    // 10 is LRU. Refresh 10 to make it MRU.
    bool refreshed = c.refresh(10);
    REQUIRE(refreshed == true);

    // Write 30 should now evict 20 (since 10 is MRU)
    c.get_write(30);

    REQUIRE(c.get_read(20) == nullptr);
    REQUIRE(c.get_read(10) != nullptr);
}

TEST_CASE("Cache resize preserves existing keys and consumes new slots first", "[cache]") {
    cache<float> c(2, 1);
    *c.get_write(1) = 1.0f;
    *c.get_write(2) = 2.0f;

    c.resize(4);

    *c.get_write(3) = 3.0f;
    *c.get_write(4) = 4.0f;

    // The two writes after resize should use the newly added empty slots.
    *c.get_write(5) = 5.0f;

    REQUIRE(c.get_read(1) == nullptr);
    float* key2 = c.get_read(2);
    REQUIRE(key2 != nullptr);
    REQUIRE(*key2 == 2.0f);
    float* key3 = c.get_read(3);
    REQUIRE(key3 != nullptr);
    REQUIRE(*key3 == 3.0f);
    float* key4 = c.get_read(4);
    REQUIRE(key4 != nullptr);
    REQUIRE(*key4 == 4.0f);
    float* key5 = c.get_read(5);
    REQUIRE(key5 != nullptr);
    REQUIRE(*key5 == 5.0f);
}
