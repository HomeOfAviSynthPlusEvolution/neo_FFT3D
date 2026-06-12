#include <catch2/catch_test_macros.hpp>
#include "cache.hpp"

TEST_CASE("Cache alignment logic conforms to power-of-two padding", "[cache]") {
    // float has size 4. alignment_size = 64 / 4 = 16.
    // _data_size should be rounded up to nearest multiple of 16.
    // If requested 1, should be 16.
    cache<float> c(1, 1);

    // Test write returns aligned memory.
    auto ptr = c.get_write(100);
    REQUIRE(ptr != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ptr->data()) % 64 == 0); // FRAME_ALIGN is 64
}

TEST_CASE("Cache LRU eviction and access ordering", "[cache]") {
    // capacity: 3, data_size: 1 float
    cache<float> c(3, 1);

    // Fill capacity
    {
        auto ptr1 = c.get_write(1);
        c.publish(ptr1, 1);
        (*ptr1)[0] = 1.0f;
        auto ptr2 = c.get_write(2);
        c.publish(ptr2, 2);
        (*ptr2)[0] = 2.0f;
        auto ptr3 = c.get_write(3);
        c.publish(ptr3, 3);
        (*ptr3)[0] = 3.0f;
    }

    // Hit order: 1 is oldest (LRU), 3 is newest (MRU)
    // Read 1 to make it MRU
    {
        auto r1 = c.get_read(1);
        REQUIRE(r1 != nullptr);
        REQUIRE((*r1)[0] == 1.0f);
    }

    // Now 2 is LRU. Write 4 should evict 2.
    {
        auto ptr4 = c.get_write(4);
        c.publish(ptr4, 4);
        (*ptr4)[0] = 4.0f;
    }

    // Verify 2 is gone, 1, 3, 4 are present
    REQUIRE(c.get_read(2) == nullptr);
    REQUIRE(c.get_read(1) != nullptr);
    REQUIRE(c.get_read(3) != nullptr);
    REQUIRE(c.get_read(4) != nullptr);
}

TEST_CASE("Cache refresh explicitly updates MRU", "[cache]") {
    cache<float> c(2, 1);
    
    auto w10 = c.get_write(10);
    c.publish(w10, 10);
    auto w20 = c.get_write(20);
    c.publish(w20, 20);
    
    // Release leases so eviction can happen
    w10.reset();
    w20.reset();

    // 10 is LRU. Refresh 10 to make it MRU.
    bool refreshed = c.refresh(10);
    REQUIRE(refreshed == true);

    // Write 30 should now evict 20 (since 10 is MRU)
    auto w30 = c.get_write(30);
    c.publish(w30, 30);
    w30.reset();

    REQUIRE(c.get_read(20) == nullptr);
    REQUIRE(c.get_read(10) != nullptr);
}

TEST_CASE("Cache resize preserves existing keys and consumes new slots first", "[cache]") {
    cache<float> c(2, 1);
    auto w1 = c.get_write(1);
    c.publish(w1, 1);
    (*w1)[0] = 1.0f;
    
    auto w2 = c.get_write(2);
    c.publish(w2, 2);
    (*w2)[0] = 2.0f;
    
    w1.reset();
    w2.reset();

    c.resize(4);

    auto w3 = c.get_write(3);
    c.publish(w3, 3);
    (*w3)[0] = 3.0f;
    
    auto w4 = c.get_write(4);
    c.publish(w4, 4);
    (*w4)[0] = 4.0f;

    w3.reset();
    w4.reset();

    // The two writes after resize should use the newly added empty slots.
    auto w5 = c.get_write(5);
    c.publish(w5, 5);
    (*w5)[0] = 5.0f;
    w5.reset();

    REQUIRE(c.get_read(1) == nullptr);
    auto key2 = c.get_read(2);
    REQUIRE(key2 != nullptr);
    REQUIRE((*key2)[0] == 2.0f);
    auto key3 = c.get_read(3);
    REQUIRE(key3 != nullptr);
    REQUIRE((*key3)[0] == 3.0f);
    auto key4 = c.get_read(4);
    REQUIRE(key4 != nullptr);
    REQUIRE((*key4)[0] == 4.0f);
    auto key5 = c.get_read(5);
    REQUIRE(key5 != nullptr);
    REQUIRE((*key5)[0] == 5.0f);
}

TEST_CASE("Cache concurrency lease safety", "[cache]") {
    cache<float> c(2, 1);
    
    // Thread A gets a write lease but doesn't publish yet
    auto lease_A = c.get_write(10);
    
    // Thread B tries to get a write lease. Since lease_A is active (use_count > 1), 
    // it shouldn't evict the slot used by lease_A. Instead it uses the second empty slot.
    auto lease_B = c.get_write(20);
    
    // Thread C tries to get a write lease. All slots are leased!
    // The cache should dynamically resize to accommodate.
    auto lease_C = c.get_write(30);
    
    REQUIRE(lease_A != nullptr);
    REQUIRE(lease_B != nullptr);
    REQUIRE(lease_C != nullptr);
    REQUIRE(lease_A != lease_B);
    REQUIRE(lease_A != lease_C);
    REQUIRE(lease_B != lease_C);
    
    // Publish them
    c.publish(lease_A, 10);
    c.publish(lease_B, 20);
    c.publish(lease_C, 30);
    
    // Drop leases
    lease_A.reset();
    lease_B.reset();
    lease_C.reset();
    
    // All 3 should be readable because the cache resized
    REQUIRE(c.get_read(10) != nullptr);
    REQUIRE(c.get_read(20) != nullptr);
    REQUIRE(c.get_read(30) != nullptr);
}
