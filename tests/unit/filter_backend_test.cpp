#include "engine/filter_backend.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string_view>

TEST_CASE("CpuFilterBackend can be constructed independently of FFT planning", "[filter_backend]") {
  auto backend = neo_fft3d::engine::CreateCpuFilterBackend();

  REQUIRE(backend != nullptr);
  REQUIRE(std::string_view(backend->Name()) == "cpu");
}
