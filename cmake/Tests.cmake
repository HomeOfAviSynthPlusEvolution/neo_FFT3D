option(NEO_FFT3D_BUILD_TEST_TOOLS "Build Neo_FFT3D baseline test tools" ON)

if (NEO_FFT3D_BUILD_TEST_TOOLS)
  add_executable(neo-fft3d_avs_frame_dump
    tests/tools/avs_frame_dump.cpp
  )
  set_property(TARGET neo-fft3d_avs_frame_dump PROPERTY CXX_STANDARD 17)
  target_link_libraries(neo-fft3d_avs_frame_dump PRIVATE avisynth_headers)

  if (NOT WIN32)
    target_link_libraries(neo-fft3d_avs_frame_dump PRIVATE dl)
  endif()
endif()

if (NEO_FFT3D_BUILD_TEST_TOOLS)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  add_test(
    NAME neo-fft3d_baseline_unit_tests
    COMMAND "${Python3_EXECUTABLE}" -m unittest tests.baseline.test_baseline_runner -v
  )
  set_tests_properties(neo-fft3d_baseline_unit_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(neo-fft3d_cache_test tests/unit/cache_test.cpp)
  target_include_directories(neo-fft3d_cache_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(neo-fft3d_cache_test PRIVATE Catch2::Catch2WithMain)
  add_test(NAME neo-fft3d_cache_test COMMAND "$<TARGET_FILE:neo-fft3d_cache_test>")

  add_executable(neo-fft3d_frame_buffer_test
    tests/unit/frame_buffer_test.cpp
    src/engine/frame_cover.cpp
    src/engine/overlap_transform.cpp
  )
  target_include_directories(neo-fft3d_frame_buffer_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(neo-fft3d_frame_buffer_test PRIVATE Catch2::Catch2WithMain DualSynth::dualsynth)
  add_test(NAME neo-fft3d_frame_buffer_test COMMAND "$<TARGET_FILE:neo-fft3d_frame_buffer_test>")

  add_executable(neo-fft3d_filter_backend_test
    tests/unit/filter_backend_test.cpp
    src/engine/cpu_filter_backend.cpp
    src/core/cpu/cpu_dispatch.cpp
    src/core/cpu/apply_hwy.cpp
    src/core/cpu/kalman_hwy.cpp
    src/core/cpu/sharpen_hwy.cpp
    src/code_impl/Apply.cpp
    src/code_impl/Kalman.cpp
    src/code_impl/Sharpen.cpp
  )
  target_include_directories(neo-fft3d_filter_backend_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(neo-fft3d_filter_backend_test PRIVATE Catch2::Catch2WithMain DualSynth::dualsynth hwy)
  add_test(NAME neo-fft3d_filter_backend_test COMMAND "$<TARGET_FILE:neo-fft3d_filter_backend_test>")

  add_executable(neo-fft3d_c_apply_null_slots_test
    tests/unit/c_apply_null_slots_test.cpp
    src/code_impl/Apply.cpp
  )
  target_include_directories(neo-fft3d_c_apply_null_slots_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(neo-fft3d_c_apply_null_slots_test PRIVATE DualSynth::dualsynth)

  add_test(
    NAME neo-fft3d_c_apply_null_slots_test
    COMMAND "$<TARGET_FILE:neo-fft3d_c_apply_null_slots_test>"
  )

  add_test(
    NAME neo-fft3d_baseline_smoke_verification
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/baseline_runner.py" verify
      --plugin "$<TARGET_FILE:neo-fft3d>"
      --avs-dump "$<TARGET_FILE:neo-fft3d_avs_frame_dump>"
      --golden "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/golden/smoke.json"
      --tier smoke
      --hosts vs avs
  )
  set_tests_properties(neo-fft3d_baseline_smoke_verification PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")

  add_test(
    NAME neo-fft3d_baseline_compat_verification
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/baseline_runner.py" verify
      --plugin "$<TARGET_FILE:neo-fft3d>"
      --avs-dump "$<TARGET_FILE:neo-fft3d_avs_frame_dump>"
      --golden "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/golden/compat.json"
      --tier compat
      --hosts vs avs
  )
  set_tests_properties(neo-fft3d_baseline_compat_verification PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
endif()
