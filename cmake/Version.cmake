find_package(Git REQUIRED)
execute_process(
  COMMAND ${GIT_EXECUTABLE} describe --first-parent --tags --always
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_REPO_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "^(r[0-9]+).*$" "\\1" VERSION "${GIT_REPO_VERSION}")
execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_COUNT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/version.hpp.in"
  "${GENERATED_DIR}/version.hpp"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/src/version.rc.in"
  "${GENERATED_DIR}/version.rc"
)
