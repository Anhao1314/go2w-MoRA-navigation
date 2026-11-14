cmake_minimum_required(VERSION 3.20)
project(go2w_control VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/utf-8)
  add_compile_definitions(NOMINMAX)
endif()

option(GO2W_BUILD_TESTS "Build C++ tests" ON)
option(GO2W_BUILD_PYTHON "Build the pybind11 module" OFF)
option(GO2W_BUILD_BENCHMARKS "Build benchmark apps" ON)

include(FetchContent)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found; fetching Eigen 3.4.0")
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

find_package(osqp QUIET)
if(NOT osqp_FOUND)
  message(STATUS "OSQP not found; fetching OSQP 0.6.3")
  FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v0.6.3
    GIT_SHALLOW TRUE
    GIT_SUBMODULES "lin_sys/direct/qdldl/qdldl_sources"
  )
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(OSQP_USE_LONG OFF CACHE BOOL "" FORCE)
  set(ENABLE_MKL_PARDISO OFF CACHE BOOL "" FORCE)
  # OSQP 0.6.3 predates CMake 4's removal of <3.5 compatibility.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
  FetchContent_MakeAvailable(osqp)
endif()

# OSQP 0.6.3 gives both the static and shared targets the same OUTPUT_NAME.
# That is fine on Unix but collides on MSVC (both produce osqp.lib).
if(MSVC)
  if(TARGET osqpstatic)
    set_target_properties(osqpstatic PROPERTIES OUTPUT_NAME osqp_static)
  endif()
  if(TARGET osqp)
    set_target_properties(osqp PROPERTIES OUTPUT_NAME osqp_shared)
  endif()
  if(TARGET qdldlstatic)
    set_target_properties(qdldlstatic PROPERTIES OUTPUT_NAME qdldl_static)
  endif()
  if(TARGET qdldl)
    set_target_properties(qdldl PROPERTIES OUTPUT_NAME qdldl_shared)
  endif()
endif()

add_library(go2w_core STATIC
  src/model.cpp
  src/kinematics.cpp
  src/dynamics.cpp
  src/path_tracker.cpp
  src/force_mapper.cpp
  src/safety.cpp
  src/mpc.cpp
  src/controller.cpp
)
# The pybind11 shared object (python-binding CI job) links this static
# library: without PIC the shared-object link fails with R_X86_64_PC32
# relocation errors against the generated-constants symbols.
set_property(TARGET go2w_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(go2w_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(TARGET Eigen3::Eigen)
  target_link_libraries(go2w_core PUBLIC Eigen3::Eigen)
else()
  target_link_libraries(go2w_core PUBLIC eigen)
endif()

if(TARGET osqpstatic)
  target_link_libraries(go2w_core PUBLIC osqpstatic)
elseif(TARGET osqp)
  target_link_libraries(go2w_core PUBLIC osqp)
else()
  message(FATAL_ERROR "OSQP target not found after FetchContent")
endif()

if(MSVC)
  target_compile_options(go2w_core PRIVATE /W4)
else()
  target_compile_options(go2w_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GO2W_BUILD_BENCHMARKS)
  add_executable(go2w_replay apps/go2w_replay.cpp)
  target_link_libraries(go2w_replay PRIVATE go2w_core)

  add_executable(go2w_mpc_bench apps/go2w_mpc_bench.cpp)
  target_link_libraries(go2w_mpc_bench PRIVATE go2w_core)
endif()

if(GO2W_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found; fetching Catch2 3.7.1")
    FetchContent_Declare(
      catch2
      URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(catch2)
  endif()

  add_executable(go2w_tests
    tests/test_kinematics.cpp
    tests/test_dynamics.cpp
    tests/test_mpc.cpp
    tests/test_controller.cpp
  )
  target_link_libraries(go2w_tests PRIVATE go2w_core Catch2::Catch2WithMain)
  target_compile_definitions(go2w_tests PRIVATE
    GO2W_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
  )
  add_test(NAME go2w_tests COMMAND go2w_tests)
endif()

if(GO2W_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(go2w_py bindings/pybind_module.cpp)
  target_link_libraries(go2w_py PRIVATE go2w_core)
  install(TARGETS go2w_py LIBRARY DESTINATION . RUNTIME DESTINATION .)
endif()

# Guardrail: C++ verifiable-arbitration core (architecture v0.2)
add_subdirectory(guardrail)
