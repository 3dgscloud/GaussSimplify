cmake_minimum_required(VERSION 3.26)
project(gauss_simplify
    VERSION 0.1.0
    DESCRIPTION "3D Gaussian Splat simplification library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gs/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/gs/version.h
    @ONLY
)

# --------------------------------------------------------
# 1. GaussForge (format conversion library via FetchContent)
# --------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    gauss_forge
    GIT_REPOSITORY https://github.com/3dgscloud/GaussForge.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(gauss_forge)

# --------------------------------------------------------
# 2. OpenMP (desktop parallelism; skipped on WASM)
# --------------------------------------------------------
if(NOT EMSCRIPTEN)
    find_package(OpenMP REQUIRED)
endif()

# --------------------------------------------------------
# 3. nanoflann (header-only KD-tree, WASM compatible)
# --------------------------------------------------------
add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# --------------------------------------------------------
# 4. Simplify library
# --------------------------------------------------------
add_library(gauss_simplify_lib STATIC
    src/simplify.cpp
)
target_include_directories(gauss_simplify_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(gauss_simplify_lib
    PUBLIC gauss_forge
    PRIVATE nanoflann
)
if(NOT EMSCRIPTEN AND OpenMP_CXX_FOUND)
    target_link_libraries(gauss_simplify_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# --------------------------------------------------------
# 5. CLI executable (not built for WASM)
# --------------------------------------------------------
if(NOT EMSCRIPTEN)
    add_executable(gauss_simplify src/main.cpp)
    target_link_libraries(gauss_simplify PRIVATE gauss_simplify_lib)
endif()

# --------------------------------------------------------
# 6. Benchmarks (not built for WASM)
# --------------------------------------------------------
if(NOT EMSCRIPTEN)
    add_subdirectory(benchmarks)
endif()
