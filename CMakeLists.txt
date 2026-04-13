cmake_minimum_required(VERSION 3.26)
project(gauss_simplify
    VERSION 0.1.3
    DESCRIPTION "3D Gaussian Splat simplification library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

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
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FetchContent updates while configuring" FORCE)

set(GAUSS_FORGE_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../GaussForge")
if(EXISTS "${GAUSS_FORGE_LOCAL_DIR}/CMakeLists.txt")
    FetchContent_Declare(
        gauss_forge
        SOURCE_DIR "${GAUSS_FORGE_LOCAL_DIR}"
    )
else()
    FetchContent_Declare(
        gauss_forge
        GIT_REPOSITORY https://github.com/3dgscloud/GaussForge.git
        GIT_TAG        v0.5.5
    )
endif()
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
# 4. Compilation options optimization
# --------------------------------------------------------
if(MSVC)
    add_compile_options(/W3)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_definitions(-DDEBUG_BUILD)
    else()
        add_definitions(-DRELEASE_BUILD)
        add_compile_options(/O2 /GL)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF /LTCG")
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
    endif()

    # Enable sccache if available
    find_program(SCCACHE_PROGRAM sccache)
    if(SCCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_definitions(-DDEBUG_BUILD)
    else()
        add_definitions(-DRELEASE_BUILD)
        # GCC/Clang Release optimizations
        add_compile_options(-O3 -flto -funroll-loops -fomit-frame-pointer -ftree-vectorize)

        if(NOT EMSCRIPTEN)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

# --------------------------------------------------------
# 5. Simplify library
# --------------------------------------------------------
add_library(gauss_simplify_lib STATIC
    src/simplify.cpp
    src/simplify_activate.cpp
    src/simplify_prune.cpp
    src/simplify_merge.cpp
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
# 6. WASM build (Emscripten only)
# --------------------------------------------------------
if(EMSCRIPTEN)
    include(cmakes/wasm.cmake)

    add_executable(gauss_simplify_wasm src/wasm/bindings.cpp)
    target_link_libraries(gauss_simplify_wasm PRIVATE gauss_simplify_lib)
    target_link_options(gauss_simplify_wasm PRIVATE ${GAUSS_SIMPLIFY_WASM_LINK_OPTIONS})

    set_target_properties(gauss_simplify_wasm PROPERTIES
        OUTPUT_NAME "gauss_simplify"
        SUFFIX ".js"
    )
endif()

# --------------------------------------------------------
# 7. CLI executable (not built for WASM)
# --------------------------------------------------------
if(NOT EMSCRIPTEN)
    add_executable(gauss_simplify src/main.cpp)
    target_link_libraries(gauss_simplify PRIVATE gauss_simplify_lib)
endif()

# --------------------------------------------------------
# 8. Benchmarks (not built for WASM)
# --------------------------------------------------------
if(NOT EMSCRIPTEN)
    add_subdirectory(benchmarks)
endif()

# --------------------------------------------------------
# 9. Tests
# --------------------------------------------------------
if(BUILD_TESTING AND NOT EMSCRIPTEN)
    add_subdirectory(tests)
endif()
