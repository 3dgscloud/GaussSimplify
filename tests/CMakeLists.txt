FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(gauss_simplify_tests
    simplify_tests.cpp
)

target_include_directories(gauss_simplify_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gauss_simplify_tests PRIVATE
    gauss_simplify_lib
    GTest::gtest_main
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gauss_simplify_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)
gtest_discover_tests(gauss_simplify_tests)
