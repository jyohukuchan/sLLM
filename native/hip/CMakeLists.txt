cmake_minimum_required(VERSION 3.21)

project(ullm_hip_stub VERSION 0.1.0 LANGUAGES C CXX)

add_library(ullm_hip_stub STATIC
    src/hip_stub.cpp
    src/header_c_compile.c
    src/header_cpp_compile.cpp
)

target_compile_features(ullm_hip_stub PRIVATE c_std_11 cxx_std_17)
set_target_properties(ullm_hip_stub PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ullm_hip_stub
    PUBLIC
        "${PROJECT_SOURCE_DIR}/../../include"
)

if(MSVC)
    target_compile_options(ullm_hip_stub PRIVATE /W4 /WX)
else()
    target_compile_options(ullm_hip_stub PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Werror
    )
endif()
