cmake_minimum_required(VERSION 3.6)

project(mlp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)

add_executable(mlp mlp.cpp)

target_compile_options(mlp PUBLIC -DHAVE_ROOT -DHAVE_LIBZ -DHAVE_TMFE)
install(TARGETS mlp DESTINATION bin)
