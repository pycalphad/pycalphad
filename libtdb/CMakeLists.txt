cmake_minimum_required (VERSION 2.8)
project(LIBTDB)

set(BOOST_ROOT "C:/program files/boost")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
include(PCHSupport)
find_package(Boost 1.51 REQUIRED)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()

# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE libtdb_SRCS ./ *.cpp *.hpp)
FIX_PRECOMPILED_HEADER(libtdb_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/libtdb_pch.cpp)
add_library(libtdb ${libtdb_SRCS})
add_precompiled_header(libtdb include/libtdb_pch.hpp ${CMAKE_CURRENT_SOURCE_DIR}/source/libtdb_pch.cpp)