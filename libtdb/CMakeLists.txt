cmake_minimum_required (VERSION 2.8)
project(LIBTDB)

if (NOT DEFINED TDBREAD_SOURCE_DIR)
  set (TDBREAD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../)
endif (NOT DEFINED TDBREAD_SOURCE_DIR)
set(CMAKE_MODULE_PATH ${TDBREAD_SOURCE_DIR}/cmake)
include(PCHSupport)

UNSET(Boost_LIBRARIES)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.51 REQUIRED)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()

# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE libtdb_SRCS ./ *.cpp *.hpp)
FIX_PRECOMPILED_HEADER(libtdb_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/libtdb_pch.cpp)
add_library(libtdb STATIC ${libtdb_SRCS})
add_precompiled_header(libtdb include/libtdb_pch.hpp ${CMAKE_CURRENT_SOURCE_DIR}/source/libtdb_pch.cpp)