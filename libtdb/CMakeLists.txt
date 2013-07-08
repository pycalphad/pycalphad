cmake_minimum_required (VERSION 2.8)
project(LIBTDB)

if (NOT DEFINED TDBREAD_SOURCE_DIR)
  set (TDBREAD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../)
endif (NOT DEFINED TDBREAD_SOURCE_DIR)
set(CMAKE_MODULE_PATH ${TDBREAD_SOURCE_DIR}/cmake)
include(PCHSupport)

UNSET(Boost_LIBRARIES)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS log log_setup thread filesystem system REQUIRED)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE libtdb_SRCS ./ *.cpp *.hpp)
FIX_PRECOMPILED_HEADER(libtdb_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/libtdb_pch.cpp)
add_library(libtdb STATIC ${libtdb_SRCS})
target_link_libraries(libtdb ${Boost_LIBRARIES})

add_precompiled_header(libtdb "${CMAKE_CURRENT_SOURCE_DIR}/include/libtdb_pch.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/source/libtdb_pch.cpp")