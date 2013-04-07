cmake_minimum_required (VERSION 2.8)
project(LIBGIBBS)

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

# Handle ipopt
set(IPOPT_HEADERS_DIR ${TDBREAD_SOURCE_DIR}/external/coin)
set(IPOPT_LIB_DIR ${TDBREAD_SOURCE_DIR}/lib)


# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE libgibbs_SRCS ./ *.cpp *.hpp)
FIX_PRECOMPILED_HEADER(libgibbs_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/libgibbs_pch.cpp)
add_library(libgibbs STATIC ${libgibbs_SRCS})
target_link_libraries(libgibbs libtdb)
target_link_libraries(libgibbs ${IPOPT_LIB_DIR}/IpOpt-vc10.lib)
add_precompiled_header(libgibbs include/libgibbs_pch.hpp ${CMAKE_CURRENT_SOURCE_DIR}/source/libgibbs_pch.cpp)