cmake_minimum_required (VERSION 2.8)
project(TDBREAD_EXE)

set(BOOST_ROOT "C:/program files/boost")
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
#include(PCHSupport)
find_package(Boost 1.51 REQUIRED)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()


# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE tdbread_exe_SRCS ./ *.cpp *.hpp)
#FIX_PRECOMPILED_HEADER(tdbread_exe_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/tdbread_pch.cpp)
add_executable(tdbread ${tdbread_exe_SRCS})
target_link_libraries(tdbread libgibbs)
#add_native_precompiled_header(tdbread include/tdbread_pch.hpp ${CMAKE_CURRENT_SOURCE_DIR}/source/tdbread_pch.cpp)