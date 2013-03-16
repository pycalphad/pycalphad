cmake_minimum_required (VERSION 2.8)
project(UNITTEST)

if (NOT DEFINED TDBREAD_SOURCE_DIR)
  set (TDBREAD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../)
endif (NOT DEFINED TDBREAD_SOURCE_DIR)
set(CMAKE_MODULE_PATH ${TDBREAD_SOURCE_DIR}/cmake)
include(PCHSupport)

find_package(Boost 1.51 REQUIRED unit_test_framework)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()

# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE unittest_HFILES include/*.hpp)
FILE(GLOB_RECURSE unittest_SRCS source/*.cpp)
FIX_PRECOMPILED_HEADER(unittest_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/test_pch.cpp)
add_executable(unittest ${unittest_SRCS} ${unittest_HFILES})
add_precompiled_header(unittest include/test_pch.hpp ${CMAKE_CURRENT_SOURCE_DIR}/source/test_pch.cpp)
# I am not pleased with this hack
target_link_libraries(unittest ${CMAKE_CURRENT_BINARY_DIR}/unittest_pch/test_pch.pch.obj)
target_link_libraries(unittest ${Boost_LIBRARIES})
target_link_libraries(unittest libtdb)