cmake_minimum_required (VERSION 2.8)
project(UNITTEST)

set(BOOST_ROOT "C:/program files/boost")
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
include(PCHSupport)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.51 REQUIRED unit_test_framework)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()

# Handle ipopt
set(IPOPT_HEADERS_DIR ${TDBREAD_SOURCE_DIR}/external/coin)
set(IPOPT_LIB_DIR ${TDBREAD_SOURCE_DIR}/lib)


# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE unittest_SRCS ./ *.cpp *.hpp)
FIX_PRECOMPILED_HEADER(unittest_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/test_pch.cpp)
add_executable(unittest ${unittest_SRCS})
target_link_libraries(unittest ${Boost_LIBRARIES})
target_link_libraries(unittest libtdb)
#add_precompiled_header(unittest include/test_pch.hpp ${CMAKE_CURRENT_SOURCE_DIR}/source/test_pch.cpp)