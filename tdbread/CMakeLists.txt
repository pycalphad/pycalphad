cmake_minimum_required (VERSION 2.8)
project(TDBREAD_EXE)

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
FILE(GLOB_RECURSE tdbread_exe_SRCS ./ *.cpp *.hpp)
add_executable(tdbread ${tdbread_exe_SRCS})
target_link_libraries(tdbread libgibbs)