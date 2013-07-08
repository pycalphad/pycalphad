cmake_minimum_required (VERSION 2.8)
project(TDBREAD_EXE)

UNSET(Boost_LIBRARIES)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS timer chrono log log_setup thread filesystem system REQUIRED)
if(Boost_FOUND)
   include_directories(${Boost_INCLUDE_DIRS})
endif()

# Find ALL the source and header files
include_directories(${CMAKE_SOURCE_DIR})
FILE(GLOB_RECURSE tdbread_exe_SRCS ./ *.cpp *.hpp)
add_executable(tdbread ${tdbread_exe_SRCS})

if(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -lgfortran")
endif()

target_link_libraries(tdbread ${Boost_LIBRARIES})
target_link_libraries(tdbread libgibbs)