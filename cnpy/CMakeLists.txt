CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(CNPY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
 

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

ADD_LIBRARY ( cnpy cnpy.cpp cnpy.h)
target_link_libraries(cnpy ${ZLIB_LIBRARIES}) 

 
