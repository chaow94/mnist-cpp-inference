PROJECT(mnist)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0) 

find_package( OpenCV REQUIRED )


ADD_SUBDIRECTORY( cnpy )

set(CMAKE_CXX_STANDARD 11)

ADD_EXECUTABLE(mnist mnist.cpp utils.cpp utils.hpp)
TARGET_LINK_LIBRARIES( mnist cnpy ${DIR_SRCS} ${OpenCV_LIBS})

