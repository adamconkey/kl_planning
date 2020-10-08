cmake_minimum_required(VERSION 2.8.3)
project(kl_planning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
  DEPENDS
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
