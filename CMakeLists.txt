cmake_minimum_required(VERSION 2.8.3)
project(kl_planning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  sensor_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  DisplayImage.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  DEPENDS
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
