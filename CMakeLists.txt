cmake_minimum_required(VERSION 3.0.2)
project(kl_planning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  DisplayImage.srv
  VisualizeTrajectorySamples.srv
  SetPose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  DEPENDS
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
