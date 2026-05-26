cmake_minimum_required(VERSION 2.8.3)
project(traj_bridge)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  trajectory_msgs
  ego_planner
  bspline_opt
)

find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Build dependencies
add_definitions(${EIGEN_DEFINITIONS})

add_executable(traj_bridge_node src/traj_bridge_node.cpp)
add_dependencies(traj_bridge_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(traj_bridge_node
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/ros_eval_playback.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
