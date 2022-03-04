cmake_minimum_required(VERSION 3.1)
project(ctlo)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "set build type to release" FORCE)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rosbag
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
)

find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

catkin_package(
   INCLUDE_DIRS include
#  LIBRARIES ctlo
   CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs
   DEPENDS EIGEN3 PCL
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(lidar_odometry src/lidar_odometry_node.cpp)
target_link_libraries(lidar_odometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}  ${OpenMP_CXX_FLAGS})