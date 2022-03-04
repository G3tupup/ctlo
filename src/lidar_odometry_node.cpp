#include "../include/lidar_odometry_wrapper.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_odometry");
  ros::NodeHandle node, private_node("~");
  LidarOdometryWrapper lidar_odometry_wrapper;
  if (lidar_odometry_wrapper.setup(node, private_node)) {
    lidar_odometry_wrapper.run();
  }
  return 0;
}