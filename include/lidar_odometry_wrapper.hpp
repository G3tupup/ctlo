#pragma once

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <thread>

#include "lidar_odometry.hpp"

class LidarOdometryWrapper {
 public:
  LidarOdometryWrapper() {
    base_to_lidar_transform_.frame_id_ = "base_link";
    base_to_lidar_transform_.child_frame_id_ = "lidar";
    base_to_lidar_transform_.setRotation(tf::Quaternion(0, 0, 0, 1));
    base_to_lidar_transform_.setOrigin(tf::Vector3(0, 0, 0));
    lidar_path_.header.frame_id = "base_link";
  }
  LidarOdometryWrapper(const LidarOdometryWrapper& rhs) = delete;
  LidarOdometryWrapper& operator=(const LidarOdometryWrapper& rhs) = delete;
  ~LidarOdometryWrapper() {
    if (process_thread_.joinable()) {
      process_thread_.join();
    }
  }

 private:
  std::unique_ptr<ctlo::LidarOdometry> lidar_odometry_;
  std::vector<std::string> topics_;
  std::string rosbag_folder_name_;
  std::string rosbag_file_name_;
  int accumulate_count_;
  ros::Publisher current_point_cloud_publisher_;
  ros::Publisher ground_points_publisher_;
  ros::Publisher segment_points_publisher_;
  ros::Publisher outlier_points_publisher_;
  ros::Publisher edge_features_publisher_;
  ros::Publisher plane_features_publisher_;

  ros::Publisher odometry_publisher_;
  ros::Publisher lidar_path_publisher_;

  tf::StampedTransform base_to_lidar_transform_;
  tf::TransformBroadcaster transform_broadcaster_;
  nav_msgs::Path lidar_path_;

  std::thread process_thread_;

 public:
  bool setup(ros::NodeHandle& node, ros::NodeHandle& private_node) {
    topics_.resize(1);
    if (!getParam("lidar_topic", topics_[0], private_node)) {
      return false;
    }
    if (!getParam("rosbag_folder_name", rosbag_folder_name_, private_node) ||
        !getParam("rosbag_file_name", rosbag_file_name_, private_node)) {
      return false;
    }
    if (!getParam("accumulate_count", accumulate_count_, private_node)) {
      return false;
    }
    current_point_cloud_publisher_ =
        node.advertise<sensor_msgs::PointCloud2>("/current_point_cloud", 1);
    ground_points_publisher_ =
        node.advertise<sensor_msgs::PointCloud2>("/ground_points", 1);
    segment_points_publisher_ =
        node.advertise<sensor_msgs::PointCloud2>("/segment_points", 1);
    outlier_points_publisher_ =
        node.advertise<sensor_msgs::PointCloud2>("/outlier_points", 1);
    edge_features_publisher_ =
        node.advertise<sensor_msgs::PointCloud2>("/edge_features", 1);
    plane_features_publisher_ =
        node.advertise<sensor_msgs::PointCloud2>("/plane_features", 1);
    odometry_publisher_ = node.advertise<nav_msgs::Odometry>("/odometry", 1);
    lidar_path_publisher_ = node.advertise<nav_msgs::Path>("/lidar_path", 1);
    lidar_odometry_ = ctlo::common::make_unique<ctlo::LidarOdometry>();
    return true;
  }

  void run() {
    process_thread_ = std::thread(&LidarOdometryWrapper::process, this);
    ros::spin();
  }

 private:
  void process() {
    pcl::PointCloud<pcl::PointXYZI> accumulated_point_cloud;
    pcl::PointCloud<pcl::PointXYZI> original_point_cloud;
    int count = 0;
    rosbag::Bag bag;
    std::string bag_name = rosbag_folder_name_ + rosbag_file_name_ + ".bag";
    try {
      bag.open(bag_name, rosbag::bagmode::Read);
    } catch (...) {
      std::cout << bag_name << "cannot be opened" << std::endl;
      return;
    }
    rosbag::View view(bag, rosbag::TopicQuery(topics_));
    for (auto iter = view.begin(); iter != view.end(); ++iter) {
      sensor_msgs::PointCloud2::ConstPtr msg =
          iter->instantiate<sensor_msgs::PointCloud2>();
      if (msg != nullptr) {
        pcl::fromROSMsg(*msg, original_point_cloud);
        accumulated_point_cloud += original_point_cloud;
        if (++count == accumulate_count_) {
          lidar_odometry_->process(accumulated_point_cloud);
          const auto stamp = msg->header.stamp;
          publishGroundPoints(stamp);
          publishSegmentPoints(stamp);
          publishOutlierPoints(stamp);
          publishEdgeFeatures(stamp);
          publishPlaneFeatures(stamp);
          publishPoseAndPath(stamp);
          sendTransform(stamp);
          accumulated_point_cloud.clear();
          count = 0;
        }
      }
    }
    if (ros::ok()) {
      std::cout << "finish processing data" << std::endl;
      ctlo::time_counter::output("total");
    }
  }

  void publishCurrentPointCloud(sensor_msgs::PointCloud2 point_cloud_msg) {
    point_cloud_msg.header.frame_id = "lidar";
    current_point_cloud_publisher_.publish(point_cloud_msg);
  }

  void publishGroundPoints(const ros::Time& stamp) {
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(lidar_odometry_->groundPoints(), point_cloud_msg);
    point_cloud_msg.header.stamp = stamp;
    point_cloud_msg.header.frame_id = "lidar";
    ground_points_publisher_.publish(point_cloud_msg);
  }

  void publishSegmentPoints(const ros::Time& stamp) {
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(lidar_odometry_->segmentPoints(), point_cloud_msg);
    point_cloud_msg.header.stamp = stamp;
    point_cloud_msg.header.frame_id = "lidar";
    segment_points_publisher_.publish(point_cloud_msg);
  }

  void publishOutlierPoints(const ros::Time& stamp) {
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(lidar_odometry_->outlierPoints(), point_cloud_msg);
    point_cloud_msg.header.stamp = stamp;
    point_cloud_msg.header.frame_id = "lidar";
    outlier_points_publisher_.publish(point_cloud_msg);
  }

  void publishEdgeFeatures(const ros::Time& stamp) {
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(lidar_odometry_->edgeFeatures(), point_cloud_msg);
    point_cloud_msg.header.stamp = stamp;
    point_cloud_msg.header.frame_id = "lidar";
    edge_features_publisher_.publish(point_cloud_msg);
  }

  void publishPlaneFeatures(const ros::Time& stamp) {
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(lidar_odometry_->planeFeatures(), point_cloud_msg);
    point_cloud_msg.header.stamp = stamp;
    point_cloud_msg.header.frame_id = "lidar";
    plane_features_publisher_.publish(point_cloud_msg);
  }

  void publishPoseAndPath(const ros::Time& stamp) {
    const auto& pose = lidar_odometry_->currentPose();
    nav_msgs::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = "base_link";
    odometry.pose.pose.orientation.x = pose.rotation().coeffs().x();
    odometry.pose.pose.orientation.y = pose.rotation().coeffs().y();
    odometry.pose.pose.orientation.z = pose.rotation().coeffs().z();
    odometry.pose.pose.orientation.w = pose.rotation().coeffs().w();
    odometry.pose.pose.position.x = pose.translation().x();
    odometry.pose.pose.position.y = pose.translation().y();
    odometry.pose.pose.position.z = pose.translation().z();
    odometry_publisher_.publish(odometry);
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = stamp;
    pose_stamped.header.frame_id = "base_link";
    pose_stamped.pose = odometry.pose.pose;
    lidar_path_.poses.push_back(pose_stamped);
    lidar_path_.header.stamp = stamp;
    lidar_path_publisher_.publish(lidar_path_);
  }

  void sendTransform(const ros::Time& stamp) {
    const auto& pose = lidar_odometry_->currentPose();
    base_to_lidar_transform_.stamp_ = stamp;
    base_to_lidar_transform_.setRotation(tf::Quaternion(
        pose.rotation().coeffs().x(), pose.rotation().coeffs().y(),
        pose.rotation().coeffs().z(), pose.rotation().coeffs().w()));
    base_to_lidar_transform_.setOrigin(tf::Vector3(pose.translation().x(),
                                                   pose.translation().y(),
                                                   pose.translation().z()));
    transform_broadcaster_.sendTransform(base_to_lidar_transform_);
  }

  template <typename T>
  bool getParam(const std::string& param_string, T& param,
                ros::NodeHandle& private_node) {
    if (private_node.getParam(param_string, param)) return true;
    ROS_WARN("%s param setting failed", param_string.c_str());
    return false;
  }
};