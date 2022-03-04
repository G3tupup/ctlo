#pragma once

#include "common/common.hpp"
#include "common/time_counter.hpp"
#include "map/active_feature_map.hpp"
#include "point/feature_processor.hpp"
#include "point/ndt3.hpp"

namespace ctlo {
class LidarOdometry {
 public:
  LidarOdometry() = default;
  LidarOdometry(const LidarOdometry& rhs) = delete;
  LidarOdometry& operator=(const LidarOdometry& rhs) = delete;

 private:
  point::FeatureProcessor feature_processor_;
  transform::Rigid3d current_pose_;
  transform::Rigid3d delta_pose_;
  map::ActiveFeatureMap map_;

 public:
  const transform::Rigid3d& currentPose() const { return current_pose_; }

  const pcl::PointCloud<point::RPoint>& groundPoints() const {
    return feature_processor_.groundPoints();
  }

  const pcl::PointCloud<point::RPoint>& segmentPoints() const {
    return feature_processor_.segmentPoints();
  }

  const pcl::PointCloud<point::RPoint>& outlierPoints() const {
    return feature_processor_.outlierPoints();
  }

  const pcl::PointCloud<point::RPoint>& edgeFeatures() const {
    return feature_processor_.edgeFeatures();
  }

  const pcl::PointCloud<point::RPoint>& planeFeatures() const {
    return feature_processor_.planeFeatures();
  }

  const pcl::PointCloud<point::RPoint>& edgePoints() const {
    return feature_processor_.edgePoints();
  }

  const pcl::PointCloud<point::RPoint>& planePoints() const {
    return feature_processor_.planePoints();
  }

  bool process(const pcl::PointCloud<pcl::PointXYZI>& original_point_cloud) {
    time_counter::tick("total");
    time_counter::tick("feature");
    feature_processor_.process(original_point_cloud);
    time_counter::tack("feature");

    auto begin_pose = current_pose_;
    current_pose_ = begin_pose * delta_pose_;
    time_counter::tick("location");
    map_.locate(edgeFeatures(), planeFeatures(), begin_pose, current_pose_);
    time_counter::tack("location");
    std::cout << "begin: " << begin_pose << std::endl;
    std::cout << "end: " << current_pose_ << std::endl;

    time_counter::tick("update");
    map_.update(edgeFeatures(), planeFeatures(), edgePoints(), planePoints(),
                begin_pose, current_pose_);
    time_counter::tack("update");
    time_counter::tock("total");

    delta_pose_ = transform::delta(begin_pose, current_pose_);
    feature_processor_.undistortPoints(
        transform::delta(current_pose_, begin_pose).cast<float>());
    return true;
  }
};
}  // namespace ctlo