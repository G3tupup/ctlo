#pragma once

#include <pcl/point_cloud.h>

#include <stack>
#include <unordered_set>

#include "../common/priority_queue.hpp"
#include "../transform/rigid3.hpp"
#include "../voxel/voxel_grid.hpp"
#include "point3.hpp"
#include "point_image.hpp"

namespace ctlo {
namespace point {
class FeatureProcessor {
 public:
  FeatureProcessor() {
    state_matrix_.resize(image_height_, image_width_);
    range_matrix_.resize(image_height_, image_width_);
    ground_points_.reserve(image_height_ * image_width_);
    segment_points_.reserve(image_height_ * image_width_);
    outlier_points_.reserve(image_height_ * image_width_);
    edge_features_.reserve(edge_feature_num_per_sub_line_ * image_height_ *
                           sub_image_num_);
    plane_features_.reserve(edge_feature_num_per_sub_line_ * image_height_ *
                            sub_image_num_);
    edge_points_.reserve(edge_num_per_sub_line_ * image_height_ *
                         sub_image_num_);
    for (size_t row = 0; row < image_height_; ++row) {
      for (size_t col = 0; col < image_width_; ++col) {
        RPoint& point = point_image_(row, col);
        point.row = row;
        point.col = col;
      }
    }
  }

 private:
  static constexpr size_t lidar_line_num_ = 16;
  static constexpr size_t point_num_per_line_ = 1800;
  static constexpr float elevation_range_ = math::pi<float>() * 30.f / 180.f,
                         azimuth_range_ = math::twoPi<float>();
  static constexpr float min_range_ = 0.1f, max_range_ = 120.f;
  static constexpr float lidar_mount_elevation_ = 0.f;

  static constexpr float elevation_resolution_ =
      elevation_range_ / static_cast<float>(lidar_line_num_ - 1);
  static constexpr float azimuth_resolution_ =
      azimuth_range_ / static_cast<float>(point_num_per_line_);
  static constexpr float min_elevation_ = -math::pi<float>() * 15.f / 180.f;
  static constexpr float max_elevation_ =
      min_elevation_ + elevation_resolution_ * (lidar_line_num_ - 1);
  static constexpr float min_azimuth_ = -math::pi<float>();
  static constexpr float max_azimuth_ =
      min_azimuth_ + azimuth_resolution_ * (point_num_per_line_ - 1);

  static constexpr size_t image_height_sample_step_ = 1,
                          image_width_sample_step_ = 1;
  static constexpr size_t image_height_ =
      lidar_line_num_ / image_height_sample_step_;
  static constexpr size_t image_width_ =
      point_num_per_line_ / image_width_sample_step_;
  static constexpr int neighbor_half_width_ =
      image_width_ / 400 > 3 ? image_width_ / 400 : 3;
  static constexpr int neighbor_width_ = 2 * neighbor_half_width_ + 1;
  static constexpr size_t max_ground_image_row_ =
      static_cast<size_t>((-min_elevation_ - lidar_mount_elevation_) /
                          elevation_resolution_) /
      image_height_sample_step_;

  static constexpr size_t sub_image_num_ = 8;
  static constexpr size_t sub_image_width_ = image_width_ / sub_image_num_;
  static constexpr float max_ground_angle_ = 0.2f;
  static constexpr float max_break_angle_ = 0.15f;
  static constexpr size_t outlier_max_num_ = 20;
  static constexpr float max_plane_curvature_ =
      0.06f / static_cast<float>(neighbor_half_width_);
  static constexpr float min_edge_curvature_ = max_plane_curvature_ * 3.f;
  static constexpr float invalid_curvature_ =
      (max_plane_curvature_ + min_edge_curvature_) * 0.5f;
  static constexpr size_t edge_feature_num_per_sub_line_ = 3;
  static constexpr size_t edge_num_per_sub_line_ = 10;
  static constexpr float plane_points_voxel_size_ = 0.2f;
  static constexpr float feature_num_ratio_ = 3.f;
  static constexpr bool if_lidar_clockwise_ = true;
  static constexpr bool if_column_order_ = true;

 private:
  // invalid: 0, valid: 1, ground: 2, segment: 3, outlier: 4
  Eigen::MatrixXi state_matrix_;
  Eigen::MatrixXf range_matrix_;
  PointImage<RPoint, image_height_, image_width_> point_image_;
  const std::array<Eigen::Array2i, 4> search_directions_ = {
      {Eigen::Array2i(-1, 0), Eigen::Array2i(0, -1), Eigen::Array2i(0, 1),
       Eigen::Array2i(1, 0)}};
  const std::array<Eigen::Vector2f, 2> break_factor_ = {
      {Eigen::Vector2f(
           std::cos(elevation_resolution_ * image_height_sample_step_),
           std::sin(elevation_resolution_* image_height_sample_step_) /
               std::sin(max_break_angle_)),
       Eigen::Vector2f(std::cos(azimuth_resolution_* image_width_sample_step_),
                       std::sin(azimuth_resolution_* image_width_sample_step_) /
                           std::sin(max_break_angle_))}};

  template <typename T>
  using Container = pcl::PointCloud<T>;
  // using Container = std::vector<T, Eigen::aligned_allocator<T>>;
  Container<RPoint> ground_points_, segment_points_, outlier_points_,
      edge_features_, plane_features_, edge_points_;
  voxel::VoxelGrid<RPoint, Container> plane_points_{plane_points_voxel_size_};

 public:
  bool process(const Container<pcl::PointXYZI>& points) {
    reset();
    auto order_value = getOrderValue(points);
    rasterizePoints(points, order_value);
    getDistortions(order_value);
    markGroundPoints();
    markNongroundPoints();
    getPointsCurvature();
    getAllPoints();
    std::cout << "ground: " << ground_points_.size()
              << ", segment: " << segment_points_.size()
              << ", outlier: " << outlier_points_.size()
              << ", edge_feature: " << edge_features_.size()
              << ", plane_feature: " << plane_features_.size()
              << ", edge: " << edge_points_.size()
              << ", plane: " << plane_points_.size() << std::endl;
    return true;
  }

  const Container<RPoint>& groundPoints() const { return ground_points_; }

  const Container<RPoint>& segmentPoints() const { return segment_points_; }

  const Container<RPoint>& outlierPoints() const { return outlier_points_; }

  const Container<RPoint>& edgeFeatures() const { return edge_features_; }

  const Container<RPoint>& planeFeatures() const { return plane_features_; }

  const Container<RPoint>& edgePoints() const { return edge_points_; }

  const Container<point::RPoint>& planePoints() const {
    return plane_points_.points();
  }

  void undistortPoints(const transform::Rigid3f& delta_inv) {
    undistortPoints(delta_inv, ground_points_);
    undistortPoints(delta_inv, segment_points_);
    undistortPoints(delta_inv, outlier_points_);
    undistortPoints(delta_inv, edge_features_);
    undistortPoints(delta_inv, plane_features_);
  }

 private:
  static void undistortPoints(const transform::Rigid3f& delta_inv,
                              Container<RPoint>& points) {
    for (auto& point : points) {
      point.point() =
          transform::interpolate(delta_inv, transform::Rigid3f::Identity(),
                                 point.distortion) *
          point.point();
    }
  }

  void reset() {
    ground_points_.clear();
    segment_points_.clear();
    outlier_points_.clear();
    edge_features_.clear();
    plane_features_.clear();
    edge_points_.clear();
    plane_points_.clear();
    state_matrix_.setZero(image_height_, image_width_);
    range_matrix_.setZero(image_height_, image_width_);
    point_image_.clearMarks();
  }

  static Eigen::Array2f getOrderValue(const Container<pcl::PointXYZI>& points) {
    Eigen::Array2f order_value;
    auto iter = points.begin();
    for (; iter != points.end(); ++iter) {
      if (!ifValid(*iter)) continue;
      order_value.x() = cartesianToSphere(getCartesian(*iter)).y();
      break;
    }
    iter = --points.end();
    for (; iter != points.begin(); --iter) {
      if (!ifValid(*iter)) continue;
      order_value.y() = cartesianToSphere(getCartesian(*iter)).y();
      break;
    }
    if (if_lidar_clockwise_) {
      order_value = -order_value;
    }
    while (order_value.y() - order_value.x() > 3.f * math::pi<float>()) {
      order_value.y() -= math::twoPi<float>();
    }
    while (order_value.y() - order_value.x() < math::pi<float>()) {
      order_value.y() += math::twoPi<float>();
    }
    return order_value;
  }

  void rasterizePoints(const Container<pcl::PointXYZI>& points,
                       Eigen::Array2f& order_value) {
    bool is_half_passed = false;
    for (const auto& point : points) {
      if (!ifValid(point)) continue;
      const auto cartesian = getCartesian(point);
      const auto sphere = cartesianToSphere(cartesian);
      if (sphere.x() >= min_range_ && sphere.x() <= max_range_) {
        size_t row(
            std::round((sphere.z() - min_elevation_) / elevation_resolution_));
        size_t col(
            std::round((sphere.y() - min_azimuth_) / azimuth_resolution_));
        if (col == point_num_per_line_) col = 0;
        if (row % image_height_sample_step_ == 0 &&
            col % image_width_sample_step_ == 0) {
          const size_t image_row = row / image_height_sample_step_,
                       image_col = col / image_width_sample_step_;
          if (image_row >= 0 && image_row < image_height_ && image_col >= 0 &&
              image_col < image_width_) {
            float point_order = if_lidar_clockwise_ ? -sphere.y() : sphere.y();
            if (!is_half_passed) {
              if (point_order < order_value.x() - math::halfPi<float>())
                point_order += math::twoPi<float>();
              else if (point_order >
                       order_value.x() + math::halfPi<float>() * 3.f)
                point_order -= math::twoPi<float>();
              if (point_order - order_value.x() > math::pi<float>())
                is_half_passed = true;
            } else {
              point_order += math::twoPi<float>();
              if (point_order < order_value.y() - math::halfPi<float>() * 3.f)
                point_order += math::twoPi<float>();
              else if (point_order > order_value.y() + math::halfPi<float>())
                point_order -= math::twoPi<float>();
            }
            if (!if_column_order_) {
              if (point_order < order_value.x()) order_value.x() = point_order;
              if (point_order > order_value.y()) order_value.y() = point_order;
              static size_t last_row = row;
              if (last_row != row) {
                is_half_passed = false;
                last_row = row;
              }
            }
            point_image_(image_row, image_col).point() = cartesian;
            point_image_(image_row, image_col).distortion = point_order;
            point_image_(image_row, image_col).intensity = point.intensity;
            state_matrix_(image_row, image_col) = 1;
            range_matrix_(image_row, image_col) = sphere.x();
          }
        }
      }
    }
  }

  void getDistortions(const Eigen::Array2f& order_value) {
    const float inv = 1.f / (order_value.y() - order_value.x());
    for (size_t row = 0; row < image_height_; ++row) {
      for (size_t col = 0; col < image_width_; ++col) {
        if (state_matrix_(row, col) == 1) {
          auto& point = point_image_(row, col);
          point.distortion = (point.distortion - order_value.x()) * inv;
        }
      }
    }
  }

  void markGroundPoints() {
    for (size_t row = 0; row < max_ground_image_row_; ++row) {
      for (size_t col = 0; col < image_width_; ++col) {
        if (state_matrix_(row, col) && state_matrix_(row + 1, col)) {
          const auto diff = point_image_(row + 1, col).point() -
                            point_image_(row, col).point();
          if (std::abs(
                  math::fastAtan2(diff.z(), std::hypot(diff.x(), diff.y())) +
                  lidar_mount_elevation_) <= max_ground_angle_) {
            state_matrix_(row, col) = 2;
            state_matrix_(row + 1, col) = 2;
          }
        }
      }
    }
  }

  void markNongroundPoints() {
    for (size_t row = 0; row < image_height_; ++row) {
      for (size_t col = 0; col < image_width_; ++col) {
        if (state_matrix_(row, col) == 1) markRegion(row, col);
      }
    }
  }

  void markRegion(const size_t row, const size_t col) {
    state_matrix_(row, col) = 3;
    std::stack<Eigen::Array2i> indices_to_search;
    std::vector<Eigen::Array2i> finished_indices;
    indices_to_search.push(Eigen::Array2i(row, col));
    while (!indices_to_search.empty()) {
      const auto search_index = indices_to_search.top();
      indices_to_search.pop();
      for (const auto& direction : search_directions_) {
        Eigen::Array2i neighbor_index = search_index + direction;
        if (neighbor_index.x() < 0 ||
            neighbor_index.x() >= static_cast<int>(image_height_)) {
          continue;
        }
        if (neighbor_index.y() == -1) {
          neighbor_index.y() = image_width_ - 1;
        }
        if (neighbor_index.y() == image_width_) {
          neighbor_index.y() = 0;
        }
        if (state_matrix_(neighbor_index.x(), neighbor_index.y()) == 1) {
          const auto& factor =
              (direction.x() == 0) ? break_factor_[1] : break_factor_[0];
          const float r1 = range_matrix_(search_index.x(), search_index.y());
          const float r2 =
              range_matrix_(neighbor_index.x(), neighbor_index.y());
          if (std::abs(r1 - r2) < std::min(r1, r2) * factor.y()) {
            state_matrix_(neighbor_index.x(), neighbor_index.y()) = 3;
            indices_to_search.push(neighbor_index);
          } else if (direction.x() == 0) {
            const int sign = r1 < r2 ? 1 : -1,
                      col = r1 < r2 ? neighbor_index.y() : search_index.y();
            if (direction.y() * sign > 0) {
              for (int delta_col = 0; delta_col <= neighbor_half_width_;
                   ++delta_col) {
                point_image_.mark(search_index.x(), col + delta_col);
              }
            } else {
              for (int delta_col = 0; delta_col >= -neighbor_half_width_;
                   --delta_col) {
                point_image_.mark(search_index.x(), col + delta_col);
              }
            }
          }
        }
      }
      finished_indices.push_back(search_index);
    }
    markOutlierPoints(finished_indices);
  }

  void markOutlierPoints(const std::vector<Eigen::Array2i>& indices) {
    if (indices.size() > outlier_max_num_) return;
    for (const auto& index : indices) {
      state_matrix_(index.x(), index.y()) = 4;
    }
  }

  void getPointsCurvature() {
    for (size_t row = 0; row < image_height_; ++row) {
      float range_sum = 0.f;
      size_t valid_num = 0;
      for (size_t col = 0; col < neighbor_width_; ++col) {
        if (state_matrix_(row, col)) {
          range_sum += range_matrix_(row, col);
          ++valid_num;
        }
      }
      if (valid_num > neighbor_half_width_ &&
          state_matrix_(row, neighbor_half_width_)) {
        point_image_(row, size_t(neighbor_half_width_)).curvature =
            std::abs(range_sum / static_cast<float>(valid_num) -
                     range_matrix_(row, neighbor_half_width_));
      } else {
        point_image_(row, size_t(neighbor_half_width_)).curvature =
            invalid_curvature_;
      }
      for (int col = 1; col < static_cast<int>(image_width_); ++col) {
        if (state_matrix_(row, col - 1)) {
          range_sum -= range_matrix_(row, col - 1);
          --valid_num;
        }
        auto new_col = point_image_.validCol(col - 1 + neighbor_width_);
        if (state_matrix_(row, new_col)) {
          range_sum += range_matrix_(row, new_col);
          ++valid_num;
        }
        auto cur_col = point_image_.validCol(col + neighbor_half_width_);
        if (valid_num > neighbor_half_width_ && state_matrix_(row, cur_col)) {
          point_image_(row, cur_col).curvature =
              std::abs(range_sum / static_cast<float>(valid_num) -
                       range_matrix_(row, cur_col));
        } else {
          point_image_(row, cur_col).curvature = invalid_curvature_;
        }
      }
    }
  }

  void getAllPoints() {
    for (size_t sub_i = 0; sub_i < sub_image_num_; ++sub_i) {
      common::PriorityQueue<RPoint, CurvatureGreater> sub_image_plane_points(
          image_height_ * sub_image_width_);
      size_t sub_image_edge_feature_num = 0;
      for (size_t row = 0; row < image_height_; ++row) {
        common::PriorityQueue<RPoint, CurvatureLess> sub_line_edge_points(
            sub_image_width_);
        for (size_t col = sub_i * sub_image_width_;
             col < (sub_i + 1) * sub_image_width_; ++col) {
          const RPoint& point = point_image_(row, col);
          switch (state_matrix_(row, col)) {
            case 2:
              ground_points_.push_back(point);
              if (point.curvature < max_plane_curvature_) {
                sub_image_plane_points.push(point);
                plane_points_.add(point);
              }
              break;
            case 3:
              segment_points_.push_back(point);
              if (point.curvature > min_edge_curvature_) {
                sub_line_edge_points.push(point);
              } else if (point.curvature < max_plane_curvature_) {
                sub_image_plane_points.push(point);
                plane_points_.add(point);
              }
              break;
            case 4:
              outlier_points_.push_back(point);
              break;
            default:
              break;
          }
        }
        sub_image_edge_feature_num +=
            getSubLineEdgeFeaturesAndPoints(sub_line_edge_points);
      }
      getSubImagePlaneFeatures(
          static_cast<size_t>(static_cast<float>(sub_image_edge_feature_num) *
                              feature_num_ratio_),
          sub_image_plane_points);
    }
  }

  const size_t getSubLineEdgeFeaturesAndPoints(
      common::PriorityQueue<RPoint, CurvatureLess>& sub_line_edge_points) {
    size_t edge_num = 0;
    while (edge_num < edge_num_per_sub_line_ && !sub_line_edge_points.empty()) {
      const RPoint point = std::move(sub_line_edge_points.top());
      sub_line_edge_points.pop();
      if (!point_image_.ifMarked(point.row, point.col)) {
        edge_points_.push_back(point);
        if (edge_num < edge_feature_num_per_sub_line_) {
          edge_features_.push_back(point);
          ++edge_num;
        }
        for (int delta_col = -neighbor_half_width_;
             delta_col <= neighbor_half_width_; ++delta_col) {
          point_image_.mark(static_cast<int>(point.row),
                            static_cast<int>(point.col) + delta_col);
        }
      }
    }
    return edge_num;
  }

  void getSubImagePlaneFeatures(
      const size_t num,
      common::PriorityQueue<RPoint, CurvatureGreater>& sub_image_plane_points) {
    size_t plane_num = 0;
    while (plane_num < num && !sub_image_plane_points.empty()) {
      const RPoint point = std::move(sub_image_plane_points.top());
      sub_image_plane_points.pop();
      if (!point_image_.ifMarked(point.row, point.col)) {
        plane_features_.push_back(point);
        for (int delta_col = -neighbor_half_width_;
             delta_col <= neighbor_half_width_; ++delta_col) {
          point_image_.mark(static_cast<int>(point.row),
                            static_cast<int>(point.col) + delta_col);
        }
        ++plane_num;
      }
    }
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace point
}  // namespace ctlo