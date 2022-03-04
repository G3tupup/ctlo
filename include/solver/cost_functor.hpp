#pragma once

#include <ceres/ceres.h>

#include "../transform/rigid3.hpp"

namespace ctlo {
namespace solver {
class EdgesCostFunctor {
 public:
  explicit EdgesCostFunctor(const std::vector<Eigen::Vector3f>& source_edges,
                            const std::vector<Eigen::Vector3f>& target_edges,
                            const std::vector<Eigen::Vector3f>& edge_principles,
                            const std::vector<float>& edge_distortions,
                            const std::vector<float>& weights)
      : source_edges_(source_edges),
        target_edges_(target_edges),
        edge_principles_(edge_principles),
        edge_distortions_(edge_distortions),
        weights_(weights) {}

  template <typename T>
  bool operator()(const T* const beg_tra_param, const T* const beg_rot_param,
                  const T* const end_tra_param, const T* const end_rot_param,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> beg_tra(beg_tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> beg_rot(beg_rot_param);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> end_tra(end_tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> end_rot(end_rot_param);
    for (size_t i = 0; i < source_edges_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> transformed_point =
          transform::interpolate(transform::Rigid3<T>(beg_tra, beg_rot),
                                 transform::Rigid3<T>(end_tra, end_rot),
                                 T(edge_distortions_[i])) *
          source_edges_[i].cast<T>();
      residuals[i] = ((target_edges_[i].cast<T>() - transformed_point)
                          .cross(edge_principles_[i].cast<T>()))
                         .norm() *
                     T(weights_[i]);
    }
    return true;
  }

  static ceres::CostFunction* create(
      const std::vector<Eigen::Vector3f>& source_edges,
      const std::vector<Eigen::Vector3f>& target_edges,
      const std::vector<Eigen::Vector3f>& edge_principles,
      const std::vector<float>& edge_distortions,
      const std::vector<float>& weights) {
    return (new ceres::AutoDiffCostFunction<EdgesCostFunctor, ceres::DYNAMIC, 3,
                                            4, 3, 4>(
        new EdgesCostFunctor(source_edges, target_edges, edge_principles,
                             edge_distortions, weights),
        source_edges.size()));
  }

 private:
  const std::vector<Eigen::Vector3f>& source_edges_;
  const std::vector<Eigen::Vector3f>& target_edges_;
  const std::vector<Eigen::Vector3f>& edge_principles_;
  const std::vector<float>& edge_distortions_;
  const std::vector<float>& weights_;
};

class PlanesCostFunctor {
 public:
  explicit PlanesCostFunctor(const std::vector<Eigen::Vector3f>& source_planes,
                             const std::vector<Eigen::Vector3f>& target_planes,
                             const std::vector<Eigen::Vector3f>& plane_normals,
                             const std::vector<float>& plane_distortions,
                             const std::vector<float>& weights)
      : source_planes_(source_planes),
        target_planes_(target_planes),
        plane_normals_(plane_normals),
        plane_distortions_(plane_distortions),
        weights_(weights) {}

  template <typename T>
  bool operator()(const T* const beg_tra_param, const T* const beg_rot_param,
                  const T* const end_tra_param, const T* const end_rot_param,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> beg_tra(beg_tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> beg_rot(beg_rot_param);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> end_tra(end_tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> end_rot(end_rot_param);
    for (size_t i = 0; i < source_planes_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> transformed_point =
          transform::interpolate(transform::Rigid3<T>(beg_tra, beg_rot),
                                 transform::Rigid3<T>(end_tra, end_rot),
                                 T(plane_distortions_[i])) *
          source_planes_[i].cast<T>();
      residuals[i] = (target_planes_[i].cast<T>() - transformed_point)
                         .dot(plane_normals_[i].cast<T>()) *
                     T(weights_[i]);
    }
    return true;
  }

  static ceres::CostFunction* create(
      const std::vector<Eigen::Vector3f>& source_planes,
      const std::vector<Eigen::Vector3f>& target_planes,
      const std::vector<Eigen::Vector3f>& plane_normals,
      const std::vector<float>& plane_distortions,
      const std::vector<float>& weights) {
    return (new ceres::AutoDiffCostFunction<PlanesCostFunctor, ceres::DYNAMIC,
                                            3, 4, 3, 4>(
        new PlanesCostFunctor(source_planes, target_planes, plane_normals,
                              plane_distortions, weights),
        source_planes.size()));
  }

 private:
  const std::vector<Eigen::Vector3f>& source_planes_;
  const std::vector<Eigen::Vector3f>& target_planes_;
  const std::vector<Eigen::Vector3f>& plane_normals_;
  const std::vector<float>& plane_distortions_;
  const std::vector<float>& weights_;
};

class FixedPoseCostFunctor {
 public:
  FixedPoseCostFunctor(const transform::Rigid3d& fixed_pose,
                       const double weight)
      : fixed_pose_(fixed_pose), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const tra_param, const T* const rot_param,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> tra(tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> rot(rot_param);
    const transform::Rigid3<T> delta_pose =
        transform::delta(transform::Rigid3<T>(tra, rot), fixed_pose_.cast<T>());
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_map(residuals);
    residuals_map.template block<3, 1>(0, 0) =
        delta_pose.translation() * T(weight_);
    residuals_map.template block<3, 1>(3, 0) =
        delta_pose.rotation().vec() * T(2.0 * weight_);
    return true;
  }

  static ceres::CostFunction* create(const transform::Rigid3d& fixed_pose,
                                     const double weight) {
    return (new ceres::AutoDiffCostFunction<FixedPoseCostFunctor, 6, 3, 4>(
        new FixedPoseCostFunctor(fixed_pose, weight)));
  }

 private:
  const transform::Rigid3d fixed_pose_;
  const double weight_;
};

class DeltaPoseCostFunctor {
 public:
  DeltaPoseCostFunctor(const transform::Rigid3d& delta_pose,
                       const double weight)
      : delta_pose_(delta_pose), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const beg_tra_param, const T* const beg_rot_param,
                  const T* const end_tra_param, const T* const end_rot_param,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> beg_tra(beg_tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> beg_rot(beg_rot_param);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> end_tra(end_tra_param);
    Eigen::Map<const Eigen::Quaternion<T>> end_rot(end_rot_param);
    const transform::Rigid3<T> delta_pose = transform::delta(
        transform::Rigid3<T>(end_tra, end_rot),
        transform::Rigid3<T>(beg_tra, beg_rot) * delta_pose_.cast<T>());
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_map(residuals);
    residuals_map.template block<3, 1>(0, 0) =
        delta_pose.translation() * T(weight_);
    residuals_map.template block<3, 1>(3, 0) =
        delta_pose.rotation().vec() * T(2.0 * weight_);
    return true;
  }

  static ceres::CostFunction* create(const transform::Rigid3d& delta_pose,
                                     const double weight) {
    return (
        new ceres::AutoDiffCostFunction<DeltaPoseCostFunctor, 6, 3, 4, 3, 4>(
            new DeltaPoseCostFunctor(delta_pose, weight)));
  }

 private:
  const transform::Rigid3d delta_pose_;
  const double weight_;
};
}  // namespace solver
}  // namespace ctlo