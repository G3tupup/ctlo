#pragma once

#include <memory>
#include <mutex>
#include <queue>

#include "../common/common.hpp"
#include "../point/feature_processor.hpp"
#include "../point/ndt3.hpp"
#include "../solver/ct_pose_solver.hpp"

#define OPENMPTHREAD 6

namespace ctlo {
namespace map {
class ActiveFeatureMap {
 public:
  ActiveFeatureMap()
      : edge_grid_(edge_voxel_size_, voxel::merge<point::Ndt3f>,
                   voxel::split<point::Ndt3f>),
        plane_grid_(plane_voxel_size_, voxel::merge<point::Ndt3f>,
                    voxel::split<point::Ndt3f>) {}

 private:
  template <typename T>
  using Container = pcl::PointCloud<T>;
  static constexpr float edge_voxel_size_ = 0.25f;
  static constexpr float plane_voxel_size_ = 0.25f;
  static constexpr size_t max_iteration_ = 3;
  static constexpr size_t window_size_ = 500;
  static constexpr float ratio_thresh_ =
      10.f / static_cast<float>(window_size_);
  static constexpr double fix_weight_ = 0.06;
  static constexpr double delta_weight_ = 0.08;
  voxel::VoxelGrid<point::Ndt3f, Container> edge_grid_, plane_grid_;
  std::queue<std::shared_ptr<Container<point::Ndt3f>>> added_edges_,
      added_planes_;

 public:
  bool locate(const Container<point::RPoint>& edges,
              const Container<point::RPoint>& planes,
              transform::Rigid3d& begin_pose,
              transform::Rigid3d& end_pose) const {
    std::vector<Eigen::Vector3f> source_edges, target_edges, edge_principles,
        source_planes, target_planes, plane_normals;
    std::vector<float> edge_distortions, plane_distortions;
    std::vector<float> edge_weights, plane_weights;
    source_edges.reserve(edges.size());
    target_edges.reserve(edges.size());
    edge_principles.reserve(edges.size());
    edge_distortions.reserve(edges.size());
    edge_weights.reserve(edges.size());
    source_planes.reserve(planes.size());
    target_planes.reserve(planes.size());
    plane_normals.reserve(planes.size());
    plane_distortions.reserve(planes.size());
    plane_weights.reserve(planes.size());
    const auto fixed_begin_pose = begin_pose;
    const auto fixed_delta_pose = transform::delta(begin_pose, end_pose);
    for (size_t iter = 0; iter < max_iteration_; ++iter) {
      const auto pose_bf = begin_pose.cast<float>();
      const auto pose_ef = end_pose.cast<float>();
      findCorrelations(edges, edge_grid_, pose_bf, pose_ef,
                       max_iteration_ - iter, common::getPrinciple,
                       source_edges, target_edges, edge_principles,
                       edge_distortions, edge_weights);
      findCorrelations(planes, plane_grid_, pose_bf, pose_ef,
                       max_iteration_ - iter, common::getNormal, source_planes,
                       target_planes, plane_normals, plane_distortions,
                       plane_weights);
      std::cout << "edge_correlation: " << source_edges.size()
                << ", plane_correlation: " << source_planes.size() << std::endl;
      solver::CTPoseSolver solver(begin_pose, end_pose);
      solver.addEdgesResidual(source_edges, target_edges, edge_principles,
                              edge_distortions, edge_weights);
      solver.addPlanesResidual(source_planes, target_planes, plane_normals,
                               plane_distortions, plane_weights);
      solver.addBeginPoseResidual(
          fixed_begin_pose,
          fix_weight_ * std::sqrt(static_cast<double>(source_edges.size() +
                                                      source_planes.size())));
      solver.addDeltaPoseResidual(
          fixed_delta_pose,
          delta_weight_ * std::sqrt(static_cast<double>(source_edges.size() +
                                                        source_planes.size())));
      solver.solve();
    }
    return true;
  }

  void update(const Container<point::RPoint>& edge_features,
              const Container<point::RPoint>& plane_features,
              const Container<point::RPoint>& edges,
              const Container<point::RPoint>& planes,
              const transform::Rigid3d& begin_pose,
              const transform::Rigid3d& end_pose) {
    if (ifKeyFrame(*toNdts(edge_features, begin_pose, end_pose), edge_grid_)) {
      auto edge_ndts = toNdts(edges, begin_pose, end_pose);
      edge_grid_.add(*edge_ndts);
      added_edges_.push(edge_ndts);
      if (added_edges_.size() > window_size_) {
        edge_grid_.remove(*added_edges_.front());
        added_edges_.pop();
      }
    }
    if (ifKeyFrame(*toNdts(plane_features, begin_pose, end_pose),
                   plane_grid_)) {
      auto plane_ndts = toNdts(planes, begin_pose, end_pose);
      plane_grid_.add(*plane_ndts);
      added_planes_.push(plane_ndts);
      if (added_planes_.size() > window_size_) {
        plane_grid_.remove(*added_planes_.front());
        added_planes_.pop();
      }
    }
  }

 private:
  static void findCorrelations(
      const Container<point::RPoint>& points,
      const voxel::VoxelGrid<point::Ndt3f, Container>& map,
      const transform::Rigid3f& begin_pose, const transform::Rigid3f& end_pose,
      const int bound,
      std::function<std::pair<Eigen::Vector3f, float>(const Eigen::Matrix3f&)>
          func,
      std::vector<Eigen::Vector3f>& source,
      std::vector<Eigen::Vector3f>& target,
      std::vector<Eigen::Vector3f>& eigen_vectors,
      std::vector<float>& distortions, std::vector<float>& weights) {
    const float voxel_size_inv = map.voxelSizeInv();
    source.clear();
    target.clear();
    eigen_vectors.clear();
    distortions.clear();
    weights.clear();
#ifdef OPENMPTHREAD
    std::mutex mutex;
#pragma omp parallel for num_threads(OPENMPTHREAD) \
    schedule(guided, OPENMPTHREAD)
#endif
    for (size_t n = 0; n < points.size(); ++n) {
      const auto& point = points[n];
      const Eigen::Vector3f transformed_point =
          transform::interpolate(begin_pose, end_pose, point.distortion) *
          point.point();
      Eigen::Array3i index =
          voxel::getVoxelIndex(transformed_point.x(), transformed_point.y(),
                               transformed_point.z(), voxel_size_inv);
      point::Ndt3f ndt;
      for (int i = -bound; i <= bound; ++i) {
        for (int j = -bound; j <= bound; ++j) {
          for (int k = -bound; k <= bound; ++k) {
            const auto map_index = map.get(index + Eigen::Array3i(i, j, k));
            if (map_index != std::numeric_limits<size_t>::max()) {
              ndt += map[map_index];
            }
          }
        }
      }
      if (ndt.num() < static_cast<size_t>(std::max(3, bound * 2 + 1))) continue;
      auto pair = func(ndt.var());
#ifdef OPENMPTHREAD
      std::unique_lock<std::mutex> lock(mutex);
#endif
      source.emplace_back(point.point());
      target.emplace_back(ndt.mean());
      eigen_vectors.emplace_back(pair.first);
      distortions.emplace_back(point.distortion);
      weights.emplace_back(pair.second);
    }
  }

  static std::shared_ptr<Container<point::Ndt3f>> toNdts(
      const Container<point::RPoint>& points,
      const transform::Rigid3d& begin_pose,
      const transform::Rigid3d& end_pose) {
    std::shared_ptr<Container<point::Ndt3f>> result =
        std::make_shared<Container<point::Ndt3f>>();
    result->reserve(points.size());
    const transform::Rigid3f pose_bf = begin_pose.cast<float>();
    const transform::Rigid3f pose_ef = end_pose.cast<float>();
    for (const auto& point : points) {
      result->push_back(point::Ndt3f(
          transform::interpolate(pose_bf, pose_ef, point.distortion) *
          point.point()));
    }
    return result;
  }

  static bool ifKeyFrame(const Container<point::Ndt3f>& ndts,
                         const voxel::VoxelGrid<point::Ndt3f, Container>& map) {
    size_t miss = 0;
    for (const auto& ndt : ndts) {
      if (map.get(ndt) == std::numeric_limits<size_t>::max()) ++miss;
    }
    return (static_cast<float>(miss) / static_cast<float>(ndts.size()) >=
            ratio_thresh_);
  }
};
}  // namespace map
}  // namespace ctlo