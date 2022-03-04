#pragma once

#include "cost_functor.hpp"

namespace ctlo {
namespace solver {
class CTPoseSolver {
 public:
  CTPoseSolver(transform::Rigid3d& begin_pose, transform::Rigid3d& end_pose)
      : beg_tra_param_(begin_pose.translation().data()),
        beg_rot_param_(begin_pose.rotation().coeffs().data()),
        end_tra_param_(end_pose.translation().data()),
        end_rot_param_(end_pose.rotation().coeffs().data()) {
    options_.linear_solver_type = ceres::DENSE_QR;
    options_.max_num_iterations = 6;
    options_.minimizer_progress_to_stdout = false;
    problem_.AddParameterBlock(beg_tra_param_, 3);
    problem_.AddParameterBlock(beg_rot_param_, 4);
    problem_.SetParameterization(beg_rot_param_,
                                 new ceres::EigenQuaternionParameterization);
    problem_.AddParameterBlock(end_tra_param_, 3);
    problem_.AddParameterBlock(end_rot_param_, 4);
    problem_.SetParameterization(end_rot_param_,
                                 new ceres::EigenQuaternionParameterization);
  }

  CTPoseSolver(const CTPoseSolver& rhs) = delete;
  CTPoseSolver& operator=(const CTPoseSolver& rhs) = delete;

 private:
  double *beg_tra_param_, *beg_rot_param_, *end_tra_param_, *end_rot_param_;
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;

 public:
  bool solve() {
    ceres::Solve(options_, &problem_, &summary_);
    return summary_.IsSolutionUsable();
  }

  void addEdgesResidual(const std::vector<Eigen::Vector3f>& source_edges,
                        const std::vector<Eigen::Vector3f>& target_edges,
                        const std::vector<Eigen::Vector3f>& edge_principles,
                        const std::vector<float>& edge_distortions,
                        const std::vector<float>& weights) {
    problem_.AddResidualBlock(
        EdgesCostFunctor::create(source_edges, target_edges, edge_principles,
                                 edge_distortions, weights),
        nullptr, beg_tra_param_, beg_rot_param_, end_tra_param_,
        end_rot_param_);
  }

  void addPlanesResidual(const std::vector<Eigen::Vector3f>& source_planes,
                         const std::vector<Eigen::Vector3f>& target_planes,
                         const std::vector<Eigen::Vector3f>& plane_normals,
                         const std::vector<float>& plane_distortions,
                         const std::vector<float>& weights) {
    problem_.AddResidualBlock(
        PlanesCostFunctor::create(source_planes, target_planes, plane_normals,
                                  plane_distortions, weights),
        nullptr, beg_tra_param_, beg_rot_param_, end_tra_param_,
        end_rot_param_);
  }

  void addBeginPoseResidual(const transform::Rigid3d& begin_pose,
                            const double weight) {
    problem_.AddResidualBlock(FixedPoseCostFunctor::create(begin_pose, weight),
                              nullptr, beg_tra_param_, beg_rot_param_);
  }

  void addDeltaPoseResidual(const transform::Rigid3d& delta_pose,
                            const double weight) {
    problem_.AddResidualBlock(DeltaPoseCostFunctor::create(delta_pose, weight),
                              nullptr, beg_tra_param_, beg_rot_param_,
                              end_tra_param_, end_rot_param_);
  }
};
}  // namespace solver
}  // namespace ctlo