#pragma once

#include "math.hpp"

namespace ctlo {
namespace common {
template <typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

inline std::pair<Eigen::Vector3f, float> getNormal(const Eigen::Matrix3f& cov) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(cov);
  float lambda0 = 0.f, lambda1 = 0.f,
        lambda2 = std::sqrt(eigen_solver.eigenvalues()[2]);
  if (eigen_solver.eigenvalues()[0] > 0) {
    lambda0 = std::sqrt(eigen_solver.eigenvalues()[0]);
  }
  if (eigen_solver.eigenvalues()[1] > 0) {
    lambda1 = std::sqrt(eigen_solver.eigenvalues()[1]);
  }
  return std::make_pair(eigen_solver.eigenvectors().col(0),
                        std::abs(lambda1 - lambda0) / lambda2);
}

inline std::pair<Eigen::Vector3f, float> getPrinciple(
    const Eigen::Matrix3f& cov) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(cov);
  float lambda1 = 0.f, lambda2 = std::sqrt(eigen_solver.eigenvalues()[2]);
  if (eigen_solver.eigenvalues()[1] > 0) {
    lambda1 = std::sqrt(eigen_solver.eigenvalues()[1]);
  }
  return std::make_pair(eigen_solver.eigenvectors().col(2),
                        std::abs(lambda2 - lambda1) / lambda2);
}

struct Array3iHash {
  size_t operator()(const Eigen::Array3i& key) const {
    return ((1 << 20) - 1) &
           (key.x() * 73856093 ^ key.y() * 19349663 ^ key.z() * 83492791);
  }
};

struct Array3iEqual {
  bool operator()(const Eigen::Array3i& key1,
                  const Eigen::Array3i& key2) const {
    return key1.x() == key2.x() && key1.y() == key2.y() && key1.z() == key2.z();
  }
};
}  // namespace common
}  // namespace ctlo