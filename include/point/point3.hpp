#pragma once

#include <pcl/point_types.h>

#include "../common/math.hpp"

#define ADD_POINT3D                                              \
  union EIGEN_ALIGN16 {                                          \
    float data[4];                                               \
    struct {                                                     \
      float x;                                                   \
      float y;                                                   \
      float z;                                                   \
    };                                                           \
  };                                                             \
  inline Eigen::Map<Eigen::Vector3f> point() {                   \
    return (Eigen::Vector3f::Map(data));                         \
  }                                                              \
  inline const Eigen::Map<const Eigen::Vector3f> point() const { \
    return (Eigen::Vector3f::Map(data));                         \
  }

namespace ctlo {
namespace point {
template <typename T>
using Point3 = Eigen::Matrix<T, 3, 1>;
using Point3f = Point3<float>;
using Point3d = Point3<double>;

template <typename T>
inline bool ifValid(const T& point) {
  return (std::isfinite(point.x) && std::isfinite(point.y) &&
          std::isfinite(point.z));
}

template <typename T>
inline Point3f getCartesian(const T& point) {
  if (!ifValid(point)) return Point3f::Zero();
  return point.getVector3fMap();
}

template <typename T>
inline Point3<T> cartesianToSphere(const Point3<T>& cartesian) {
  const T range = cartesian.norm();
  if (math::ifZero(range)) {
    return Point3<T>::Zero();
  }
  return {range, std::atan2(cartesian.y(), cartesian.x()),
          std::asin(cartesian.z() / range)};
}

struct EIGEN_ALIGN16 RPoint {
  using FloatType = float;
  ADD_POINT3D;
  FloatType curvature;
  FloatType distortion;
  FloatType intensity;
  size_t row, col;

  const FloatType operator[](const size_t dim) const { return data[dim]; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CurvatureLess {
  bool operator()(const RPoint& lhs, const RPoint& rhs) {
    return lhs.curvature < rhs.curvature;
  }
};

struct CurvatureGreater {
  bool operator()(const RPoint& lhs, const RPoint& rhs) {
    return lhs.curvature > rhs.curvature;
  }
};
}  // namespace point
}  // namespace ctlo

POINT_CLOUD_REGISTER_POINT_STRUCT(ctlo::point::RPoint,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, distortion,
                                      distortion)(float, intensity, intensity))