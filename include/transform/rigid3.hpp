#pragma once

#include <iostream>
#include <string>

#include "../common/math.hpp"

namespace ctlo {
namespace transform {
template <typename T>
class Rigid3 {
  friend std::ostream& operator<<(std::ostream& os, const Rigid3& rigid3) {
    os << rigid3.debugString();
    return os;
  }

 public:
  using FloatType = T;
  using Vector3 = Eigen::Matrix<FloatType, 3, 1>;
  using Rotation3 = Eigen::Quaternion<FloatType>;
  using AngleAxis = Eigen::AngleAxis<FloatType>;

  Rigid3() : translation_(Vector3::Zero()), rotation_(Rotation3::Identity()) {}
  explicit Rigid3(const Vector3& translation, const Rotation3& rotation)
      : translation_(translation), rotation_(rotation) {}
  explicit Rigid3(const Vector3& translation, const AngleAxis& angle_axis)
      : translation_(translation), rotation_(angle_axis) {}

 private:
  Vector3 translation_;
  Rotation3 rotation_;

 public:
  Vector3& translation() { return translation_; }
  const Vector3& translation() const { return translation_; }
  Rotation3& rotation() { return rotation_; }
  const Rotation3& rotation() const { return rotation_; }

  Rigid3& normalize() {
    rotation_.normalize();
    return *this;
  }

  Rigid3 inversed() const {
    const Rotation3 rotation = rotation_.conjugate();
    const Vector3 translation = -(rotation * translation_);
    return Rigid3(translation, rotation);
  }

  template <typename OtherType>
  Rigid3<OtherType> cast() const {
    return Rigid3<OtherType>(translation_.template cast<OtherType>(),
                             rotation_.template cast<OtherType>());
  }

  static Rigid3 Translation(const Vector3& vector) {
    return Rigid3(vector, Rotation3::Identity());
  }

  static Rigid3 Translation(const FloatType x, const FloatType y,
                            const FloatType z) {
    return Translation(Vector3(x, y, z));
  }

  static Rigid3 Rotation(const Rotation3& rotation) {
    return Rigid3(Vector3::Zero(), rotation);
  }

  static Rigid3 Rotation(const AngleAxis& rotation) {
    return Rotation(Rotation3(rotation));
  }

  static Rigid3 Identity() { return Rigid3(); }

 private:
  std::string debugString() const {
    static const std::string rigid_debug_string[4] = {"T:(", ",", "), R:(",
                                                      ")"};
    return rigid_debug_string[0] + std::to_string(translation().x()) +
           rigid_debug_string[1] + std::to_string(translation().y()) +
           rigid_debug_string[1] + std::to_string(translation().z()) +
           rigid_debug_string[2] + std::to_string(rotation().w()) +
           rigid_debug_string[1] + std::to_string(rotation().x()) +
           rigid_debug_string[1] + std::to_string(rotation().y()) +
           rigid_debug_string[1] + std::to_string(rotation().z()) +
           rigid_debug_string[3];
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename FloatType>
inline Rigid3<FloatType> operator*(const Rigid3<FloatType>& lhs,
                                   const Rigid3<FloatType>& rhs) {
  return Rigid3<FloatType>(
      lhs.rotation() * rhs.translation() + lhs.translation(),
      lhs.rotation() * rhs.rotation());
}

template <typename FloatType>
inline typename Rigid3<FloatType>::Vector3 operator*(
    const Rigid3<FloatType>& rigid,
    const typename Rigid3<FloatType>::Vector3& point) {
  return (rigid.rotation() * point + rigid.translation());
}

template <typename FloatType>
inline Rigid3<FloatType> delta(const Rigid3<FloatType>& lhs,
                               const Rigid3<FloatType>& rhs) {
  Rigid3<FloatType> result = lhs.inversed() * rhs;
  result.normalize();
  return result;
}

template <typename FloatType>
inline Rigid3<FloatType> interpolate(const Rigid3<FloatType>& begin_rigid,
                                     const Rigid3<FloatType>& end_rigid,
                                     const FloatType interpolation) {
  using ceres::abs;
  using ceres::acos;
  using ceres::sin;
  const typename Rigid3<FloatType>::Vector3 translation =
      begin_rigid.translation() +
      (end_rigid.translation() - begin_rigid.translation()) * interpolation;
  FloatType scale0, scale1;
  FloatType d = begin_rigid.rotation().dot(end_rigid.rotation());
  FloatType absD = abs(d);
  if (absD >= FloatType(0.999)) {
    scale0 = FloatType(1) - interpolation;
    scale1 = interpolation;
  } else {
    FloatType theta = acos(absD);
    FloatType sinTheta = sin(theta);
    scale0 = sin((FloatType(1) - interpolation) * theta) / sinTheta;
    scale1 = sin(interpolation * theta) / sinTheta;
  }
  if (d < FloatType(0)) scale1 = -scale1;
  const typename Rigid3<FloatType>::Rotation3 rotation(
      scale0 * begin_rigid.rotation().coeffs() +
      scale1 * end_rigid.rotation().coeffs());
  return Rigid3<FloatType>(translation, rotation);
}

using Rigid3d = Rigid3<double>;
using Rigid3f = Rigid3<float>;
}  // namespace transform
}  // namespace ctlo