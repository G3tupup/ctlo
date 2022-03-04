#pragma once

#include <iostream>
#include <string>

#include "../transform/rigid3.hpp"

namespace ctlo {
namespace point {
template <typename T>
class EIGEN_ALIGN16 Ndt3 {
  friend std::ostream& operator<<(std::ostream& os, const Ndt3<T>& ndt3) {
    os << ndt3.debugString();
    return os;
  }

 public:
  using FloatType = T;

  explicit Ndt3()
      : Ndt3(0, Eigen::Matrix<FloatType, 3, 1>::Zero(),
             Eigen::Matrix<FloatType, 3, 3>::Zero(),
             Eigen::Matrix<FloatType, 3, 3>::Zero()) {}

  explicit Ndt3(const size_t num, const Eigen::Matrix<FloatType, 3, 1>& mean,
                const Eigen::Matrix<FloatType, 3, 3>& n_var)
      : num_(num), mean_(mean), n_var_(n_var), if_need_to_update_var_(true) {}

  explicit Ndt3(const size_t num, const Eigen::Matrix<FloatType, 3, 1>& mean,
                const Eigen::Matrix<FloatType, 3, 3>& n_var,
                const Eigen::Matrix<FloatType, 3, 3>& var)
      : num_(num),
        mean_(mean),
        n_var_(n_var),
        var_(var),
        if_need_to_update_var_(false) {}

  explicit Ndt3(const Eigen::Matrix<FloatType, 3, 1>& mean,
                const Eigen::Matrix<FloatType, 3, 3>& var =
                    Eigen::Matrix<FloatType, 3, 3>::Zero())
      : Ndt3(1, mean, var, var) {}

 private:
  size_t num_;
  Eigen::Matrix<FloatType, 3, 1> mean_;
  Eigen::Matrix<FloatType, 3, 3> n_var_;
  mutable Eigen::Matrix<FloatType, 3, 3> var_;
  mutable bool if_need_to_update_var_;

 public:
  const size_t num() const { return num_; }

  const Eigen::Matrix<FloatType, 3, 1>& mean() const { return mean_; }

  const Eigen::Matrix<FloatType, 3, 3>& nVar() const { return n_var_; }

  const Eigen::Matrix<FloatType, 3, 3>& var() const {
    updateVar();
    return var_;
  }

  const bool ifNeedToUpdateVar() const { return if_need_to_update_var_; }

  const FloatType operator[](const size_t dim) const { return mean_[dim]; }

  Ndt3& operator+=(const Ndt3<T>& rhs) {
    const size_t new_num = num_ + rhs.num_;
    const Eigen::Matrix<T, 3, 1> delta = rhs.mean_ - mean_;
    mean_ += (static_cast<T>(rhs.num_) / static_cast<T>(new_num)) * delta;
    n_var_ += (rhs.n_var_ +
               (delta * delta.transpose()) *
                   (static_cast<T>(num_ * rhs.num_) / static_cast<T>(new_num)));
    num_ = new_num;
    if_need_to_update_var_ = true;
    return *this;
  }

  Ndt3& operator-=(const Ndt3<T>& rhs) {
    assert(num_ >= rhs.num_);
    const size_t new_num = num_ - rhs.num_;
    if (0 == new_num) {
      *this = Ndt3();
    } else {
      const Eigen::Matrix<T, 3, 1> delta = rhs.mean_ - mean_;
      mean_ -= (static_cast<T>(rhs.num_) / static_cast<T>(new_num)) * delta;
      n_var_ -= (rhs.n_var_ + (delta * delta.transpose()) *
                                  (static_cast<T>(num_ * rhs.num_) /
                                   static_cast<T>(new_num)));
      num_ = new_num;
      if_need_to_update_var_ = true;
    }
    return *this;
  }

 private:
  void updateVar() const {
    if (if_need_to_update_var_) {
      var_ = Eigen::Matrix<FloatType, 3, 3>::Zero();
      if (num_ > 1) {
        var_ += n_var_ / static_cast<FloatType>(num_ - 1);
      }
      if_need_to_update_var_ = false;
    }
  }

  std::string debugString() const {
    static const std::string debug_string[5] = {"Num: ", "Mean: ", ", ", "\n",
                                                "Var:"};
    return debug_string[0] + std::to_string(num()) + debug_string[2] +
           debug_string[1] + std::to_string(mean()[0]) + debug_string[2] +
           std::to_string(mean()[1]) + debug_string[2] +
           std::to_string(mean()[2]) + debug_string[3] + debug_string[4] +
           debug_string[3] + std::to_string(var()(0, 0)) + debug_string[2] +
           std::to_string(var()(0, 1)) + debug_string[2] +
           std::to_string(var()(0, 2)) + debug_string[3] +
           std::to_string(var()(1, 0)) + debug_string[2] +
           std::to_string(var()(1, 1)) + debug_string[2] +
           std::to_string(var()(1, 2)) + debug_string[3] +
           std::to_string(var()(2, 0)) + debug_string[2] +
           std::to_string(var()(2, 1)) + debug_string[2] +
           std::to_string(var()(2, 2)) + debug_string[3];
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T>
Ndt3<T> operator+(const Ndt3<T>& lhs, const Ndt3<T>& rhs) {
  Ndt3<T> result = lhs;
  result += rhs;
  return result;
}

template <typename T>
Ndt3<T> operator-(const Ndt3<T>& lhs, const Ndt3<T>& rhs) {
  Ndt3<T> result = lhs;
  result -= rhs;
  return result;
}

template <typename T>
inline Ndt3<T> transform(const Ndt3<T>& ndt, const transform::Rigid3<T>& rigid3,
                         const Eigen::Matrix<T, 3, 3>& rot,
                         const Eigen::Matrix<T, 3, 3>& rot_t) {
  return Ndt3<T>(ndt.num(), rigid3 * ndt.mean(), rot * ndt.nVar() * rot_t);
}

using Ndt3d = Ndt3<double>;
using Ndt3f = Ndt3<float>;
}  // namespace point
}  // namespace ctlo