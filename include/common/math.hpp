#pragma once

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <vector>

namespace ctlo {
namespace math {
template <typename T>
inline constexpr T limitZero() {
  return T(1e-6);
}

template <typename T>
inline constexpr T pi() {
  return T(3.1415926535897932384626);
}

template <typename T>
inline constexpr T twoPi() {
  return pi<T>() * T(2);
}

template <typename T>
inline constexpr T halfPi() {
  return pi<T>() * T(0.5);
}

template <typename T>
inline constexpr T pow2(const T& value) {
  return value * value;
}

template <typename T>
inline constexpr T pow3(const T& value) {
  return value * value * value;
}

template <typename T>
inline constexpr bool ifZero(const T& value) {
  return std::abs(value) <= limitZero<T>();
}

template <typename T>
inline std::pair<T, T> mad(std::vector<T>& values) {
  std::nth_element(values.begin(), values.begin() + values.size() / 2,
                   values.end());
  const T median = values[values.size() / 2];
  std::vector<T> bias(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bias[i] = std::abs(values[i] - median);
  }
  std::nth_element(bias.begin(), bias.begin() + values.size() / 2, bias.end());
  return std::make_pair(median, T(1.4826) * bias[values.size() / 2]);
}

namespace {
template <typename T>
inline constexpr T ck1() {
  return T(-0.0464964749);
}

template <typename T>
inline constexpr T ck2() {
  return T(0.15931422);
}

template <typename T>
inline constexpr T ck3() {
  return T(-0.327622764);
}
}  // namespace

template <typename T>
inline T fastAtan2(const T y, const T x) {
  const T ax = std::abs(x), ay = std::abs(y);
  const T a = std::min(ax, ay) / (std::max(ax, ay) + limitZero<T>());
  const T s = pow2(a);
  T r = ((ck1<T>() * s + ck2<T>()) * s + ck3<T>()) * s * a + a;
  if (ay > ax) r = halfPi<T>() - r;
  if (x < 0) r = pi<T>() - r;
  if (y < 0) r = -r;
  return r;
}
}  // namespace math
}  // namespace ctlo