#pragma once

#include <bitset>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "../common/common.hpp"

namespace ctlo {
namespace voxel {
template <typename T>
inline Eigen::Array3i getVoxelIndex(const T x, const T y, const T z,
                                    const T voxel_size_inv) {
  return {static_cast<int>(std::floor(x * voxel_size_inv)),
          static_cast<int>(std::floor(y * voxel_size_inv)),
          static_cast<int>(std::floor(z * voxel_size_inv))};
}

template <typename PointT>
inline Eigen::Array3i getVoxelIndex(
    const PointT& point, const typename PointT::FloatType voxel_size_inv) {
  return getVoxelIndex(point[0], point[1], point[2], voxel_size_inv);
}

template <typename PointT>
inline bool doNothing(PointT& lhs, const PointT& rhs) {
  return true;
}

template <typename PointT>
inline bool replace(PointT& lhs, const PointT& rhs) {
  lhs = rhs;
  return true;
}

template <typename PointT>
inline bool merge(PointT& lhs, const PointT& rhs) {
  lhs += rhs;
  return true;
}

template <typename PointT>
inline bool split(PointT& lhs, const PointT& rhs) {
  lhs -= rhs;
  return (lhs.num() == 0);
}

template <typename PointT, template <typename> class Container>
class VoxelGrid {
 public:
  using FloatType = typename PointT::FloatType;
  using Operation = std::function<bool(PointT&, const PointT&)>;
  using HashMapType =
      std::unordered_map<Eigen::Array3i, size_t, common::Array3iHash,
                         common::Array3iEqual>;

 public:
  VoxelGrid(const FloatType voxel_size = default_voxel_size_,
            Operation add = doNothing<PointT>,
            Operation remove = doNothing<PointT>)
      : add_(add), remove_(remove) {
    setVoxelSize(voxel_size);
  }
  VoxelGrid(const VoxelGrid& rhs) = default;
  VoxelGrid& operator=(const VoxelGrid& rhs) = default;

 private:
  static constexpr FloatType default_voxel_size_{0.05};
  Operation add_, remove_;
  FloatType voxel_size_inv_;
  HashMapType key_map_;
  Container<PointT> points_;
  std::vector<size_t> vacancy_;

 public:
  const Container<PointT>& points() const { return points_; }

  const PointT& operator[](const size_t index) const { return points_[index]; }

  const size_t size() const { return points_.size(); }

  const FloatType voxelSizeInv() const { return voxel_size_inv_; }

  const FloatType voxelSize() const { return FloatType(1) / voxel_size_inv_; }

  const size_t get(const Eigen::Array3i& index) const {
    auto iter = key_map_.find(index);
    if (iter != key_map_.end()) {
      return iter->second;
    }
    return std::numeric_limits<size_t>::max();
  }

  const size_t get(const PointT& point) const {
    return get(getVoxelIndex(point, voxel_size_inv_));
  }

  void set(const Container<PointT>& point_cloud) {
    clear();
    add(point_cloud);
  }

  void add(const Container<PointT>& point_cloud) {
    for (const auto& point : point_cloud) add(point);
  }

  void add(const PointT& point) {
    const auto index = getVoxelIndex(point, voxel_size_inv_);
    auto iter = key_map_.find(index);
    if (iter != key_map_.end()) {
      add_(points_[iter->second], point);
    } else {
      key_map_[index] = vacantAdd(point);
    }
  }

  void remove(const Container<PointT>& point_cloud) {
    for (const auto& point : point_cloud) remove(point);
  }

  void remove(const PointT& point) {
    auto iter = key_map_.find(getVoxelIndex(point, voxel_size_inv_));
    assert(iter != key_map_.end());
    if (remove_(points_[iter->second], point)) {
      vacancy_.push_back(iter->second);
      key_map_.erase(iter);
    }
  }

  void clear() {
    points_.clear();
    key_map_.clear();
    vacancy_.clear();
  }

  void setVoxelSize(const FloatType voxel_size) {
    voxel_size_inv_ = FloatType(1) / voxel_size;
  }

  void setAddOperation(Operation add) { add_ = add; }

  void setRemoveOperation(Operation remove) { remove_ = remove; }

 private:
  const size_t vacantAdd(const PointT& point) {
    size_t result = points_.size();
    if (vacancy_.empty()) {
      points_.push_back(point);
    } else {
      result = vacancy_.back();
      vacancy_.pop_back();
      points_[result] = point;
    }
    return result;
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace voxel
}  // namespace ctlo
