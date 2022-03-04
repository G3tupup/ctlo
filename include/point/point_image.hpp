#pragma once

#include <array>
#include <unordered_set>

namespace ctlo {
namespace point {
template <typename PointT, std::size_t Height = 1, std::size_t Width = 1>
class PointImage {
 public:
  PointImage() = default;

 private:
  static constexpr size_t height_ = Height;
  static constexpr size_t width_ = Width;
  static constexpr size_t size_ = height_ * width_;
  std::array<PointT, size_> points_;
  std::unordered_set<size_t> marked_indices_;

 public:
  static size_t validCol(const int col) {
    int ncol = col;
    while (ncol < 0) {
      ncol += width_;
    }
    if (ncol >= width_) {
      ncol = ncol % width_;
    }
    return static_cast<size_t>(ncol);
  }

  static size_t index(const int row, const int col) {
    return static_cast<size_t>(row) * width_ + validCol(col);
  }

  static size_t index(const size_t row, const size_t col) {
    return row * width_ + col;
  }

  const PointT& operator()(const int row, const int col) const {
    return points_.at(index(row, col));
  }

  PointT& operator()(const int row, const int col) {
    return points_.at(index(row, col));
  }

  const PointT& operator()(const size_t row, const size_t col) const {
    return points_.at(index(row, col));
  }

  PointT& operator()(const size_t row, const size_t col) {
    return points_.at(index(row, col));
  }

  void clearMarks() { marked_indices_.clear(); }

  void mark(const int row, const int col) {
    auto i = index(row, col);
    if (marked_indices_.find(i) == marked_indices_.end())
      marked_indices_.emplace(i);
  }

  bool ifMarked(const size_t row, const size_t col) {
    return (marked_indices_.find(index(row, col)) != marked_indices_.end());
  }
};
}  // namespace point
}  // namespace ctlo