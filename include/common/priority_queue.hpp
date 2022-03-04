#pragma once

#include <queue>

namespace ctlo {
namespace common {
template <class T, class Compare>
class PriorityQueue : public std::priority_queue<T, std::vector<T>, Compare> {
 public:
  using size_type =
      typename std::priority_queue<T, std::vector<T>, Compare>::size_type;

 public:
  PriorityQueue(const size_type capacity = 16) { reserve(capacity); }

  void reserve(const size_type capacity) { this->c.reserve(capacity); }
};
}  // namespace common
}  // namespace ctlo