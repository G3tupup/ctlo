#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../common/math.hpp"

namespace ctlo {
namespace time_counter {
namespace {
struct TimeDetail {
  size_t num;
  double min;
  double max;
  double mean;
  std::vector<double> values;
};

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using TimeDuration = std::chrono::nanoseconds;

static std::unordered_map<std::string, TimePoint> g_tick_time_points;
static std::unordered_map<std::string, TimeDetail> g_details;

inline std::string toString2D(const double count) {
  std::ostringstream oss;
  oss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << count;
  return oss.str();
}

inline std::string toString3SD(const double count) {
  std::ostringstream oss;
  oss << std::setprecision(3) << count;
  return oss.str();
}
}  // namespace

inline std::string toStringDuration(double count) {
  static const std::string duration_name[7]{"ns",  "us", "ms", "s",
                                            "min", "h",  "d"};
  if (count < 1000.0) return toString2D(count) + duration_name[0];
  count /= 1000.0;
  if (count < 1000.0) return toString2D(count) + duration_name[1];
  count /= 1000.0;
  if (count < 1000.0) return toString2D(count) + duration_name[2];
  count /= 1000.0;
  if (count < 60.0) return toString2D(count) + duration_name[3];
  long int long_count = static_cast<long int>(count);
  std::string fixed = std::to_string(long_count % 60l) + duration_name[3];
  long_count /= 60l;
  if (long_count < 60l)
    return std::to_string(long_count) + duration_name[4] + fixed;
  fixed = std::to_string(long_count % 60l) + duration_name[4] + fixed;
  long_count /= 60l;
  if (long_count < 24l)
    return std::to_string(long_count) + duration_name[5] + fixed;
  fixed = std::to_string(long_count % 24l) + duration_name[5] + fixed;
  long_count /= 24l;
  return std::to_string(long_count) + duration_name[6] + fixed;
}

inline std::string toStringDuration(const TimeDuration& duration) {
  return toStringDuration(static_cast<double>(duration.count()));
}

inline void tick(const std::string& topic = std::string()) {
  g_tick_time_points[topic] = Clock::now();
}

inline void tack(const std::string& topic = std::string()) {
  auto iter = g_tick_time_points.find(topic);
  if (iter == g_tick_time_points.end()) {
    std::cerr << topic << " hasn't ticked!" << std::endl;
  }
  TimeDuration duration =
      std::chrono::duration_cast<TimeDuration>(Clock::now() - iter->second);
  std::cout << topic << " costs " << toStringDuration(duration) << std::endl;
  g_tick_time_points.erase(iter);
}

inline void tock(const std::string& topic) {
  auto iter = g_tick_time_points.find(topic);
  if (iter == g_tick_time_points.end()) {
    std::cerr << topic << " hasn't ticked!" << std::endl;
  }
  TimeDuration duration =
      std::chrono::duration_cast<TimeDuration>(Clock::now() - iter->second);
  const double duration_count = static_cast<double>(duration.count());
  g_tick_time_points.erase(iter);
  auto detail_iter = g_details.find(topic);
  if (detail_iter != g_details.end()) {
    ++detail_iter->second.num;
    const double tmp = (duration_count - detail_iter->second.mean) /
                       static_cast<double>(detail_iter->second.num);
    detail_iter->second.mean += tmp;
    detail_iter->second.min = duration_count < detail_iter->second.min
                                  ? duration_count
                                  : detail_iter->second.min;
    detail_iter->second.max = duration_count > detail_iter->second.max
                                  ? duration_count
                                  : detail_iter->second.max;
    detail_iter->second.values.emplace_back(duration_count);
  } else {
    g_details[topic] = {
        1, duration_count, duration_count, duration_count, {duration_count}};
  }
}

inline void output(const std::string& topic) {
  auto detail_iter = g_details.find(topic);
  if (detail_iter != g_details.end()) {
    double median, sigma;
    std::tie(median, sigma) = math::mad(detail_iter->second.values);
    std::array<size_t, 8> nums;
    nums.fill(0);
    for (const auto& value : detail_iter->second.values) {
      if (value < median - 3. * sigma) {
        ++nums[0];
      } else if (value < median - 2. * sigma) {
        ++nums[1];
      } else if (value < median - sigma) {
        ++nums[2];
      } else if (value < median) {
        ++nums[3];
      } else if (value < median + sigma) {
        ++nums[4];
      } else if (value < median + 2. * sigma) {
        ++nums[5];
      } else if (value < median + 3. * sigma) {
        ++nums[6];
      } else {
        ++nums[7];
      }
    }
    int start_index = 0, end_index = 7;
    for (int i = 0; i < 8; ++i) {
      if (detail_iter->second.min <
          median + sigma * (static_cast<double>(i) - 3.)) {
        start_index = i;
        break;
      }
    }
    for (int i = 7; i >= 0; --i) {
      if (detail_iter->second.max >
          median + sigma * (static_cast<double>(i) - 4.)) {
        end_index = i;
        break;
      }
    }
    std::string prefixs[8];
    double percentages[8];
    for (int i = start_index; i <= end_index; ++i) {
      auto& prefix = prefixs[i];
      if (i == start_index) {
        prefix += " [" + toStringDuration(detail_iter->second.min);
      } else {
        prefix += " [" + toStringDuration(
                             median + sigma * (static_cast<double>(i) - 4.));
      }
      if (i == end_index) {
        prefix += ", " + toStringDuration(detail_iter->second.max) + "]: ";
      } else {
        prefix +=
            ", " +
            toStringDuration(median + sigma * (static_cast<double>(i) - 3.)) +
            "): ";
      }
      percentages[i] = static_cast<double>(nums[i]) /
                       static_cast<double>(detail_iter->second.num) * 100.;
      prefix += std::to_string(nums[i]) + "/" +
                std::to_string(detail_iter->second.num) + "(" +
                toString3SD(percentages[i]) + "%) ▏";
    }
    auto iter =
        std::max_element(prefixs, prefixs + 8,
                         [](const std::string& lhs, const std::string& rhs) {
                           return lhs.size() < rhs.size();
                         });
    static const std::string bars[8]{"█", "▉", "▊", "▋", "▌", "▍", "▎", "▏"};
    std::string histogram;
    for (int i = start_index; i <= end_index; ++i) {
      std::string blanks;
      for (size_t j = 0; j < (iter->size() - prefixs[i].size()); ++j) {
        blanks += " ";
      }
      auto pos = prefixs[i].find(':');
      prefixs[i].insert(pos + 1, blanks);
      histogram += prefixs[i];
      int bar_num = (percentages[i] * 0.2);
      for (int j = 0; j < bar_num; ++j) {
        histogram += bars[0];
      }
      double res = percentages[i] - 5. * static_cast<double>(bar_num);
      int n = std::lround(res * 1.6);
      if (n != 0) {
        histogram += bars[8 - n];
      }
      histogram += "\n";
    }
    std::cout << topic << " costs "
              << toStringDuration(detail_iter->second.mean) << " on the average"
              << std::endl
              << histogram << std::endl;
  } else {
    std::cerr << topic << " hasn't ticked!" << std::endl;
  }
}

inline void clear(const std::string& topic) {
  auto detail_iter = g_details.find(topic);
  if (detail_iter != g_details.end()) {
    g_details.erase(detail_iter);
  } else {
    std::cerr << topic << " hasn't ticked!" << std::endl;
  }
}

inline void clearAll() { g_details.clear(); }
}  // namespace time_counter
}  // namespace ctlo