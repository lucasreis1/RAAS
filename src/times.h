#pragma once
#include <chrono>
#define CSV_FILE "/tmp/overhead_times.csv"

inline std::chrono::time_point<std::chrono::steady_clock>& get_start_time() {
  static std::chrono::time_point<std::chrono::steady_clock> start_time;
  return start_time;
}

// this is the time we start applying approximations
inline std::chrono::time_point<std::chrono::steady_clock>& get_approx_start_time() {
  static std::chrono::time_point<std::chrono::steady_clock> approx_start_time;
  return approx_start_time;
}
