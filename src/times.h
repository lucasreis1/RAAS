#pragma once
#include <chrono>

inline std::chrono::time_point<std::chrono::steady_clock>& get_start_time() {
  static std::chrono::time_point<std::chrono::steady_clock> start_time;
  return start_time;
}
