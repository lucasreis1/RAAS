#include "utils.h"

std::string getLastSubstringAfterSlash(const std::string &input) {
  auto lastSlashPos = input.find_last_of('/');

  if (lastSlashPos == std::string::npos)
    return input;

  return input.substr(lastSlashPos+1);
}

bool isNan(double v) { return v != v; }
