#include "utils.h"
#include <unistd.h>

std::string getLastSubstringAfterSlash(const std::string &input) {
  auto lastSlashPos = input.find_last_of('/');

  if (lastSlashPos == std::string::npos)
    return input;

  return input.substr(lastSlashPos + 1);
}

bool isNan(double v) { return v != v; }

llvm::Expected<long> get_current_rss() {
  FILE *fp = fopen("/proc/self/statm", "r");
  if (fp) {
    long rss = 0L;
    if (fscanf(fp, "%*ld%ld", &rss) == 1) {
      rss = rss * sysconf(_SC_PAGESIZE) / 1024;
    }
    fclose(fp);
    return rss;
  } else
    return llvm::make_error<llvm::StringError>(
        "Unable to open /proc/self/statm. Are we running on Linux?",
        llvm::inconvertibleErrorCode());
}
