#include "CustomDemangler.h"
#include "cxxabi.h"

using namespace llvm;

// return mangled function name
std::string CustomDemangler::getFunction(std::string function) {
  return mangledMap[function];
}

Expected<CustomDemangler> CustomDemangler::Create(const Module *M) {
  StringMap<std::string> mangledMap;
  size_t count = 0;
  // search for the desired functions inside the module
  for (auto &F : *M) {
    // already got all functions
    if (count == functionNames.size())
      break;
    std::string mangled = F.getName().str();
    int status;
    // demangle the name
    const char *demangled =
        abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
    // not valid mangled name, implies no demangling is needed
    if (status == -2)
      demangled = mangled.c_str();
    else {
      std::string demangled_str = demangled;
      // remove everything beyond the function name
      demangled = demangled_str.substr(0, demangled_str.find("(")).c_str();
    }

    auto it = std::find(functionNames.begin(), functionNames.end(), demangled);
    if (it != functionNames.end()) {
      mangledMap[*it] = mangled;
      count++;
    }
  }

  if (count == functionNames.size())
    return CustomDemangler(mangledMap);

  std::string missing;
  for (auto functionName : functionNames) {
    if (mangledMap.find(functionName) == mangledMap.end())
      missing += functionName + ",";
  }

  missing.pop_back();
  return make_error<StringError>("Functions missing from evaluation file:" +
                                     missing,
                                 inconvertibleErrorCode());
}

CustomDemangler::CustomDemangler(StringMap<std::string> MM) : mangledMap(MM) {}

// unmangled -> mangled map
StringMap<std::string> mangledMap;
// list of functions to lookup mangled names into
const std::set<std::string> CustomDemangler::functionNames = {"storeOriginal",
                                                              "compare"};
