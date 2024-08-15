#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include <set>
#include <string>

// front-end ABI demangler for IR demangling needs
class CustomDemangler {
public:
  // return mangled function name
  std::string getFunction(std::string function);

  static llvm::Expected<CustomDemangler> Create(const llvm::Module *M);

private:
  CustomDemangler(llvm::StringMap<std::string> MM);
  // unmangled -> mangled map
  llvm::StringMap<std::string> mangledMap;
  // list of functions to lookup mangled names into
  static const std::set<std::string> functionNames;
};
