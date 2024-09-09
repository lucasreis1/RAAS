#include "embedding.h"
#include "jit/Core.h"
#include "jit/JIT.h"
#include <fstream>

std::unique_ptr<llvm::orc::ApproxJIT> J;

bool start_JIT(std::string evalBCFile, std::string appModulesFile,
               std::string preciseModulesFile,
               std::string forbiddenApproximationsFile) {
  // JIT already initialized, leave
  if (J.get() != nullptr) {
    return false;
  }

  initializeTarget();
  J = ExitOnErr(llvm::orc::ApproxJIT::Create(evalBCFile));
  // TODO: fix platform initialization
  // ExitOnErr(J->initializePlatform());

  if (forbiddenApproximationsFile != "") {
    auto forbiddenApproxList =
        ForbiddenApproximations::Create(forbiddenApproximationsFile);
    if (!forbiddenApproxList)
      llvm_unreachable("Forbidden approximation file invalid!");
    J->setForbiddenApproxList(std::move(*forbiddenApproxList));
  }

  if (appModulesFile == "")
    return true;

  std::ifstream FileStream(appModulesFile);

  if (!FileStream.is_open()) {
    fprintf(stderr, "Unable to locate module file %s", appModulesFile.c_str());
    return false;
  }

  std::string mod;
  while (std::getline(FileStream, mod)) {
    if (auto Err = J->addModuleApproxFile(mod)) {
      fprintf(stderr, "Unable to load approx module %s\n", mod.c_str());
      exit(1);
    }
  }

  FileStream.close();

  if (preciseModulesFile == "")
    return true;

  FileStream.open(preciseModulesFile);

  if (!FileStream.is_open()) {
    fprintf(stderr, "Unable to locate module file %s",
            preciseModulesFile.c_str());
    return false;
  }

  while (std::getline(FileStream, mod)) {
    if (auto Err = J->addModuleFile(mod)) {
      fprintf(stderr, "Unable to load precise module %s\n", mod.c_str());
      exit(1);
    }
  }

  return true;
}

void printOpportunities() {
  J->printRankedOpportunities();
}
