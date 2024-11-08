#include "embedding.h"
#include "jit/Core.h"
#include "jit/JIT.h"
#include <dlfcn.h>
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

int initializeEmbedding(int argc, char *argv[], int &i) {
  // file with list of modules separated by new line
  std::string approximableModulesFile = "";
  std::string preciseModulesFile = "";
  std::string forbiddenApproximationsFile = "";
  std::string evalModule = "";

  // arg parsing
  for (i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--eval_mod") == 0)
      evalModule = argv[++i];
    else if (strcmp(argv[i], "--app_mod") == 0)
      approximableModulesFile = argv[++i];
    else if (strcmp(argv[i], "--prec_mod") == 0)
      preciseModulesFile = argv[++i];
    else if (strcmp(argv[i], "--forbidden_approx") == 0)
      forbiddenApproximationsFile = argv[++i];
    else
      break;
  }

  if (evalModule == "")
    return -1;

  // auto *lib = dlopen("libtorch_cpu.so", RTLD_LAZY | RTLD_GLOBAL);
  // if (lib == nullptr) {
  //   fprintf(stderr, "dlopen libtorch_cpu_approx failure: %s\n", dlerror());
  //   return 1;
  // }

  //  start JIT with argv[1] as evaluation module file
  return start_JIT(evalModule, approximableModulesFile, preciseModulesFile,
                   forbiddenApproximationsFile);
}

void endEmbedding() { printOpportunities(true); }

void printOpportunities(bool csv_format) {
  J->printRankedOpportunities(csv_format);
}
