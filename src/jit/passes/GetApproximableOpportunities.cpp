/* * * * * * * * * * * * * * * * * * * * * * *
 *                                           *
 * This pass parses a module and prints all  *
 * approximable opportunities found inside   *
 *                                           *
 * * * * * * * * * * * * * * * * * * * * * * */

#include "Passes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

class ApproxOpportunityList : public PassInfoMixin<ApproxOpportunityList> {
public:
  ApproxOpportunityList() { passlist::buildPasses(); }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    fprintf(stderr, "Searching for approximation opportunities in module %s\n",
            M.getName().str().c_str());
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      fprintf(stderr, "%d approx opportunities in Function %s\n",
              passlist::searchApproximableCalls(F), F.getName().str().c_str());
    }
    return PreservedAnalyses::all();
  }

  static StringRef name() { return "approxopplist"; };
};

/* New PM Registration */
llvm::PassPluginLibraryInfo getInstrFnPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ApproxOppList", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager &PM, OptimizationLevel Level) {
                  PM.addPass(ApproxOpportunityList());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::ModulePassManager &MPM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == ApproxOpportunityList::name()) {
                    MPM.addPass(ApproxOpportunityList());
                    return true;
                  }
                  return false;
                });
          }};
}

#ifndef LLVM_INSTRFN_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getInstrFnPluginInfo();
}
#endif
