#include "Passes.h"
#include "llvm/IR/PassManager.h"

namespace passlist {
std::list<std::shared_ptr<ApproximablePass>> Passes;

void buildPasses() {
  if (Passes.size())
    return;
  Passes.push_front(std::move(std::make_unique<FunctionApproximation>()));
  Passes.push_front(std::move(std::make_unique<LoopPerforation>()));
  Passes.push_front(std::move(std::make_unique<GEMMApproximation>()));
}

void addPassesToPM(ModulePassManager &MPM) {
  MPM.addPass(createModuleToFunctionPassAdaptor(FunctionApproximation()));
  MPM.addPass(createModuleToFunctionPassAdaptor(LoopPerforation()));
  MPM.addPass(createModuleToFunctionPassAdaptor(GEMMApproximation()));
}

bool isApproximable(const Function &F) {
  // skip delcarations
  if (F.isDeclaration())
    return false;

  for (auto &apprPass : Passes) {
    if (apprPass->isApproximable(F))
      return true;
  }

  return false;
}

void printApproxOpportunities(const Function &F) {
  // skip delcarations
  if (F.isDeclaration())
    return;

  for (auto &apprPass : Passes) {
    bool hasApprox = false;
    if (apprPass->isApproximable(F)) {
      if (!hasApprox) {
        errs() << "=============== " << F.getName() << " ===============\n";
        hasApprox = true;
      }
      apprPass->printApproximationOpportunities(F);
    }
  }
}

void printApproxOpportunities(const Module &M) {
  for (const auto &F : M)
    printApproxOpportunities(F);
}

unsigned searchApproximableCalls(const Function &F) {
  // skip delcarations
  if (F.isDeclaration())
    return 0;

  unsigned approxCalls = 0;
  for (auto &apprPass : Passes) {
    if (apprPass->isApproximable(F)) {
      approxCalls += apprPass->searchApproximableCalls(F);
    }
  }

  return approxCalls;
}

} // namespace passlist
