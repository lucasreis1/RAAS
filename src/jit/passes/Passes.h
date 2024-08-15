#pragma once
#include "../Core.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"

using namespace llvm;

/**
 * this class exists solely to transfer configuration data to other passes on
 * the optimization chain.
 */
class ApproximationAnalysis : public AnalysisInfoMixin<ApproximationAnalysis> {
  friend AnalysisInfoMixin<ApproximationAnalysis>;
  static AnalysisKey Key;

private:
  using fnNameMapPair = std::pair<StringRef, configurationPerTechniqueMap>;
  fnNameMapPair configurationMapPair;

  bool hasOpportunities(approxTechnique T);

public:
  struct ApproxAnalysisWrapperResult {
  public:
    ApproxAnalysisWrapperResult(configurationPerTechniqueMap configs)
        : configurationMap(configs) {}

    Expected<std::vector<int>> getConfiguration(approxTechnique T) {
      auto mapIter = configurationMap.find(T);
      if (mapIter == configurationMap.end())
        return make_error<StringError>("no approximations for this technique",
                                       inconvertibleErrorCode());
      return mapIter->second;
    }

  private:
    configurationPerTechniqueMap configurationMap;
  };

  // functionName, configMap pair
  ApproximationAnalysis(fnNameMapPair configPair);
  using Result = ApproxAnalysisWrapperResult;

  ApproxAnalysisWrapperResult run(Function &F, FunctionAnalysisManager &);
};

/**
 * This class defines primitives for each approximable pass.
 */
class ApproximablePass : public PassInfoMixin<ApproximablePass> {
public:
  virtual bool isApproximable(const Function &F) = 0;
  virtual unsigned searchApproximableCalls(const Function &F) = 0;
  virtual void printApproximationOpportunities(const Function &F) = 0;

  virtual ~ApproximablePass() = default;

  ApproximablePass(approxTechnique T) : T(T) {}
  const approxTechnique T;
};

namespace passlist {
extern std::list<std::shared_ptr<ApproximablePass>> Passes;
// build the list of approximable passes
void buildPasses();
// add passes to a MPM. 
// We need to do it manually instead of parsing through the list of passes because 
//  (a) the calls to add passes require ownership of the object 
//  (b) we can't pass an abstract class do the function
void addPassesToPM(ModulePassManager &MPM);
// checks if a function is approximable
bool isApproximable(const Function &F);

// prints opportunity list for a function
void printApproxOpportunities(const Function &F);
// prints opportunity list for a module
void printApproxOpportunities(const Module &F);
// counts the number of opportunities
unsigned searchApproximableCalls(const Function &F);
} // namespace passlist

/**
 * This approximation parses valid loops and applies perforation, executing only
 * 1/(2^p) * n iterations for each loop.
 */
class LoopPerforation : public ApproximablePass {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  bool isApproximable(const Function &F);
  unsigned searchApproximableCalls(const Function &F);
  void printApproximationOpportunities(const Function &F);

  static bool tryToPerforateLoop(Function *F, Loop *L, LoopInfo &LI,
                                 unsigned param);

  LoopPerforation() : ApproximablePass(LPERF) {}

private:
  class PerforableLoop {
  public:
    PerforableLoop(Loop *l, LoopInfo &li, Function *parentF, unsigned perfRate);

    ~PerforableLoop();

    static bool isPerforable(Loop *L);

    bool perforateLoop();

  private:
    IRBuilderBase *IRB;
    unsigned perforRate;
    Loop *L;
    LoopInfo &LI;
    Function *parentFun;
    BasicBlock *loopBody = nullptr;
    BasicBlock *loopLatch = nullptr;
  };

  static LoopInfo getLoopInfo(const Function &F);
};

/**
 * This approximation parses function calls that could be replaced by more
 * relaxed versions presente on the fastapprox library.
 */
class FunctionApproximation : public ApproximablePass {
public:
  bool isApproximable(const Function &F);
  unsigned searchApproximableCalls(const Function &F);
  void printApproximationOpportunities(const Function &F);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  FunctionApproximation() : ApproximablePass(FAP) {}

private:
  static const approxTechnique T = FAP;
};


/**
 * This approximation parses function calls to GEMM that could be replaced by more
 * relaxed versions present on the MKL library.
 */
class GEMMApproximation : public ApproximablePass {
public:
  bool isApproximable(const Function &F);
  unsigned searchApproximableCalls(const Function &F);
  void printApproximationOpportunities(const Function &F);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  GEMMApproximation() : ApproximablePass(GAP) {}

private:
  static const approxTechnique T = GAP;
};
