#include "ApproxLayer.h"
#include "Core.h"
#include "misc/CustomDemangler.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/EPCIndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"

#define BASE_ITERATIONS 5
#define SKIPPABLE_LOOPS 2

#define DEBUG_TYPE "raas"

static llvm::ExitOnError ExitOnErr;

void initializeTarget();

#ifdef __cplusplus
extern "C"
#endif
    void *
    $jump_to_jit(const char *fnName);

namespace llvm {
namespace orc {
class ApproxJIT {
private:
  std::unique_ptr<ExecutionSession> ES;
  std::unique_ptr<EPCIndirectionUtils> EPCIU;

  DataLayout DL;
  MangleAndInterner Mangle;

  // Layers
  ObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;
  // general optimization layer
  IRTransformLayer TransformLayer;
  // structures for dealing with approximations
  ApproxLayer APLayer;
  // run general IPO transformartions before sending module to APLayer
  IRTransformLayer ModuleTransformLayer;

  EvaluationSystem evaluator;

  CustomDemangler demangler;

  // main dylib
  JITDylib &MainJD;
  // support dylib (standard symbols from process + original/instrumented
  // symbols for approximable functions)
  // JITDylib &SupportJD;

  // DSO handles for initalizers
  DenseMap<orc::JITDylib *, orc::ExecutorAddr> DSOHandles;

  static void handleLazyCallThroughError() {
    errs() << "LazyCallThrough error: Could not find function body";
    exit(1);
  }

  const DataLayout &getDataLayout() const { return DL; }

  int numberOfLoops = 0;

  // Loop maths for evaluation
  bool isOnSkippableLoops() { return numberOfLoops < SKIPPABLE_LOOPS; }

  bool isOnBaseLoops() {
    return !isOnSkippableLoops() and
           numberOfLoops < BASE_ITERATIONS + SKIPPABLE_LOOPS;
  }

  bool isOnApproximableLoops() {
    return !isOnBaseLoops() and !isOnSkippableLoops();
  }

  int getLoopNumber() { return this->numberOfLoops - SKIPPABLE_LOOPS + 1; }

public:
  ApproxJIT(std::unique_ptr<ExecutionSession> ES,
            std::unique_ptr<EPCIndirectionUtils> EPCIU,
            JITTargetMachineBuilder JTMB, DataLayout DL,
            CustomDemangler demangler, ThreadSafeModule evalModule);

  ~ApproxJIT();

  static Expected<std::unique_ptr<ApproxJIT>>
  Create(std::string AppModule, std::string evalModuleFile);
  static Expected<std::unique_ptr<ApproxJIT>>
  Create(std::string evalModuleFile);

  Expected<ExecutorSymbolDef>
  lookup(StringRef Name, JITDylibLookupFlags Flags =
                             JITDylibLookupFlags::MatchExportedSymbolsOnly);
  Expected<ExecutorSymbolDef> lookupOrLoadSymbol(StringRef Name,
                                                 StringRef ModuleLoc);

  /**
   * collects output information and starts decision making to re-select
   * approximations
   */
  void approxReevaluation();

  // platform shenanigans
  Error initializePlatform();
  Error deinitializePlatform();

  /**
   * adds approximable module file to JIT
   */
  Error addModuleApproxFile(std::string moduleFile);

  /**
   * adds module file to JIT outside approximation scope
   */
  Error addModuleFile(std::string moduleFile);

  /**
   * adds module file to JIT outside approximation scope
   */
  static Expected<ThreadSafeModule> loadModule(std::string bitcodeFile);

  void setForbiddenApproxList(std::unique_ptr<ForbiddenApproximations> fapList);

  /**
   * run main function from module
   */
  void runAsMain(ExecutorAddr mainAddr, ArrayRef<std::string> args) {
    ExitOnErr(EPCIU->getExecutorProcessControl().runAsMain(mainAddr, args));
  }

  /**
   * Print configuration info from the evalution system. This is arbitrary
   * and will depend on the evaluator selected. Clients must ensure this is
   * only called after convergence, as this MAY reorder
   * data structures that store configurations
   */
  void printRankedOpportunities(bool csv_format = false) {
    evaluator.printRankedOpportunities(csv_format);
  }
};

void enableDebug();

} // namespace orc
} // namespace llvm
