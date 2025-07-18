#include "Core.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/LazyReexports.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace orc;

class ApproxMaterializationUnit;

class ApproxLayer : public IRLayer {
  friend ApproxMaterializationUnit;

public:
  using approximableFunctions = std::set<std::string>;
  using IndirectStubsManagerBuilder =
      std::function<std::unique_ptr<IndirectStubsManager>()>;

  ApproxLayer(ExecutionSession &ES, DataLayout &DL, IRLayer &baseLayer,
              JITTargetAddress jitAddress, EvaluationSystem &evaluationSystem,
              LazyCallThroughManager &LCTMgr,
              IndirectStubsManagerBuilder BuildIndirectStubsManager);
  ApproxLayer(ExecutionSession &ES, DataLayout &DL, IRLayer &baseLayer,
              EvaluationSystem &evaluationSystem,
              LazyCallThroughManager &LCTMgr,
              IndirectStubsManagerBuilder BuildIndirectStubsManager,
              bool noApprox = false);

  void emit(std::unique_ptr<MaterializationResponsibility> R,
            ThreadSafeModule TSM) override;
  // emits approximable versions of functions
  void emitApprox(
      std::unique_ptr<MaterializationResponsibility> R,
      std::pair<std::string, configurationPerTechniqueMap> nameConfigPair);

  Expected<ExecutorSymbolDef>
  lookup(JITDylib &JD, SymbolStringPtr MangledName,
         JITDylibLookupFlags = JITDylibLookupFlags::MatchExportedSymbolsOnly);

  // given a function, add an approximate version for a specific approx.
  // combination
  llvm::Error addApproximateVersion(std::string functionName,
                                    configurationPerTechniqueMap configuration);

  MaterializationUnit::Interface
  getInterface(StringRef FunctionName,
               configurationPerTechniqueMap configuration);

  // update all approximations based on evaluator input
  Error updateApproximations();

private:
  struct PerFunctionResources {
  public:
    PerFunctionResources(JITDylib &Dylib, ThreadSafeModule TSM,
                         JITSymbolFlags flags)
        : Dylib(Dylib), TSM(std::move(TSM)), SymbolFlags(flags) {}

    JITDylib &getDylib() { return Dylib; }

    const ThreadSafeModule &getModule() { return TSM; }

    const Function *getFunction() {
      Function *Fn = nullptr;

      TSM.withModuleDo([&](Module &M) {
        for (auto &function : M)
          // there should only be one function defined in the module
          if (!function.isDeclaration()) {
            Fn = &function;
          }
      });
      return Fn;
    }

    llvm::Error addRT(ResourceTrackerSP RSP) {
      if (RT)
        return make_error<StringError>(
            "Function already has temp ResourceTracker associated with it!",
            inconvertibleErrorCode());

      RT = RSP;
      return Error::success();
    }

    llvm::Error removeRT() {
      if (RT == nullptr)
        return make_error<StringError>(
            "Function has no ResourceTracker associated with it!",
            inconvertibleErrorCode());
      RT = nullptr;

      return Error::success();
    }

    const ResourceTrackerSP getRT() { return RT; }

    const JITSymbolFlags getFlags() { return SymbolFlags; }

  private:
    JITDylib &Dylib;
    ThreadSafeModule TSM;
    JITSymbolFlags SymbolFlags;
    // each function has a resource tracker associated with it's approximate
    // symbols. This is intended do store symbols that will be discarded later
    // on
    ResourceTrackerSP RT = nullptr;
  };

  JITTargetAddress jitAddress;

  void InstrumentFunctions(ThreadSafeModule &TSM);

  approximableFunctions getApproximableFunctions(const ThreadSafeModule &M,
                                                 const SymbolFlagsMap &symbols);

  void splitModule(ThreadSafeModule &TSM, JITDylib &JD,
                   approximableFunctions &approxFns);

  Expected<ThreadSafeModule>
  approximateModule(ThreadSafeModule TSM, StringRef functionName,
                    configurationPerTechniqueMap configuration);

  // Remove the resources associated to all symbols not belonging to this
  // specific combination from the dylib. In practice, we are clearing the dylib
  // and letting RAAS recompile the correct symbol as we don't have a sensible
  // way to move a single symbol from one RT to another
  llvm::Error removeUnusedConfigurationSymbols(std::string functionName);
  // targeting a similar approach from CompileOnDemand
  StringMap<PerFunctionResources> FunctionNameToResourcesMap;
  // shamelessly stolen from CODLayer
  struct PerDylibResources {
  public:
    PerDylibResources(JITDylib &ApproxD,
                      std::unique_ptr<IndirectStubsManager> ISMgr)
        : ApproxD(ApproxD), ISMgr(std::move(ISMgr)) {}
    JITDylib &getApproxDylib() { return ApproxD; }
    IndirectStubsManager &getISManager() { return *ISMgr; }

  private:
    JITDylib &ApproxD;
    std::unique_ptr<IndirectStubsManager> ISMgr;
  };
  using PerDylibResourcesMap = std::map<const JITDylib *, PerDylibResources>;

  Expected<PerFunctionResources &> getFunctionResources(std::string function);
  PerDylibResources &getPerDylibResources(JITDylib &TargetD);

  mutable std::mutex ApproxLayerMutex;

  EvaluationSystem &evaluationSystem;
  SymbolLinkagePromoter PromoteSymbols;
  IRLayer &baseLayer;
  IndirectStubsManagerBuilder BuildIndirectStubsManager;
  PerDylibResourcesMap DylibResources;
  LazyCallThroughManager &LCTMgr;

  /* set to true if we want to run without applying transformations.
   * Use only for overhead measure
   */
  bool ignoreApproximations;

  const DataLayout &DL;
};

// Aliases that point to a declaration are not acceptable in IR. If we split a
// module, make sure the aliases point to a usable stub
static void keepAliasesAlive(Module &originalModule,
                             std::string approximableFn);

/// Render approximableFunctions.
raw_ostream &operator<<(raw_ostream &OS,
                        const ApproxLayer::approximableFunctions &approxFns);
