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
              JITTargetAddress jitAddress, EvaluationSystem &evaluator,
              LazyCallThroughManager &LCTMgr,
              IndirectStubsManagerBuilder BuildIndirectStubsManager);
  ApproxLayer(ExecutionSession &ES, DataLayout &DL, IRLayer &baseLayer,
              EvaluationSystem &evaluator, LazyCallThroughManager &LCTMgr,
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

    const JITSymbolFlags getFlags() { return SymbolFlags; }

  private:
    JITDylib &Dylib;
    ThreadSafeModule TSM;
    JITSymbolFlags SymbolFlags;
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

  EvaluationSystem &evaluator;
  SymbolLinkagePromoter PromoteSymbols;
  IRLayer &baseLayer;
  IndirectStubsManagerBuilder BuildIndirectStubsManager;
  PerDylibResourcesMap DylibResources;
  LazyCallThroughManager &LCTMgr;

  // set to true if we want to run without applying transformations
  // use only for overhead measure
  bool ignoreApproximations;

  const DataLayout &DL;
};

/// Render approximableFunctions.
raw_ostream &operator<<(raw_ostream &OS,
                        const ApproxLayer::approximableFunctions &approxFns);
