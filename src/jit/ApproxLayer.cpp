#include "ApproxLayer.h"
#include "Core.h"
#include "passes/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/LazyReexports.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ExtractGV.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Scalar/Reg2Mem.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <regex>
#include <utility>

#define DEBUG_TYPE "raas"

using namespace llvm;
using namespace orc;

static llvm::ExitOnError exitOnErr;

class ApproxMaterializationUnit : public MaterializationUnit {
public:
  ApproxMaterializationUnit(
      ApproxLayer &L,
      std::pair<std::string, configurationPerTechniqueMap> nameConfigPair)
      : MaterializationUnit(
            L.getInterface(nameConfigPair.first, nameConfigPair.second)),
        L(L), nameConfigPair(nameConfigPair) {}

  StringRef getName() const override {
    return "<ApproximationMaterializationUnit>";
  }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    L.emitApprox(std::move(R), nameConfigPair);
  }

private:
  void discard(const llvm::orc::JITDylib &JD,
               const llvm::orc::SymbolStringPtr &sym) override {
    LLVM_DEBUG(dbgs() << "[RAAS] Discarding symbol " << sym << " (noop)\n";);
  }

  ApproxLayer &L;
  std::pair<std::string, configurationPerTechniqueMap> nameConfigPair;
};

// tells the requested dylib that the symbol related to a MU is called
// functionName + combination, so a specific lookup with that name triggers
// approximation for that function
MaterializationUnit::Interface
ApproxLayer::getInterface(StringRef FunctionName,
                          configurationPerTechniqueMap configuration) {
  MangleAndInterner Mangle(getExecutionSession(), DL);
  SymbolFlagsMap Symbols;

  auto combination =
      EvaluationSystem::getCombinationFromConfiguration(configuration);
  Symbols[Mangle(FunctionName.str() + combination)] =
      exitOnErr(getFunctionResources(FunctionName.str())).getFlags();

  return MaterializationUnit::Interface(std::move(Symbols), nullptr);
}

// our combination is always a set [number]/[rate];...
// [number] - approx. technique enum
// [rate] - describes the level of approximation, this is what we need to count
bool isPreciseCombination(std::string combination) {
  const std::regex regex("([0-9]+/0;?)+");

  return std::regex_match(combination, regex);
}

bool isPreciseCombination(configurationPerTechniqueMap configuration) {
  for (auto &II : configuration) {
    for (auto el : II.second)
      if (el != 0)
        return false;
  }
  return true;
}

Expected<ExecutorSymbolDef>
ApproxLayer::lookup(JITDylib &JD, SymbolStringPtr MangledName,
                    JITDylibLookupFlags LookupFlags) {
  return getExecutionSession().lookup({{&JD, LookupFlags}}, MangledName);
}

ApproxLayer::ApproxLayer(ExecutionSession &ES, DataLayout &DL,
                         IRLayer &baseLayer, JITTargetAddress jitAddress,
                         EvaluationSystem &evaluationSystem,
                         LazyCallThroughManager &LCTMgr,
                         IndirectStubsManagerBuilder BuildIndirectStubsManager)
    : IRLayer(ES, baseLayer.getManglingOptions()), jitAddress(jitAddress),
      evaluationSystem(evaluationSystem), baseLayer(baseLayer),
      BuildIndirectStubsManager(std::move(BuildIndirectStubsManager)),
      LCTMgr(LCTMgr), DL(DL){};

ApproxLayer::ApproxLayer(ExecutionSession &ES, DataLayout &DL,
                         IRLayer &baseLayer, EvaluationSystem &evaluationSystem,
                         LazyCallThroughManager &LCTMgr,
                         IndirectStubsManagerBuilder BuildIndirectStubsManager,
                         bool ignoreApprox)
    : IRLayer(ES, baseLayer.getManglingOptions()),
      evaluationSystem(evaluationSystem), baseLayer(baseLayer),
      BuildIndirectStubsManager(std::move(BuildIndirectStubsManager)),
      LCTMgr(LCTMgr), ignoreApproximations(ignoreApprox), DL(DL) {}

void ApproxLayer::emitApprox(
    std::unique_ptr<MaterializationResponsibility> R,
    std::pair<std::string, configurationPerTechniqueMap> nameConfigPair) {

  auto &FnName = nameConfigPair.first;
  auto &configuration = nameConfigPair.second;

  LLVM_DEBUG(
      auto combination =
          EvaluationSystem::getCombinationFromConfiguration(configuration);
      if (combination == "") {
        dbgs() << "[RAAS] emitting precise version of function " << FnName
               << "\n";
      } else {
        dbgs() << "[RAAS] emitting approx version of function " << FnName
               << " with combination " << combination << "\n";
      });

  auto resourcesOrErr = getFunctionResources(FnName);

  if (auto Err = resourcesOrErr.takeError()) {
    getExecutionSession().reportError(std::move(Err));
    llvm_unreachable("unable to find resources for function");
  }

  auto &resources = *resourcesOrErr;

  // first time emitting this function, we need to tell the evaluation system
  // that it exists
  if (not evaluationSystem.findFunction(FnName)) {
    LLVM_DEBUG(dbgs() << "[RAAS] adding function " << FnName
                      << " to evaluation system!\n";);
    evaluationSystem.addFunction(*resources.getFunction());
  }

  auto clonedModule = cloneToNewContext(resources.getModule());

  if (!isPreciseCombination(configuration)) {
    auto approxModuleOrErr =
        approximateModule(std::move(clonedModule), FnName, configuration);

    if (auto Err = approxModuleOrErr.takeError()) {
      getExecutionSession().reportError(std::move(Err));
      llvm_unreachable("couldn't approximate module");
    }
    clonedModule = std::move(approxModuleOrErr.get());
  }

  baseLayer.emit(std::move(R), std::move(clonedModule));
}

void ApproxLayer::emit(std::unique_ptr<MaterializationResponsibility> R,
                       ThreadSafeModule TSM) {
  LLVM_DEBUG(dbgs() << "[RAAS] emitting module "
                    << TSM.getModuleUnlocked()->getName() << "\n");
  // promoted symbols in the module before splitting
  if (auto Err = TSM.withModuleDo([&](Module &M) -> llvm::Error {
        auto PromotedGlobals = PromoteSymbols(M);

        if (!PromotedGlobals.empty()) {
          SymbolFlagsMap SymbolFlags;

          // map symbols
          IRSymbolMapper::defaultSymbolMapper(
              PromotedGlobals, getExecutionSession(), *getManglingOptions(),
              SymbolFlags);

          return R->defineMaterializing(SymbolFlags);
        }
        return Error::success();
      })) {
    getExecutionSession().reportError(std::move(Err));
    R->failMaterialization();
  }

  auto approxFns = getApproximableFunctions(TSM, R->getSymbols());
  LLVM_DEBUG(dbgs() << "[RAAS] Removing from module "
                    << TSM.getModuleUnlocked()->getName()
                    << " approximable functions: " << approxFns << '\n';);

  splitModule(TSM, R->getTargetJITDylib(), approxFns);

  auto &ApprxDylib =
      getPerDylibResources(R->getTargetJITDylib()).getApproxDylib();

  MangleAndInterner Mangle(getExecutionSession(),
                           TSM.getModuleUnlocked()->getDataLayout());

  SymbolAliasMap toReexportSymbols;

  for (auto fnName : approxFns) {
    auto MangledName = Mangle(fnName);
    // make approx materialization unit with empty config (precise version of
    // function)
    if (auto Err =
            ApprxDylib.define(std::make_unique<ApproxMaterializationUnit>(
                *this,
                std::make_pair(fnName, configurationPerTechniqueMap())))) {
      R->failMaterialization();
      getExecutionSession().reportError(std::move(Err));
    }

    auto Fn = exitOnErr(getFunctionResources(fnName)).getFunction();
    toReexportSymbols[MangledName] =
        SymbolAliasMapEntry(MangledName, JITSymbolFlags::fromGlobalValue(*Fn));
  }

  // replace symbols with a lazy callthrough
  if (auto Err = R->replace(lazyReexports(
          LCTMgr, getPerDylibResources(R->getTargetJITDylib()).getISManager(),
          ApprxDylib, toReexportSymbols))) {
    getExecutionSession().reportError(std::move(Err));
    R->failMaterialization();
  }

  return baseLayer.emit(std::move(R), std::move(TSM));
}

llvm::Error
ApproxLayer::addApproximateVersion(std::string functionName,
                                   configurationPerTechniqueMap configuration) {
  auto resourcesOrErr = getFunctionResources(functionName);
  if (auto Err = resourcesOrErr.takeError()) {
    return Err;
  }
  auto &resources = *resourcesOrErr;

  std::string combination =
      EvaluationSystem::getCombinationFromConfiguration(configuration);
  combination = isPreciseCombination(combination) ? "" : combination;

  MangleAndInterner Mangle(getExecutionSession(), DL);

  auto mangledSymbName = Mangle(functionName + combination);
  auto &approxJD = getPerDylibResources(resources.getDylib()).getApproxDylib();

  auto RT = resources.getRT();

  if (RT == nullptr) {
    RT = approxJD.createResourceTracker();
    exitOnErr(resources.addRT(RT));
  }

  auto &ISMgr = getPerDylibResources(resources.getDylib()).getISManager();
  ExecutorAddr functionAddress;
  LLVM_DEBUG(dbgs() << "[RAAS] Looking up symbol " << mangledSymbName << " in "
                    << approxJD.getName() << "\n";);
  auto SymbOrErr = this->lookup(approxJD, mangledSymbName,
                                JITDylibLookupFlags::MatchAllSymbols);
  if (auto Err = SymbOrErr.takeError()) {
    LLVM_DEBUG(dbgs() << "[RAAS] Symbol " << mangledSymbName << " not found in "
                      << approxJD.getName()
                      << ". Adding Materialization Unit with combination "
                      << combination << '\n');
    // no problem, we need to compile the symbol
    consumeError(std::move(Err));
    // not yet compiled, add a MU to materialize when looked up
    if (auto Err = approxJD.define(
            std::make_unique<ApproxMaterializationUnit>(
                *this, std::make_pair(functionName, configuration)),
            RT)) {
      return Err;
    }

    // trigger lookup again to force materialization of the symbol
    exitOnErr(this->lookup(approxJD, mangledSymbName,
                           JITDylibLookupFlags::MatchAllSymbols));

    // create another trampoline to which the new pointer will jump to, to
    // materialize the new version of the function
    auto CallThroughTrampoline = LCTMgr.getCallThroughTrampoline(
        approxJD, mangledSymbName,
        [&ISManager = ISMgr,
         StubSymb = Mangle(functionName)](ExecutorAddr ResolvedAddr) -> Error {
          return ISManager.updatePointer(*StubSymb, ResolvedAddr);
        });

    if (!CallThroughTrampoline)
      return CallThroughTrampoline.takeError();

    functionAddress = *CallThroughTrampoline;
  } else
    functionAddress = SymbOrErr->getAddress();

  if (auto Err = ISMgr.updatePointer(functionName, functionAddress))
    return Err;

  return Error::success();
}

std::set<std::string>
ApproxLayer::getApproximableFunctions(const ThreadSafeModule &TSM,
                                      const SymbolFlagsMap &symbols) {
  std::set<std::string> approxFns;
  MangleAndInterner Mangle(getExecutionSession(),
                           TSM.getModuleUnlocked()->getDataLayout());
  TSM.withModuleDo([&](Module &M) {
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      // skip main function
      if (F.getName() == "main")
        continue;

      // only add symbols that are listed from this MU and are defined as
      // approximable
      if (symbols.count(Mangle(F.getName())) and
          evaluationSystem.isApproximable(F))
        approxFns.insert(F.getName().data());
    }
  });
  return approxFns;
}

void keepAliasesAlive(Module &M, std::string approximableFn) {
  SmallVector<GlobalAlias *, 4> movableAliases;
  auto Fn = M.getFunction(approximableFn);

  assert(Fn && "Function does not exist in module!");
  for (auto &GA : M.aliases()) {
    if (GA.getAliasee()->stripPointerCasts() == Fn)
      movableAliases.push_back(&GA);
  }

  if (not movableAliases.empty()) {
    // create a function stub for the aliases that must call the approx Fn
    Function *Stub = Function::Create(Fn->getFunctionType(),
                                      GlobalValue::LinkageTypes::PrivateLinkage,
                                      Fn->getName() + "_stub", &M);

    Stub->copyAttributesFrom(Fn);
    auto BB = BasicBlock::Create(M.getContext(), "entry", Stub);
    IRBuilder<> Builder(BB);

    SmallVector<Value *, 8> Args;
    for (auto &Arg : Stub->args())
      Args.push_back(&Arg);

    auto *CI = Builder.CreateCall(Fn->getFunctionType(), Fn, Args);
    CI->setCallingConv(Fn->getCallingConv());

    Builder.CreateRet(Stub->getReturnType()->isVoidTy() ? nullptr : CI);

    for (auto GA : movableAliases)
      GA->setAliasee(Stub);
  }
}

// split module, returns module that contains all functions from the original
void ApproxLayer::splitModule(ThreadSafeModule &TSM, JITDylib &JD,
                              approximableFunctions &approxFns) {
  std::vector<GlobalValue *> toDelete;

  for (auto function : approxFns) {
    // this function was already defined in another MU. Do not add it again
    assert(FunctionNameToResourcesMap.find(function) ==
               FunctionNameToResourcesMap.end() &&
           "function already defined from another MU!");

    keepAliasesAlive(*TSM.getModuleUnlocked(), function);
    auto clonedModule = cloneToNewContext(TSM, [&](const GlobalValue &GV) {
      if (GV.getName() == function) {
        toDelete.push_back(const_cast<GlobalValue *>(&GV));
        return true;
      }
      return false;
    });

    // run reg2mem pass to avoid complications when doing CFG hacking for
    // approximation
    FunctionAnalysisManager FAM;
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return LoopAnalysis(); });
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });

    FunctionPassManager FPM;
    FPM.addPass(RegToMemPass());

    clonedModule.withModuleDo([&](Module &M) {
      for (auto &F : M) {
        if (!F.isDeclaration())
          FPM.run(F, FAM);
      }

      M.setModuleIdentifier(M.getModuleIdentifier() + function);
    });

    Function *Fn = clonedModule.getModuleUnlocked()->getFunction(function);
    assert(Fn && "Cloned function must exist in the cloned module!\n");

    FunctionNameToResourcesMap.insert(std::make_pair(
        function, PerFunctionResources(JD, std::move(clonedModule),
                                       JITSymbolFlags::fromGlobalValue(*Fn))));
  }

  TSM.withModuleDo([&](Module &M) {
    // remove functions from the original module
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    ModulePassManager PM;
    PM.addPass(ExtractGVPass(toDelete));
    PM.addPass(StripDeadPrototypesPass());
    PM.addPass(StripDeadDebugInfoPass());
    PM.addPass(StripDeadPrototypesPass());
    PM.addPass(GlobalDCEPass());
    PM.addPass(VerifierPass());
    PM.run(M, MAM);
  });
}

// quick hack to reset configurations whe running --no-approx
static configurationPerTechniqueMap
resetConfiguration(configurationPerTechniqueMap configuration) {
  auto copyConfig = configuration;
  for (auto &II : copyConfig) {
    for (auto &el : II.second) {
      el = 0;
    }
  }
  return copyConfig;
}

Expected<ThreadSafeModule>
ApproxLayer::approximateModule(ThreadSafeModule TSM, StringRef functionName,
                               configurationPerTechniqueMap configuration) {
  if (auto Err = TSM.withModuleDo([&](Module &M) -> Error {
        LoopAnalysisManager LAM;
        FunctionAnalysisManager FAM;
        CGSCCAnalysisManager CGAM;
        ModuleAnalysisManager MAM;

        // reset our configuration if not approximating
        if (ignoreApproximations)
          FAM.registerPass([&] {
            return ApproximationAnalysis(std::make_pair(
                functionName, resetConfiguration(configuration)));
          });
        else
          FAM.registerPass([&] {
            return ApproximationAnalysis(
                std::make_pair(functionName, configuration));
          });

        PassBuilder PB;
        PB.registerModuleAnalyses(MAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        ModulePassManager MPM;
        passlist::addPassesToPM(MPM);
        MPM.addPass(VerifierPass());

        MPM.run(M, MAM);

        auto Function = M.getFunction(functionName);
        if (Function == nullptr)
          return make_error<StringError>("Module does not contain function.",
                                         inconvertibleErrorCode());
        std::string combination =
            EvaluationSystem::getCombinationFromConfiguration(configuration);
        combination = isPreciseCombination(combination) ? "" : combination;
        // change name of function so we get the appropriate symbol
        auto originalName = Function->getName();
        Function->setName(originalName + combination);
        return Error::success();
      }))
    return Err;

  return std::move(TSM);
}

Error ApproxLayer::updateApproximations() {
  auto toUpdateMap = evaluationSystem.updateSuggestedConfigurations();

  // iterate over the map, add a (possibly) new approximate version to each
  // function changed by the evaluation system
  for (auto &II : toUpdateMap) {
    auto Function = II.first().str();
    auto &config = II.second;
    LLVM_DEBUG(
        dbgs() << "[RAAS] calling addApproximateVersion for function "
               << Function << " with combination "
               << EvaluationSystem::getCombinationFromConfiguration(config)
               << '\n');
    // if we are at optimal combination for this function, remove the other
    // configurations. We do this before adding the approximate version because
    // we will need to recompile the symbol after deleting the tempRT
    if (evaluationSystem.hasFoundOptimalConfiguration(Function)) {
      LLVM_DEBUG(
          dbgs() << "[RAAS] Removing all combinations but "
                 << EvaluationSystem::getCombinationFromConfiguration(config)
                 << " from function " << Function << '\n');
      if (auto Err = removeUnusedConfigurationSymbols(Function))
        return Err;
    }

    if (auto Err = this->addApproximateVersion(Function, config))
      return Err;
  }
  return Error::success();
}

// targeting a similar approach from CompileOnDemand
ApproxLayer::PerDylibResources &
ApproxLayer::getPerDylibResources(JITDylib &TargetD) {
  auto I = DylibResources.find(&TargetD);

  if (I == DylibResources.end()) {
    auto &ApproxD =
        getExecutionSession().createBareJITDylib(TargetD.getName() + ".approx");
    JITDylibSearchOrder NewLinkOrder;

    TargetD.withLinkOrderDo([&](const JITDylibSearchOrder &TargetLinkOrder) {
      NewLinkOrder = TargetLinkOrder;
    });

    // ApproxD.addToLinkOrder(TargetD,
    //                        JITDylibLookupFlags::MatchExportedSymbolsOnly);

    assert(!NewLinkOrder.empty() && NewLinkOrder.front().first == &TargetD &&
           NewLinkOrder.front().second ==
               JITDylibLookupFlags::MatchAllSymbols &&
           "TargetD must be at the front of its own search order and match "
           "non-exported symbol");
    // NewLinkOrder.insert(NewLinkOrder.begin(),
    //                     {&ApproxD, JITDylibLookupFlags::MatchAllSymbols});
    ApproxD.setLinkOrder(NewLinkOrder, false);
    // TargetD.setLinkOrder(std::move(NewLinkOrder), false);
    //
    PerDylibResources PDR(ApproxD, BuildIndirectStubsManager());

    I = DylibResources.insert(std::make_pair(&TargetD, std::move(PDR))).first;
  }
  return I->second;
}

Expected<ApproxLayer::PerFunctionResources &>
ApproxLayer::getFunctionResources(const std::string FunctionName) {
  auto I = FunctionNameToResourcesMap.find(FunctionName);

  if (I == FunctionNameToResourcesMap.end())
    return make_error<StringError>("Unable to find resources for this function",
                                   inconvertibleErrorCode());

  return I->second;
}

llvm::Error
ApproxLayer::removeUnusedConfigurationSymbols(std::string functionName) {
  auto resources = getFunctionResources(functionName);
  if (auto Err = resources.takeError()) {
    return Err;
  }

  auto &perDyLibR = getPerDylibResources(resources->getDylib());
  MangleAndInterner Mangle(getExecutionSession(), DL);

  auto RT = resources->getRT();
  if (RT == nullptr)
    return make_error<StringError>("Missing resource tracker. Are you sure we "
                                   "have dangling symbols for this function?",
                                   inconvertibleErrorCode());

  // remove the temp resource tracker
  if (auto Err = RT->remove())
    return Err;

  exitOnErr(resources->removeRT());
  return Error::success();
}

/// Render approximableFunctions
raw_ostream &operator<<(raw_ostream &OS,
                        const ApproxLayer::approximableFunctions &approxFns) {
  OS << "[ ";

  if (!approxFns.empty()) {
    OS << approxFns.begin()->data();

    for (auto &fn : llvm::drop_begin(approxFns))
      OS << ", " << fn.data();
  }

  OS << " ]";
  return OS;
}
