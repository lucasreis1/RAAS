#include "JIT.h"
#include "SimpleEvaluator.h"
#include "passes/Passes.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

void initializeTarget() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
}

extern std::unique_ptr<orc::ApproxJIT> J;
// symbol that will be looked up by any instrumented function
#ifdef EMBEDDING
#ifdef __cplusplus
extern "C"
#endif
    void *
    $jump_to_jit(const char *fnName) {
  LLVM_DEBUG(dbgs() << "[RAAS] Jumping to JIT from function " << fnName
                    << '\n';);
  //  JIT not yet initialized, just leave
  if (not J) {
    return nullptr;
  }
  auto symOrErr = J->lookup(fnName, JITDylibLookupFlags::MatchAllSymbols);
  if (auto Err = symOrErr.takeError()) {
    // LLVM_DEBUG(dbgs() << "[RAAS] Symbol " << fnName << " not found!\n";);
    consumeError(std::move(Err));
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "[RAAS] Symbol " << fnName << " found!\n";);
  return (void *)symOrErr->getAddress().getValue();
}
#endif

#ifdef __cplusplus
extern "C"
#endif
    void
    $raas_evaluation() {
  if (not J)
    llvm_unreachable("JIT not initialied during reevaluation call!");

  J->approxReevaluation();
}

namespace llvm {
namespace orc {
// standard optimization
Expected<ThreadSafeModule> static optimizeModule(
    ThreadSafeModule TSM, const MaterializationResponsibility &R) {
  if (auto Err = TSM.withModuleDo([](Module &M) -> Error {
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

        ModulePassManager MPM;
        if (auto Err = PB.parsePassPipeline(MPM, "default<O2>"))
          return Err;
        MPM.run(M, MAM);

        return Error::success();
      }))
    return Err;

  return std::move(TSM);
}

// IPO optimization (add before splitting module for approximation in the
// APLayer)
Expected<ThreadSafeModule> static IPOOptimizeModule(
    ThreadSafeModule TSM, const MaterializationResponsibility &R) {
  if (auto Err = TSM.withModuleDo([](Module &M) -> Error {
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

        ModulePassManager MPM;
        if (auto Err = PB.parsePassPipeline(MPM, IPO_PIPELINE))
          return Err;
        MPM.run(M, MAM);
        return Error::success();
      }))
    return Err;

  return std::move(TSM);
}

ApproxJIT::ApproxJIT(std::unique_ptr<ExecutionSession> ES,
                     std::unique_ptr<EPCIndirectionUtils> EPCIU,
                     JITTargetMachineBuilder JTMB, DataLayout DL,
                     CustomDemangler demangler, ThreadSafeModule evalModule)
    : ES(std::move(ES)), EPCIU(std::move(EPCIU)), DL(std::move(DL)),
      Mangle(*this->ES, this->DL),
      ObjectLayer(*this->ES,
                  ExitOnErr(jitlink::InProcessMemoryManager::Create())),
      CompileLayer(*this->ES, ObjectLayer,
                   std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
      TransformLayer(*this->ES, CompileLayer, optimizeModule),
      APLayer(
          *this->ES, this->DL, TransformLayer, evaluator,
          this->EPCIU->getLazyCallThroughManager(),
          [this] { return this->EPCIU->createIndirectStubsManager(); }, false),
      ModuleTransformLayer(*this->ES, APLayer, IPOOptimizeModule),
      evaluator(std::make_unique<SimpleEvaluator>()),
      MainJD(cantFail(this->ES->createJITDylib("<main>"))),
      demangler(demangler) {

  // SupportJD.addGenerator(
  //     cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
  //         DL.getGlobalPrefix())));

#ifndef EMBEDDING
  auto MatU = absoluteSymbols(
      {{Mangle("$raas_evaluation"),
        {ExecutorAddr::fromPtr(&$raas_evaluation), JITSymbolFlags::Exported}}});
  cantFail(MainJD.define(MatU));
#endif

  MainJD.addGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          DL.getGlobalPrefix())));
  // MainJD.addToLinkOrder(SupportJD, JITDylibLookupFlags::MatchAllSymbols);

  // build pass list
  passlist::buildPasses();

  //  get symbols from the fastapprox library to our approx dylib
  if (auto DLSGOrErr = DynamicLibrarySearchGenerator::Load(
          "libfastapprox.so", DL.getGlobalPrefix()))
    MainJD.addGenerator(std::move(*DLSGOrErr));
  else {
    llvm_unreachable("libfastapprox.so not found. Ensure the library is in "
                     "your LD_LIBRARY_PATH!\n");
  }

  // Set our platform
  // TODO: check platform generically
  auto P = ExitOnErr(llvm::orc::ELFNixPlatform::Create(*this->ES, ObjectLayer,
                                                       MainJD, RUNTIME_PATH));
  this->ES->setPlatform(std::move(P));

  // add evaluation module to JIT-compiled symbols
  evalModule.getModuleUnlocked()->setDataLayout(getDataLayout());
  ExitOnErr(TransformLayer.add(MainJD.getDefaultResourceTracker(),
                               std::move(evalModule)));

  // debug if env variable is set
  if (std::getenv("DEBUG"))
    enableDebug();
}

ApproxJIT::~ApproxJIT() {
  if (auto Err = ES->endSession())
    ES->reportError(std::move(Err));
  if (auto Err = EPCIU->cleanup())
    ES->reportError(std::move(Err));
}

Expected<std::unique_ptr<ApproxJIT>>
ApproxJIT::Create(std::string evalModuleFile) {
  auto EPC = SelfExecutorProcessControl::Create(
      std::make_shared<orc::SymbolStringPool>());
  if (!EPC)
    return EPC.takeError();

  auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

  auto EPCIU = EPCIndirectionUtils::Create(ES->getExecutorProcessControl());
  if (!EPCIU)
    return EPCIU.takeError();

  (*EPCIU)->createLazyCallThroughManager(
      *ES, ExecutorAddr::fromPtr(&handleLazyCallThroughError));

  if (auto Err = setUpInProcessLCTMReentryViaEPCIU(**EPCIU))
    return std::move(Err);

  JITTargetMachineBuilder JTMB(
      ES->getExecutorProcessControl().getTargetTriple());

  auto DL = JTMB.getDefaultDataLayoutForTarget();

  if (!DL) {
    return DL.takeError();
  }

  auto evalModule = loadModule(evalModuleFile);
  if (!evalModule)
    return evalModule.takeError();
  auto demangler = CustomDemangler::Create(evalModule->getModuleUnlocked());

  if (!demangler)
    return demangler.takeError();

  return std::make_unique<ApproxJIT>(std::move(ES), std::move(*EPCIU),
                                     std::move(JTMB), std::move(*DL),
                                     *demangler, std::move(*evalModule));
}

Expected<std::unique_ptr<ApproxJIT>>
ApproxJIT::Create(std::string AppModule, std::string evalModuleFile) {
  auto JITOrErr = Create(evalModuleFile);
  if (!JITOrErr)
    return JITOrErr.takeError();

  auto JIT = std::move(*JITOrErr);

  if (auto Err = JIT->addModuleApproxFile(AppModule))
    return std::move(Err);

  return std::move(JIT);
}

Expected<ExecutorSymbolDef> ApproxJIT::lookup(StringRef Name,
                                              JITDylibLookupFlags Flags) {
  return ES->lookup({{&MainJD, Flags}}, Mangle(Name.str()));
}

// mapper that allows approximate symbols to be present in a Dylib, even if they
// are usually discarded. The only difference between this and standard mapping
// is that we change linkage type for  approximable symbols if they are
// linkonce_odr
static void allowApproximateSymbolsMapper(
    ArrayRef<GlobalValue *> GVs, ExecutionSession &ES,
    const IRSymbolMapper::ManglingOptions &MO, SymbolFlagsMap &SymbolFlags,
    IRSymbolMapper::SymbolNameToDefinitionMap *SymbolToDefinition) {
  if (GVs.empty())
    return;

  MangleAndInterner Mangle(ES, GVs[0]->getParent()->getDataLayout());
  for (auto *G : GVs) {
    assert(G && "GVs cannot contain null elements");
    // change linkage type of approximable functions so their symbols are not
    // discarded
    if (auto F = dyn_cast<Function>(G)) {
      if (passlist::isApproximable(*F) && F->hasLinkOnceODRLinkage())
        F->setLinkage(GlobalValue::WeakODRLinkage);
    }

    // Follow static linkage behaviour to decide which GVs get a named symbol
    if (!G->hasName() || G->isDeclaration() || G->hasLocalLinkage() ||
        G->hasAvailableExternallyLinkage() || G->hasAppendingLinkage() ||
        G->hasLinkOnceODRLinkage()) {
      continue;
    }

    if (G->isThreadLocal() && MO.EmulatedTLS) {
      auto *GV = cast<GlobalVariable>(G);

      auto Flags = JITSymbolFlags::fromGlobalValue(*GV);

      auto EmuTLSV = Mangle(("__emutls_v." + GV->getName()).str());
      SymbolFlags[EmuTLSV] = Flags;
      if (SymbolToDefinition)
        (*SymbolToDefinition)[EmuTLSV] = GV;

      // If this GV has a non-zero initializer we'll need to emit an
      // __emutls.t symbol too.
      if (GV->hasInitializer()) {
        const auto *InitVal = GV->getInitializer();

        // Skip zero-initializers.
        if (isa<ConstantAggregateZero>(InitVal))
          continue;
        const auto *InitIntValue = dyn_cast<ConstantInt>(InitVal);
        if (InitIntValue && InitIntValue->isZero())
          continue;

        auto EmuTLST = Mangle(("__emutls_t." + GV->getName()).str());
        SymbolFlags[EmuTLST] = Flags;
        if (SymbolToDefinition)
          (*SymbolToDefinition)[EmuTLST] = GV;
      }
      continue;
    }

    // Otherwise we just need a normal linker mangling.
    auto MangledName = Mangle(G->getName());
    SymbolFlags[MangledName] = JITSymbolFlags::fromGlobalValue(*G);
    if (G->getComdat() &&
        G->getComdat()->getSelectionKind() != Comdat::NoDeduplicate)
      SymbolFlags[MangledName] |= JITSymbolFlags::Weak;
    if (SymbolToDefinition)
      (*SymbolToDefinition)[MangledName] = G;
  }
}

Expected<ExecutorSymbolDef> ApproxJIT::lookupOrLoadSymbol(StringRef Name,
                                                          StringRef ModuleLoc) {
  LLVM_DEBUG(dbgs() << "[RAAS] looking up symbol " << Name << " in module @ "
                    << ModuleLoc << '\n');

  auto SymOrErr = this->lookup(Name, JITDylibLookupFlags::MatchAllSymbols);
  if (auto Err = SymOrErr.takeError()) {
    // Consume the error.
    consumeError(std::move(Err));

    LLVM_DEBUG(
        dbgs()
        << "[RAAS] Symbol " << Name
        << " not found. Loading module from disk to emmit required symbols.\n");
    SMDiagnostic error;
    ThreadSafeContext TSctx(std::make_unique<LLVMContext>());
    std::unique_ptr<Module> m =
        parseIRFile(ModuleLoc, error, *TSctx.getContext());

    if (m.get() == nullptr) {
      error.print("jit", errs());
      return make_error<StringError>("module file not found",
                                     inconvertibleErrorCode());
    }

    if (verifyModule(*m.get(), &errs())) {
      return make_error<StringError>("module invalid",
                                     inconvertibleErrorCode());
    }

    ThreadSafeModule TSM(std::move(m), TSctx);

    LLVM_DEBUG(dbgs() << "[RAAS] Module loaded. Adding it do Dylib.\n");
    ExitOnErr(
        APLayer.add(MainJD, std::move(TSM), allowApproximateSymbolsMapper));
    LLVM_DEBUG(dbgs() << "[RAAS] Searching for " << Name << " on "
                      << MainJD.getName() << " to trigger materialization\n");
    SymOrErr =
        ExitOnErr(this->lookup(Name, JITDylibLookupFlags::MatchAllSymbols));
  }

  return SymOrErr;
}

double measure_time() {
  double time;
  FILE *f = fopen(".jit_time.txt", "r");
  if (f == nullptr)
    llvm_unreachable("could not open time_sampling file!\n");

  fscanf(f, "%lf", &time);
  fclose(f);
  return time;
}

void ApproxJIT::approxReevaluation() {
  double time = measure_time();
  // this is the first loop, we will use it only to store precise output
  // values. We do not want time measures from this loop
  // to avoid measuring a cold cache
  if (this->numberOfLoops == 0) {
    using voidFnPtr = void (*)();
    // lookup storeOriginal fn from the evaluation file
    auto storeOriginalMangled = demangler.getFunction("storeOriginal");
    auto storeOrigSymb = ExitOnErr(this->lookup(storeOriginalMangled));
    voidFnPtr storeOrigFn = storeOrigSymb.getAddress().toPtr<voidFnPtr>();
    storeOrigFn();
  }
  // use these loops exclusively to accumulate a precise time average
  else if (isOnBaseLoops()) {
    evaluator.incrementPreciseTime(time / (BASE_ITERATIONS));
    fprintf(stdout, "thus ends base loop %d/%d with %lf time!\n",
            getLoopNumber(), BASE_ITERATIONS, time);
  } else if (isOnApproximableLoops()) { // those are the loops where we are
                                        // approximating
    using doubleFnPtr = double (*)();
    // lookup compare fn from the evaluation file
    auto compareMangled = demangler.getFunction("compare");
    auto compareSymb = ExitOnErr(this->lookup(compareMangled));
    // find the JIT symbol that points to our error calculation function
    doubleFnPtr compareFn = compareSymb.getAddress().toPtr<doubleFnPtr>();
    auto err = compareFn();
    // store the difference between outputs
    evaluator.updateQualityValues(err, time);
    fprintf(stdout,
            "thus ends iteration %d with %lf time (%lf speedup / %lf error)!\n",
            getLoopNumber(), time, evaluator.getLastSpeedup(),
            evaluator.getLastError());

    // at the end of a loop, call our evaluator heuristic to update the
    // suggested configuration
    ExitOnErr(APLayer.updateApproximations());
  }
  // reset all counters to true, so we can jump to the JIT when we se an
  // approximable function
  this->numberOfLoops++;
}

Expected<ThreadSafeModule> ApproxJIT::loadModule(std::string bitcodeFile) {
  ThreadSafeContext TSctx(std::make_unique<LLVMContext>());
  SMDiagnostic error;
  std::unique_ptr<Module> m =
      parseIRFile(bitcodeFile, error, *TSctx.getContext());

  if (m.get() == nullptr)
    return make_error<StringError>("Invalid file for module\n",
                                   inconvertibleErrorCode());

  if (verifyModule(*m.get(), &errs()))
    return make_error<StringError>("File does not produce valid IR module\n",
                                   inconvertibleErrorCode());

  auto TSM = ThreadSafeModule(std::move(m), TSctx);

  return TSM;
}

void enableDebug() {
  DebugFlag = true;
  const char *debugTypes[] = {"raas", "orc"};
  setCurrentDebugTypes(debugTypes, 1);
}

Error ApproxJIT::initializePlatform() {
  // run static initializers
  using llvm::orc::shared::SPSExecutorAddr;
  using llvm::orc::shared::SPSString;
  using SPSDLOpenSig = SPSExecutorAddr(SPSString, int32_t);
  enum dlopen_mode : int32_t {
    ORC_RT_RTLD_LAZY = 0x1,
    ORC_RT_RTLD_NOW = 0x2,
    ORC_RT_RTLD_LOCAL = 0x4,
    ORC_RT_RTLD_GLOBAl = 0x8
  };

  if (auto Sym = this->lookup("__orc_rt_jit_dlopen_wrapper")) {
    auto wrapperAddr = ExecutorAddr(Sym->getAddress());
    if (auto E = this->ES->callSPSWrapper<SPSDLOpenSig>(
            wrapperAddr, DSOHandles[&MainJD], MainJD.getName(),
            int32_t(ORC_RT_RTLD_LAZY)))
      return E;
    return Error::success();
  } else
    return Sym.takeError();
}

Error ApproxJIT::deinitializePlatform() {
  using llvm::orc::shared::SPSExecutorAddr;
  using SPSDLCloseSig = int32_t(SPSExecutorAddr);

  if (auto Sym = this->lookup("__orc_rt_jit_dlclose_wrapper")) {
    auto wrapperAddr = ExecutorAddr(Sym->getAddress());
    int32_t result = 0;
    auto E = this->ES->callSPSWrapper<SPSDLCloseSig>(wrapperAddr, result,
                                                     DSOHandles[&MainJD]);
    if (E)
      return E;
    else if (result)
      return make_error<StringError>("dlclose failed",
                                     inconvertibleErrorCode());
    DSOHandles.erase(&MainJD);
  } else
    return Sym.takeError();

  return Error::success();
}

Error ApproxJIT::addModuleFile(std::string moduleFile) {
  auto mod = ApproxJIT::loadModule(moduleFile);
  if (auto Err = mod.takeError())
    return std::move(Err);

  ExitOnErr(TransformLayer.add(MainJD, std::move(*mod)));
  LLVM_DEBUG(dbgs() << "[RAAS] Added module " << moduleFile << "\n";);
  return Error::success();
}

Error ApproxJIT::addModuleApproxFile(std::string moduleFile) {
  auto mod = ApproxJIT::loadModule(moduleFile);
  if (auto Err = mod.takeError())
    return std::move(Err);

  ExitOnErr(ModuleTransformLayer.add(MainJD, std::move(*mod),
                                     allowApproximateSymbolsMapper));
  LLVM_DEBUG(dbgs() << "[RAAS] Added approx module " << moduleFile << "\n";);
  return Error::success();
}

void ApproxJIT::setForbiddenApproxList(
    std::unique_ptr<ForbiddenApproximations> fapList) {
  evaluator.setForbiddenApproxList(std::move(fapList));
}

} // namespace orc
} // namespace llvm
