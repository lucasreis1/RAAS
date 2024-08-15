#include "Passes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"
#include <unordered_set>

using namespace llvm;

static std::unordered_set<std::string> replaceableFunctions = {
    "sgemm_", "sgemm", "cblas_sgemm"};

static bool isFunctionCallReplaceable(const CallInst *CI) {
  auto calledF = CI->getCalledFunction();
  if (!calledF)
    return false;
  // skip non-approximable functions
  if (!replaceableFunctions.count(calledF->getName().str()))
    return false;

  // check return type
  if (calledF->getReturnType()->getTypeID() != Type::VoidTyID)
    return false;
  // check number of arguments an if they are variable
  if (calledF->isVarArg() || calledF->arg_size() != 13)
    return false;

  // check if all arguments are pointers (the best we can do, I guess).
  // TODO: look into the alloca operations for all pointers to know the type for
  // each one
  for (auto &arg : calledF->args())
    if (arg.getType()->getTypeID() != Type::PointerTyID)
      return false;

  return true;
}

// param to function
static const std::vector<std::string> paramToApproxFn = {
    "approx_gemm_f16f16f32", "approx_gemm_bf16bf16f32", "approx_gemm_hgemm",
    "approx_gemm_s16s16s32", "approx_gemm_s8u8s32"};

static Type *findTypeForPointer(Value *Ptr) {
  if (auto alloca = dyn_cast<AllocaInst>(Ptr))
    return alloca->getAllocatedType();

  llvm_unreachable(
      "are we dealing with anything other than allocas? change this if so!\n");
  return Type::getVoidTy(Ptr->getContext());
}

static bool tryToOptimize(CallInst *CI, unsigned param) {
  if (!param)
    return false;

  unsigned idx = param - 1;

  auto parentModule = CI->getParent()->getParent()->getParent();

  auto &context = parentModule->getContext();
  auto voidTy = Type::getVoidTy(context);
  auto int8Ty = Type::getInt8Ty(context);
  auto int32Ty = Type::getInt32Ty(context);
  auto floatTy = Type::getFloatTy(context);
  auto ptrTy = PointerType::get(context, 0);
  // Function type for the approx GEMM functions
  FunctionType *FuncType = FunctionType::get(
      voidTy,
      {int8Ty, int8Ty, int32Ty, int32Ty, int32Ty, floatTy, ptrTy, int32Ty,
       ptrTy, int32Ty, floatTy, ptrTy, int32Ty},
      false);

  auto approxFnName = paramToApproxFn.at(idx);
  auto approxFnPtr = parentModule->getOrInsertFunction(approxFnName, FuncType);

  IRBuilder<> Builder(CI);

  // Create areguments for the call
  auto fcParams = FuncType->params();
  SmallVector<Value *, 16> fcCallArgs;
  for (int i = 0; i < fcParams.size(); ++i) {
    auto originalArg = CI->getArgOperand(i);
    if (fcParams[i]->isPointerTy()) {
      fcCallArgs.push_back(originalArg);
      continue;
    }

    // we should not trigger this, as gemm calls on pytorch use a function that
    // only asks for pointers
    if (originalArg->getType()->getTypeID() != Type::PointerTyID)
      llvm_unreachable("Are you sure you're running this from pytorch?\n");

    auto ptrType = findTypeForPointer(originalArg);

    auto trueParam = Builder.CreateLoad(ptrType, originalArg);
    fcCallArgs.push_back(trueParam);
  }

  auto approxCall = Builder.CreateCall(approxFnPtr, fcCallArgs);
  CI->replaceAllUsesWith(approxCall);
  CI->eraseFromParent();

  return true;
}

bool GEMMApproximation::isApproximable(const Function &F) {
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      auto calledF = CI->getCalledFunction();

      if (isFunctionCallReplaceable(CI))
        return true;
    }
  }
  return false;
}

unsigned GEMMApproximation::searchApproximableCalls(const Function &F) {
  unsigned count = 0;
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      auto calledF = CI->getCalledFunction();
      if (isFunctionCallReplaceable(CI))
        count++;
    }
  }
  return count;
}

void GEMMApproximation::printApproximationOpportunities(const Function &F) {
  unsigned count = 0;
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      auto calledF = CI->getCalledFunction();
      if (!isFunctionCallReplaceable(CI))
        continue;
      if (DILocation *Loc = CI->getDebugLoc()) {
        std::string fileName = Loc->getFilename().str();
        fprintf(stderr, "%s @ %s:%d\n", calledF->getName().str().c_str(),
                fileName.c_str(), Loc->getLine());
      }
    }
  }
}

PreservedAnalyses GEMMApproximation::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  auto ConfigOrErr =
      AM.getResult<ApproximationAnalysis>(F).getConfiguration(this->T);

  if (auto Err = ConfigOrErr.takeError()) {
    consumeError(std::move(Err));
    return PreservedAnalyses::all();
  }

  auto configuration = *ConfigOrErr;

  std::vector<CallInst *> toReplace;

  bool modified = false;

  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      if (isFunctionCallReplaceable(CI))
        toReplace.push_back(CI);
    }
  }

  if (toReplace.size() != configuration.size())
    llvm_unreachable("List of replaceable calls not in accordance to number of "
                     "opportunities for this function ");

  // iterate on all instructions that will be replaced
  for (size_t i = 0; i < toReplace.size(); ++i)
    modified |= tryToOptimize(toReplace[i], configuration[i]);

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserveSet<CFGAnalyses>();

  return modified ? PA : PreservedAnalyses::all();
}

/* New PM Registration */
llvm::PassPluginLibraryInfo getGEMMApproximationPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "GEMMAprpoximation", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerVectorizerStartEPCallback(
                [](llvm::FunctionPassManager &PM, OptimizationLevel Level) {
                  PM.addPass(GEMMApproximation());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "gemmapprox") {
                    PM.addPass(GEMMApproximation());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getGEMMApproximationPluginInfo();
}
