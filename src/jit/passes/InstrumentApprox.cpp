// This pass instruments approximable functions to make it feasible to modify
// them from the JIT
#include "Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <regex>

using namespace llvm;

class ApproxInstrumentation : public PassInfoMixin<ApproxInstrumentation> {
public:
  ApproxInstrumentation(bool approxOrNot = false) : willApprox(approxOrNot) {
    passlist::buildPasses();
  }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    // first time seeing an approximate fn in this module
    int approxFnCount = 0;
    bool firstInstrumentedFn = true;
    if (!willApprox)
      addGlobalValues(M);
    for (auto &F : M) {
      if (not F.isDeclaration() and not F.hasLocalLinkage() and
          not F.hasAvailableExternallyLinkage() and
          not F.getName().starts_with("raas_orig")) {
        // if approximating this module, instrument the funciton accordingly
        if (willApprox and passlist::isApproximable(F)) {
          if (firstInstrumentedFn) {
            firstInstrumentedFn = false;
            addGlobalValues(M);
          }
          approxFnCount++;
          instrumentFunctionApprox(F);
        } else if (not willApprox) // if not, all functions are instrumented
                                   // the same way
          instrumentFunctionApprox(F);
      }
    }

    errs() << "instrumented " << approxFnCount << " function(s) for module "
           << M.getName() << '\n';
    return PreservedAnalyses::all();
  }

  //  just jump to the JIT, no questions asked
  void instrumentFunction(Function &F) {
    Type *Int64 = Type::getInt64Ty(F.getContext());
    Type *Int1 = Type::getInt1Ty(F.getContext());
    Type *Int8Ptr = Type::getInt8PtrTy(F.getContext());

    std::vector<Instruction *> entry_Insts;

    for (auto &I : F.getEntryBlock()) {
      I.dropAllReferences();
      entry_Insts.push_back(&I);
    }

    // erase instructions for entry block
    for (auto I : entry_Insts) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }

    IRBuilder<> B(&F.getEntryBlock());

    Constant *functionNamePtr =
        B.CreateGlobalStringPtr(F.getName(), "functionName$" + F.getName());

    auto jump_to_jitFn = F.getParent()->getFunction("$jump_to_jit");
    if (jump_to_jitFn == nullptr)
      llvm_unreachable("did not find jump_to_jit() function in the module");

    Value *recompiledFnAddr = B.CreateCall(jump_to_jitFn, {functionNamePtr});

    Value *recompiledFn = B.CreateIntToPtr(recompiledFnAddr, F.getType());
    std::vector<Value *> callArgs;

    for (auto &A : F.args())
      callArgs.push_back(&A);

    CallInst *recompFuncCall =
        B.CreateCall(F.getFunctionType(), recompiledFn, callArgs);
    recompFuncCall->setTailCall(true);

    if (F.getReturnType()->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(recompFuncCall);

    EliminateUnreachableBlocks(F);
  }

  void instrumentFunctionApprox(Function &F) {
    // clone the function to store its original definition in a different symbol
    // name
    ValueToValueMapTy VM;
    auto clonedFn = CloneFunction(&F, VM);
    clonedFn->setName("raas_orig" + F.getName());

    Type *Int64 = Type::getInt64Ty(F.getContext());
    Type *Int1 = Type::getInt1Ty(F.getContext());
    Type *Int8Ptr = Type::getInt8PtrTy(F.getContext());

    std::vector<Instruction *> entry_Insts;

    for (auto &I : F.getEntryBlock()) {
      I.dropAllReferences();
      entry_Insts.push_back(&I);
    }

    // erase instructions for entry block
    for (auto I : entry_Insts) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }

    // if RAAS was initialized, jump_to_jit
    // else, jump to the original function symbol
    auto initBB = BasicBlock::Create(F.getContext(), "raas_init", &F);
    auto notInitBB =
        BasicBlock::Create(F.getContext(), "raas_not_init", &F, initBB);
    auto jumpToJitDefined =
        BasicBlock::Create(F.getContext(), "jumpToJitDefined", &F, notInitBB);

    IRBuilder<> B(&F.getEntryBlock());
    auto jump_to_jitFn = F.getParent()->getFunction("$jump_to_jit");
    if (jump_to_jitFn == nullptr)
      llvm_unreachable("did not find jump_to_jit() function in the module");

    // check if the pointer to the function is null. jump to the original function if it is.
    auto jitPtrIsNullCmp = B.CreateICmpEQ(
        jump_to_jitFn,
        ConstantPointerNull::get(PointerType::get(F.getContext(), 0)));

    B.CreateCondBr(jitPtrIsNullCmp, notInitBB, jumpToJitDefined);

    B.SetInsertPoint(jumpToJitDefined);
    Constant *functionNamePtr =
        B.CreateGlobalStringPtr(F.getName(), "functionName$" + F.getName());

    // Constant *IRModLoc = B.CreateGlobalStringPtr(bitcodeFileLoc, "irmodloc");

    Value *recompiledFnAddr = B.CreateCall(jump_to_jitFn, {functionNamePtr});
    auto recompiledFnFound =
        B.CreateICmpEQ(recompiledFnAddr, ConstantInt::get(Int64, 0));
    B.CreateCondBr(recompiledFnFound, notInitBB, initBB);
    // auto raas_is_initFn = F.getParent()->getFunction("$raas_is_initialized");
    // if (not raas_is_initFn)
    //   llvm_unreachable("$raas_is_initialized function not found in module");

    // auto raas_is_init = B.CreateCall(raas_is_initFn);
    // auto raasIsInitCmp =
    //     B.CreateICmpEQ(raas_is_init, ConstantInt::get(Int1, 1));
    // B.CreateCondBr(raasIsInitCmp, initBB, notInitBB);

    // call args from original fn
    std::vector<Value *> callArgs;

    for (auto &A : F.args())
      callArgs.push_back(&A);

    // if RAAS was not initialized
    B.SetInsertPoint(notInitBB);

    auto originalFn =
        F.getParent()->getFunction(("raas_orig" + F.getName()).str());

    CallInst *originalFnCall =
        B.CreateCall(F.getFunctionType(), originalFn, callArgs);

    originalFnCall->setTailCall(true);

    if (F.getReturnType()->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(originalFnCall);

    // if RAAS was initialized
    B.SetInsertPoint(initBB);
    // Constant *functionNamePtr =
    //     B.CreateGlobalStringPtr(F.getName(), "functionName$" + F.getName());

    // Constant *IRModLoc = B.CreateGlobalStringPtr(bitcodeFileLoc, "irmodloc");

    // auto jump_to_jitFn = F.getParent()->getFunction("$jump_to_jit");

    // Value *recompiledFnAddr =
    //     B.CreateCall(jump_to_jitFn, {functionNamePtr});
    Value *recompiledFn = B.CreateIntToPtr(recompiledFnAddr, F.getType());

    CallInst *recompFuncCall =
        B.CreateCall(F.getFunctionType(), recompiledFn, callArgs);
    recompFuncCall->setTailCall(true);

    if (F.getReturnType()->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(recompFuncCall);

    EliminateUnreachableBlocks(F);
  }

  // add global values required by the JIT
  void addGlobalValues(Module &M) {
    // create function -> $jump_to_jit("fn_name", "loc/to/bcfile.bc");
    // Function::Create(FunctionType::get(Int64, {Int8Ptr}, false),
    //                 GlobalValue::ExternalWeakLinkage, "$jump_to_jit", M);
    Function::Create(FunctionType::get(Type::getInt64Ty(M.getContext()),
                                       {PointerType::get(M.getContext(), 0)},
                                       false),
                     GlobalValue::ExternalWeakLinkage, "$jump_to_jit", M);
    // create link to global value that signals if RAAS was already initialized
    // Function::Create(FunctionType::get(Int1, {}, false),
    //                 GlobalValue::ExternalWeakLinkage, "$raas_is_initialized",
    //                 M);
  }

  static StringRef name() { return "instrappfn"; };

private:
  bool willApprox;
};

/* New PM Registration */
llvm::PassPluginLibraryInfo getInstrFnPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrApprFn", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager &PM, OptimizationLevel Level) {
                  PM.addPass(ApproxInstrumentation());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::ModulePassManager &MPM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  std::regex instrappRegex("instrappfn(<[\\w/.]+>)?");
                  if (std::regex_match(Name.str(), instrappRegex)) {
                    std::regex name_re("instrappfn<?([\\w/.]+)>?");
                    std::string approxOrNot =
                        std::regex_replace(Name.str(), name_re, "$1");
                    bool shouldApprox = approxOrNot == "true" ? true : false;
                    MPM.addPass(ApproxInstrumentation(shouldApprox));
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
