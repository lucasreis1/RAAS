#include "Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"
#include <unordered_set>

using namespace llvm;

// list of functions we can replace
const std::unordered_set<std::string> functionReplacementList = {
    "exp",  "exp2", "pow",    "sin",     "cos",  "tan",  "log", "log2",
    "erfc", "erf",  "lgamma", "digamma", "sinh", "cosh", "tanh"};

std::string approxVersion(int param) {
  std::string opString;
  switch (param) {
  case 1: {
    opString = "fast";
    break;
  }
  case 2: {
    opString = "faster";
    break;
  }
  }
  return opString;
}

enum FunctionOptions { POW, SINorCOS, TAN, OTHER };

// We want to replace both float and double implementations
// thus, remove the final f
static StringRef formatString(StringRef functionName) {
  if (functionName.endswith("f") && functionName != "erf")
    return functionName.drop_back();
  return functionName;
}

static FunctionOptions resolveOptions(StringRef name) {
  name = formatString(name);
  if (name == "pow")
    return POW;
  if (name == "sin" || name == "cos")
    return SINorCOS;
  if (name == "TAN")
    return TAN;
  return OTHER;
}

static void copyCallMetadata(CallInst *C1, CallInst *C2) {
  SmallVector<std::pair<unsigned, MDNode *>, 0> MD;
  C1->getAllMetadata(MD);
  for (auto mdit : MD) {
    C2->setMetadata(mdit.first, mdit.second);
  }
  C2->setCallingConv(C1->getCallingConv());

  SmallVector<AttributeSet, 16> NewArgAttrib;

  // copy attributes from the last parameter up until we exhaust the number of
  // parameters on C2. This is needed as C2 may have less parameters than the
  // original call
  int c2ArgSize = C2->getCalledFunction()->arg_size();
  int c1ArgSize = C1->getCalledFunction()->arg_size();

  auto c1Attrs = C1->getAttributes();

  for (int i = c1ArgSize - 1; i >= (c1ArgSize - c2ArgSize); i--) {
    NewArgAttrib.push_back(c1Attrs.getParamAttrs(i));
  }

  C2->setAttributes(AttributeList::get(C1->getContext(), c1Attrs.getFnAttrs(),
                                       c1Attrs.getRetAttrs(), NewArgAttrib));
}

class ReplaceableCall {
public:
  ReplaceableCall(CallInst *CI, unsigned param) : Call(CI), option(param) {
    opString = approxVersion(option);

    createArgumentsArray();

    requiresTwoCalls = populateReplacementRequirements();
  }

  void replaceCalls() {
    LLVMContext &ctx = Call->getContext();

    IRBuilder<> builder(ctx);
    // parent module
    Module *parent = Call->getParent()->getParent()->getParent();

    if (requiresTwoCalls) {
      // Create a basic block to branch to after the two calls
      auto bit = ++BasicBlock::iterator(Call);
      auto afterIfElse = Call->getParent()->splitBasicBlock(bit, "afterIfElse");

      // Create basic blocks required for ifelse comparison
      auto ifBB = BasicBlock::Create(ctx, "ifComp.then",
                                     afterIfElse->getParent(), afterIfElse);
      auto elseBB = BasicBlock::Create(ctx, "ifComp.else",
                                       afterIfElse->getParent(), afterIfElse);

      BasicBlock *callParent = Call->getParent();
      // Create the branch function and replace the one created automatically by
      // SplitBasicBlock()
      callParent->getTerminator()->eraseFromParent();
      BranchInst::Create(ifBB, elseBB, Comparison, callParent);
      // auto condJump = BranchInst::Create(ifBB, elseBB, Comparison,
      // callParent);
      //  Call->getParent()->getTerminator()->replaceAllUsesWith
      //   Create both new function prototypes
      //   no dyn_cast here because we are sure the Value is a Function and not
      //   a bitcast
      auto newFunction = cast<Function>(
          parent->getOrInsertFunction(opString, fcType).getCallee());
      auto secondNewFunction = cast<Function>(
          parent->getOrInsertFunction(secondOpString, secondFcType)
              .getCallee());

      // Insert the call and jump into both basic blocks
      builder.SetInsertPoint(ifBB);

      auto newCall = builder.CreateCall(newFunction, argsArray);
      builder.CreateBr(afterIfElse);
      builder.SetInsertPoint(elseBB);
      auto secondNewCall =
          builder.CreateCall(secondNewFunction, secondArgsArray);
      builder.CreateBr(afterIfElse);

      // Set calling metadata IF any
      copyCallMetadata(Call, newCall);
      copyCallMetadata(Call, secondNewCall);

      // Allocate a new pointer to receive result from both newCalls
      BasicBlock *Entry = &Call->getParent()->getParent()->front();

      builder.SetInsertPoint(&*Entry->getFirstInsertionPt());
      auto resultPtr = builder.CreateAlloca(Type::getFloatTy(ctx), nullptr,
                                            "resultFromApproxCall");

      // Store the result from calls into resultPtr
      builder.SetInsertPoint(newCall->getNextNode());
      builder.CreateStore(newCall, resultPtr);
      builder.SetInsertPoint(secondNewCall->getNextNode());
      builder.CreateStore(secondNewCall, resultPtr);

      // Load pointer into variable result
      builder.SetInsertPoint(afterIfElse, afterIfElse->getFirstInsertionPt());
      auto result = builder.CreateLoad(resultPtr->getAllocatedType(), resultPtr,
                                       "result");
      // if Call is a double, convert result from float to double
      if (Call->getType()->getTypeID() == Type::DoubleTyID) {
        // extended fp to double
        auto ext = builder.CreateFPExt(result, Type::getDoubleTy(ctx));
        Call->replaceAllUsesWith(ext);
      }
      // else, just replace uses of Call with result
      else {
        Call->replaceAllUsesWith(result);
      }
    } else {
      // Add the function prototype to the module
      // auto FunctionCallee = parent->getOrInsertFunction(opString, fcType);
      parent->getOrInsertFunction(opString, fcType);
      auto newFunction = cast<Function>(
          parent->getOrInsertFunction(opString, fcType).getCallee());
      builder.SetInsertPoint(Call);
      // Create the instruction call and place it before Call
      auto newCall = builder.CreateCall(newFunction, argsArray);
      // ensure conventions, attributes and metadata are equal
      copyCallMetadata(Call, newCall);
      // Replace the old instruction with the new call, including its uses. If
      // Call is doubleType, create FPEXtention instruciton to receive the
      // result of newCall
      if (Call->getType()->getTypeID() == Type::DoubleTyID) {
        auto ext = builder.CreateFPExt(newCall, Type::getDoubleTy(ctx));
        Call->replaceAllUsesWith(ext);
      } else {
        Call->replaceAllUsesWith(newCall);
      }
    }
    // remove Call from the BB
    Call->eraseFromParent();
  }

private:
  void createArgumentsArray() {
    IRBuilder<> builder(Call);

    auto FloatTy = Type::getFloatTy(Call->getContext());
    // Arguments must be converted to float (if they are double precision)
    for (Value *AI : Call->args()) {
      // if the argument is a single precision type, we are ok
      if (AI->getType()->getTypeID() == Type::FloatTyID)
        this->argsArray.push_back(AI);
      // if it is double, we must truncate it to fit a less precise value
      else
        this->argsArray.push_back(builder.CreateFPTrunc(AI, FloatTy));
    }
  }

  // populates the variables required for function call creation according to
  // the function that will be replaced
  bool populateReplacementRequirements() {

    LLVMContext &ctx = Call->getContext();
    IRBuilder<> builder(ctx);
    Function *currentFunction = Call->getCalledFunction();

    Type *FloatTy = Type::getFloatTy(ctx);

    bool requiresTwoCalls = false;
    StringRef functionName = formatString(currentFunction->getName());

    // the exp2() function is called pow2() in fastapprox
    if (functionName == "exp2")
      opString += "pow2";
    else
      opString += functionName;

    builder.SetInsertPoint(Call);

    switch (resolveOptions(functionName)) {
    case POW: {
      // use pow2 if first arguments is a constant of value 2
      if (auto arg0 = dyn_cast<ConstantFP>(Call->arg_begin())) {
        if (arg0->isExactlyValue(2.0)) {
          opString += '2';
          // Construct the FunctionType
          fcType = FunctionType::get(FloatTy, FloatTy, false);
          // Reduce the argument array
          this->argsArray.erase(argsArray.begin());
        } else
          fcType = FunctionType::get(FloatTy, {FloatTy, FloatTy}, false);
      } else {
        // Compare the first argument with 2.0 and branch there
        auto valueTwo = ConstantFP::get(FloatTy, 2.0);
        Comparison = builder.CreateFCmpOEQ(argsArray[0], valueTwo);

        secondOpString = opString;
        opString += '2';
        fcType = FunctionType::get(FloatTy, FloatTy, false);
        secondFcType = FunctionType::get(FloatTy, {FloatTy, FloatTy}, false);
        secondArgsArray = argsArray;
        argsArray.erase(argsArray.begin());
        requiresTwoCalls = true;
      }
      break;
    }
    // for sin, cos and tan, the fast/faster versions
    // should be used only on specific angle intervals
    case SINorCOS: { // [-Pi,Pi]
      if (auto arg0 = dyn_cast<ConstantFP>(Call->arg_begin())) {
        if (fabs(arg0->getValueAPF().convertToFloat()) > M_PI)
          opString += "full";
        fcType = FunctionType::get(FloatTy, FloatTy, false);
      } else {
        // build comparisons with Pi and -Pi
        auto valuePI = ConstantFP::get(FloatTy, M_PI);
        auto valueMinusPI = ConstantFP::get(FloatTy, -M_PI);
        auto firstComparison = builder.CreateFCmpOLE(argsArray[0], valuePI);
        auto secondComparison =
            builder.CreateFCmpOGE(argsArray[0], valueMinusPI);
        firstComparison->setName("ltPI");
        secondComparison->setName("gtMinusPI");
        // logical and comparison
        Comparison = builder.CreateICmpEQ(firstComparison, secondComparison);

        secondOpString = opString + "full";
        fcType = FunctionType::get(FloatTy, FloatTy, false);
        secondFcType = FunctionType::get(FloatTy, FloatTy, false);
        secondArgsArray = argsArray;
        requiresTwoCalls = true;
      }
      break;
    }
    case TAN: { // [-Pi/2, Pi/2]
      if (auto *arg0 = dyn_cast<ConstantFP>(Call->arg_begin())) {
        if (2 * fabs(arg0->getValueAPF().convertToFloat()) > M_PI)
          opString += "full";
        fcType = FunctionType::get(FloatTy, FloatTy, false);
      } else {
        auto valuePI = ConstantFP::get(FloatTy, M_PI / 2.0);
        auto valueMinusPI = ConstantFP::get(FloatTy, -M_PI / 2.0);
        auto firstComparison = builder.CreateFCmpOLE(argsArray[0], valuePI);
        auto secondComparison =
            builder.CreateFCmpOGE(argsArray[0], valueMinusPI);
        firstComparison->setName("lePI/2");
        secondComparison->setName("geMinusPI/2");
        // logical and comparison
        Comparison = builder.CreateICmpEQ(firstComparison, secondComparison);

        secondOpString = opString + "full";
        fcType = FunctionType::get(FloatTy, FloatTy, false);
        secondFcType = FunctionType::get(FloatTy, FloatTy, false);
        secondArgsArray = argsArray;
        requiresTwoCalls = true;
      }
      break;
    }
    case OTHER: {
      auto smvTypes = SmallVector<Type *, 4>(
          currentFunction->getFunctionType()->getNumParams(), FloatTy);
      fcType = FunctionType::get(FloatTy, smvTypes, false);
      break;
    }
    } // switch
    return requiresTwoCalls;
  }

  CallInst *Call;
  unsigned option;

  // first call
  std::string opString;
  FunctionType *fcType;

  // POSSIBLE second call
  std::string secondOpString;
  FunctionType *secondFcType;

  SmallVector<Value *, 4> argsArray, secondArgsArray;

  Value *Comparison;

  bool requiresTwoCalls;
};

static bool tryToOptimize(CallInst *Call, unsigned param) {
  if (param) {
    ReplaceableCall RC(Call, param);
    RC.replaceCalls();
    return true;
  } else
    return false;
}

bool FunctionApproximation::isApproximable(const Function &F) {
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *calledF = CI->getCalledFunction();

      if (calledF == nullptr)
        continue;

      StringRef functionName = formatString(calledF->getName());

      // there is a call to approximate in this function
      if (functionReplacementList.count(functionName.str()))
        return true;
    }
  }
  return false;
}

// iterate through a function and return the number of opportunities
unsigned FunctionApproximation::searchApproximableCalls(const Function &F) {
  unsigned count = 0;
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *calledF = CI->getCalledFunction();

      if (calledF == nullptr)
        continue;

      // count annother approximate call
      StringRef functionName = formatString(calledF->getName());
      if (functionReplacementList.count(functionName.str()))
        count++;
    }
  }
  return count;
}

void FunctionApproximation::printApproximationOpportunities(const Function &F) {
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *calledF = CI->getCalledFunction();

      if (calledF == nullptr)
        continue;

      StringRef functionName = formatString(calledF->getName());
      if (not functionReplacementList.count(functionName.str()))
        continue;

      if (DILocation *Loc = CI->getDebugLoc()) {
        std::string fileName = Loc->getFilename().str();
        fprintf(stderr, "%s @ %s:%d\n", calledF->getName().str().c_str(),
                fileName.c_str(), Loc->getLine());
      }
    }
  }
}

PreservedAnalyses FunctionApproximation::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  auto ConfigOrErr =
      AM.getResult<ApproximationAnalysis>(F).getConfiguration(this->T);

  // no approximations for this technique
  if (auto Err = ConfigOrErr.takeError()) {
    consumeError(std::move(Err));
    return PreservedAnalyses::all();
  }

  auto configuration = *ConfigOrErr;

  std::vector<CallInst *> toReplace;

  bool modified = false;
  for (auto I = inst_begin(F); I != inst_end(F); ++I) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      // skip calls to indirect function invocations  (invokes)
      if (Function *calledF = CI->getCalledFunction()) {
        StringRef functionName = formatString(calledF->getName());
        // add calls to specific functions to the list
        if (functionReplacementList.count(functionName.str()))
          toReplace.push_back(CI);
      }
    }
  }

  // should be an error / exception
  if (toReplace.size() != configuration.size())
    llvm_unreachable("List of replaceable calls not in accordance to number of "
                     "opportunities for this function ");
  // iterate on all instructions that will be replaced
  for (size_t i = 0; i < toReplace.size(); i++) {
    modified |= tryToOptimize(toReplace[i], configuration[i]);
  }

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserveSet<CFGAnalyses>();

  return modified ? PA : PreservedAnalyses::all();
}

// Register pass on newPM
llvm::PassPluginLibraryInfo getFApproxPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "FApproxPass", "v0.1", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef PassName, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (PassName == "fapprox") {
                    FPM.addPass(FunctionApproximation());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getFApproxPluginInfo();
}
