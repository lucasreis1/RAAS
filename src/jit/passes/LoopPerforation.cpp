#include "Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

LoopPerforation::PerforableLoop::PerforableLoop(Loop *l, LoopInfo &li,
                                                Function *parentF,
                                                unsigned perfRate)
    : perforRate(perfRate), L(l), LI(li), parentFun(parentF) {
  IRB = new IRBuilder<>(parentFun->getContext());
}
LoopPerforation::PerforableLoop::~PerforableLoop() { delete IRB; }

bool LoopPerforation::PerforableLoop::isPerforable(Loop *L) {
  auto termInst = dyn_cast_or_null<BranchInst>(L->getHeader()->getTerminator());
  // do not perforate loops not ending in a branch inst
  if (termInst == nullptr)
    return false;

  // skip loops with more than one latch
  if (L->getLoopLatch() == nullptr)
    return false;

  // the body starts on the first BB after the header that is not a latch or
  // exit block
  bool foundBody = false;
  for (auto BI : termInst->successors()) {
    if (!L->contains(BI) or L->isLoopLatch(BI))
      continue;
    foundBody = true;
    break;
  }

  // if we still haven't found the body, that means it is part of the
  // latch/header by CFG simplification. Such loops are not perforable as we
  // would not be skipping anything. Loops without a preheader are skipped as
  // we need this BB to increment the approx counter
  return foundBody && L->getLoopPreheader();
}

bool LoopPerforation::PerforableLoop::perforateLoop() {
  // L->print(errs(), true);
  Type *Int32 = Type::getInt32Ty(parentFun->getContext());

  auto *entryPt = &*parentFun->getEntryBlock().getFirstInsertionPt();

  IRB->SetInsertPoint(entryPt);

  auto allocaCounter =
      IRB->CreateAlloca(Int32, nullptr, "perforation_counter_ptr");
  allocaCounter->setAlignment(Align(4));

  IRB->SetInsertPoint(L->getLoopPreheader()->getTerminator());
  IRB->CreateStore(ConstantInt::get(Int32, 0), allocaCounter)
      ->setAlignment(Align(4));

  auto termInst = dyn_cast_or_null<BranchInst>(L->getHeader()->getTerminator());
  if (termInst == nullptr) {
    errs() << "should no be reaching here!\n";
    return false;
  }

  // the body starts on the first BB after the header that belongs to the
  // loop, is not a latch or exiting block
  for (auto BI : termInst->successors()) {
    if (!L->contains(BI) or L->isLoopLatch(BI) or BI == L->getExitBlock())
      continue;
    loopBody = BI;
    break;
  }

  loopLatch = L->getLoopLatch();

  if (loopBody == nullptr) {
    errs() << "should not be reaching here\n";
    exit(1);
  }

  BasicBlock *approxCheckBB = BasicBlock::Create(
      parentFun->getContext(), "approx_check", parentFun, loopBody);

  IRB->SetInsertPoint(approxCheckBB);
  auto counter = IRB->CreateLoad(Int32, allocaCounter, "counter");
  counter->setAlignment(Align(4));
  // extract the LSB bits from the counter
  auto LSBCounter =
      IRB->CreateAnd(counter, ConstantInt::get(Int32, (1 << perforRate) - 1));
  // check if counter is null
  auto truncatedIsNull = IRB->CreateIsNull(LSBCounter);
  // if LSB is zero, jump to body
  // else, jump to latch
  IRB->CreateCondBr(truncatedIsNull, loopBody, loopLatch);

  termInst->replaceSuccessorWith(loopBody, approxCheckBB);

  L->addBasicBlockToLoop(approxCheckBB, LI);

  IRB->SetInsertPoint(loopLatch, loopLatch->getFirstInsertionPt());
  // increment counter
  auto loadCounter = IRB->CreateLoad(Int32, allocaCounter);
  loadCounter->setAlignment((Align(4)));
  auto counterIncr = IRB->CreateNSWAdd(loadCounter, ConstantInt::get(Int32, 1));
  IRB->CreateStore(counterIncr, allocaCounter)->setAlignment((Align(4)));
  return true;
}

bool LoopPerforation::tryToPerforateLoop(Function *F, Loop *L,
                                                LoopInfo &LI, unsigned param) {
  if (param == 0)
    return false;
  else {
    auto PL = LoopPerforation::PerforableLoop(L, LI, F, param);
    return PL.perforateLoop();
  }
}

PreservedAnalyses LoopPerforation::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto ConfigOrErr =
      AM.getResult<ApproximationAnalysis>(F).getConfiguration(this->T);

  if (auto Err = ConfigOrErr.takeError()) {
    consumeError(std::move(Err));
    return PreservedAnalyses::all();
  }

  auto configuration = *ConfigOrErr;

  std::vector<Loop *> toPerforate;
  // toPerforate.reserve(configuration.size());
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  for (Loop *TopLevelLoop : LI) {
    for (Loop *L : depth_first(TopLevelLoop))
      if (PerforableLoop::isPerforable(L)) {
        toPerforate.push_back(L);
      }
  }

  if (toPerforate.size() != configuration.size())
    llvm_unreachable("List of perforable loops not in accordance to number of "
                     "opportunities for this function ");

  bool modified = false;

  for (unsigned i = 0; i < toPerforate.size(); i++) {
    Loop *L = toPerforate[i];
    unsigned param = configuration[i];
    modified |= tryToPerforateLoop(&F, L, LI, param);
  }

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  if (modified)
    return PA;

  return PreservedAnalyses::all();
}

// create loopinfo object for static contexts
LoopInfo LoopPerforation::getLoopInfo(const Function &F) {
  DominatorTree DT = DominatorTree(const_cast<Function &>(F));
  return LoopInfo(DT);
}

bool LoopPerforation::isApproximable(const Function &F) {
  // we need loop info analysis
  auto loopInfo = getLoopInfo(F);

  // if a perforable loop is found, function is approximable
  for (Loop *TopLevelLoop : loopInfo) {
    for (Loop *L : depth_first(TopLevelLoop)) {
      if (PerforableLoop::isPerforable(L))
        return true;
    }
  }
  return false;
}

void LoopPerforation::printApproximationOpportunities(const Function &F) {
  auto loopInfo = getLoopInfo(F);

  for (Loop *TopLevelLoop : loopInfo) {
    for (Loop *L : depth_first(TopLevelLoop)) {
      if (PerforableLoop::isPerforable(L)) {
        if (DILocation *Loc = L->getStartLoc()) {
          std::string fileName = Loc->getFilename().str();
          fprintf(stderr, "Perforable loop @ %s:%d\n", fileName.c_str(),
                  Loc->getLine());
        }
      }
    }
  }
}

unsigned LoopPerforation::searchApproximableCalls(const Function &F) {
  auto loopInfo = getLoopInfo(F);

  unsigned perforableCount = 0;
  for (Loop *TopLevelLoop : loopInfo) {
    for (Loop *L : depth_first(TopLevelLoop)) {
      if (PerforableLoop::isPerforable(L))
        perforableCount++;
    }
  }
  return perforableCount;
}
