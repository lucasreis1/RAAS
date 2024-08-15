#include "jit/JIT.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::orc;

#define DEBUG_TYPE "raas"

std::unique_ptr<orc::ApproxJIT> J;

namespace {
static cl::opt<bool>
    clPrintFunctions("list-functions", cl::NotHidden, cl::init(false),
                     cl::Optional,
                     cl::desc("print list of approximable functions and exit"));

static cl::opt<bool>
    clNoApprox("no-approx", cl::NotHidden, cl::init(false), cl::Optional,
               cl::desc("Execute the JIT pipeline without applying "
                        "approximations. Ideal for overhead checks"));

static cl::opt<std::string>
    inputFile("<bitcode-file>", cl::Positional, cl::Required,
              cl::desc("Path to the application bitcode file"));

static cl::list<std::string>
    Dylibs("dlopen", cl::desc("Dynamic libraries to load before linking"));

static cl::opt<std::string>
    evaluationFile("evaluation-file", cl::Positional, cl::Required,
                   cl::desc("path to the evaluation bitcode file"));

// json file that lists skippable approximations
// should follow:
// function_name : {
//  technique_name : {
//    "op_number": max_allowed_int,
//    ...
//  }
// }
static cl::opt<std::string>
    forbiddenApproxFile("forbidden-approx-list", cl::NotHidden, cl::init(""),
                        cl::Optional,
                        cl::desc("json file for skippable approximations"));

static cl::list<std::string> InputArgv(cl::ConsumeAfter,
                                       cl::desc("<program arguments>..."));
} // namespace

Error loadDylibs() {
  for (const auto &Dylib : Dylibs) {
    std::string ErrMsg;
    if (sys::DynamicLibrary::LoadLibraryPermanently(Dylib.c_str(), &ErrMsg))
      return make_error<StringError>(ErrMsg, inconvertibleErrorCode());
  }
  return Error::success();
}

int main(int argc, char *argv[]) {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  SMDiagnostic error;

  // add our options to a specific category
  // TODO: find a better way to do this
  cl::OptionCategory ourCategory("Options");
  clPrintFunctions.addCategory(ourCategory);
  inputFile.addCategory(ourCategory);
  evaluationFile.addCategory(ourCategory);
  Dylibs.addCategory(ourCategory);
  clNoApprox.addCategory(ourCategory);
  forbiddenApproxFile.addCategory(ourCategory);
  // hide options not from our program
  // cl::HideUnrelatedOptions(ourCategory);

  cl::ParseCommandLineOptions(argc, argv, "Runtime JIT Approximation");

  // load extra dylibs required for the modules
  ExitOnErr(loadDylibs());

  // if this option is set, print approximable functions and exit
  // if (clPrintFunctions) {
  //  passlist::printApproxOpportunities(*m.get());
  //  return 0;
  //}

  // create the JIT sending the application IR module and eval function module
  J = ExitOnErr(orc::ApproxJIT::Create(inputFile, evaluationFile));

  // read list of forbidden approximations
  if (forbiddenApproxFile != "") {
    auto forbiddenApproxList =
        ForbiddenApproximations::Create(forbiddenApproxFile);
    if (!forbiddenApproxList)
      llvm_unreachable("Forbidden approximation file invalid!");
    J->setForbiddenApproxList(std::move(forbiddenApproxList.get()));
  }

#ifndef DEBUG
  // initialize the platform that takes care of static constructors
  ExitOnErr(J->initializePlatform());
#endif

  // find the location of the main symbol in the JIT
  auto mainSymb = ExitOnErr(J->lookup("main"));

  // fill the arguments vector
  InputArgv.insert(InputArgv.begin(), inputFile);

  // run the main function
  J->runAsMain(ExecutorAddr(mainSymb.getAddress()), InputArgv);

  // print opportunities after evaluation ranked by their score
  J->printRankedOpportunities();

#ifndef DEBUG
  // run destructors
  ExitOnErr(J->deinitializePlatform());
#endif
  return 0;
}
