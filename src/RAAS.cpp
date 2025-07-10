#include "jit/JIT.h"
#include "jit/passes/Passes.h"
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

static cl::list<std::string>
    Dylibs("dlopen", cl::CommaSeparated,
           cl::desc("Dynamic libraries to load before linking"));

static cl::list<std::string> inputFiles(
    "precise-modules", cl::NotHidden, cl::CommaSeparated,
    cl::desc(
        "List of modules that are compiled as-is, without approximations."));

static cl::list<std::string>
    approxFiles("approx-modules", cl::NotHidden, cl::CommaSeparated,
                cl::desc("List of modules that are part of the application and "
                         "should be re-compiled with approximations."));

static cl::opt<std::string>
    evaluationFile("evaluation-file", cl::Positional, cl::Required,
                   cl::desc("path to the evaluation bitcode file"));

static cl::opt<double>
    errorLimit("error-limit", cl::init(0.3), cl::Optional,
               cl::desc("Error limit for evaluation system"));

static cl::opt<std::string> trainingModeOpt(
    "training-mode", cl::init(""), cl::Optional,
    cl::desc("Store JSON file for each configuration. Starts "
             "application by reading from previous JSON file. Pass a string "
             "after the option to inform the name of the json file"));

static cl::opt<bool> clPrintRankedOpportunities(
    "print-ranked", cl::init(false), cl::Optional,
    cl::desc("Print ranked opportunities after framework concludes operating"));

static cl::opt<TIER> scoringAggressiveness(
    "scoring-aggressiveness",
    cl::desc("Chose how agressive we want to value speedups over lower errors "
             "for configurations"),
    cl::values(clEnumVal(low, "Favor lower error rates first"),
               clEnumVal(medium, "Balance between favoring higher "
                                 "speedups and lower error rates"),
               clEnumVal(high, "Favor higher speedups first")),
    cl::init(low));

static cl::opt<bool> clMemoryAware(
    "memory-conscious", cl::init(false), cl::Optional,
    cl::desc("Set the evaluation system to pay attention and discarding "
             "approximations that introduce memory leaks"));

// json file that lists skippable approximations
// should follow:
// function_name : {
//  technique_name : {
//    "op_number": max_allowed_int,
//    ...
//  }
// }
static cl::opt<std::string>
    forbiddenApproxFile("forbidden-approx-list", cl::NotHidden, cl::Optional,
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
  initializeTarget();

  // add our options to a specific category
  // TODO: find a better way to do this
  cl::OptionCategory RAASCategory("Options");
  clPrintFunctions.addCategory(RAASCategory);
  inputFiles.addCategory(RAASCategory);
  approxFiles.addCategory(RAASCategory);
  evaluationFile.addCategory(RAASCategory);
  Dylibs.addCategory(RAASCategory);
  clNoApprox.addCategory(RAASCategory);
  forbiddenApproxFile.addCategory(RAASCategory);
  errorLimit.addCategory(RAASCategory);
  trainingModeOpt.addCategory(RAASCategory);
  clPrintRankedOpportunities.addCategory(RAASCategory);
  scoringAggressiveness.addCategory(RAASCategory);
  clMemoryAware.addCategory(RAASCategory);
  // hide options not from our program
  cl::HideUnrelatedOptions(RAASCategory);

  cl::ParseCommandLineOptions(argc, argv, "Runtime JIT Approximation");

  // load extra dylibs required for the modules
  ExitOnErr(loadDylibs());

  // if this option is set, print approximable functions and exit
  if (clPrintFunctions) {
    SMDiagnostic error;
    LLVMContext ctx;
    passlist::buildPasses();
    for (auto &file : approxFiles) {
      auto m = parseIRFile(file, error, ctx);

      if (m.get() == nullptr) {
        errs() << "Invalid file for module " << m->getName() << '\n';
        return 1;
      }
      passlist::printApproxOpportunities(*m.get());
    }
    return 0;
  }

  // create the JIT sending the application IR module and eval function module
  // if trainingMode is set to default, let the JIT decide program name
  if (trainingModeOpt.getNumOccurrences())
    J = ExitOnErr(orc::ApproxJIT::Create(evaluationFile, scoringAggressiveness,
                                         errorLimit, clNoApprox,
                                         trainingModeOpt));
  else
    J = ExitOnErr(orc::ApproxJIT::Create(evaluationFile, scoringAggressiveness,
                                         errorLimit, clNoApprox, ""));

  // read list of forbidden approximations
  if (forbiddenApproxFile.getNumOccurrences()) {
    auto forbiddenApproxList =
        ForbiddenApproximations::Create(forbiddenApproxFile);
    if (auto Err = forbiddenApproxList.getError()) {
      errs() << Err.message() << "\n";
      std::exit(1);
    }
    J->setForbiddenApproxList(std::move(forbiddenApproxList.get()));
  }

  // monitor memory consumption
  if (clMemoryAware)
    J->setMemoryAware();

  if (not inputFiles.getNumOccurrences() and
      not approxFiles.getNumOccurrences()) {
    errs() << "At least one precise or approx module must be informed!\n";
    std::exit(1);
  }

  if (inputFiles.getNumOccurrences()) {
    for (auto inputFile : inputFiles)
      ExitOnErr(J->addModuleFile(inputFile));
  }

  if (approxFiles.getNumOccurrences()) {
    for (auto approxFile : approxFiles)
      ExitOnErr(J->addModuleApproxFile(approxFile));
  }

#ifndef DEBUG
  // initialize the platform that takes care of static constructors
  ExitOnErr(J->initializePlatform());
#endif

  // find the location of the main symbol in the JIT
  auto mainSymb = ExitOnErr(J->lookup("main"));

  // fill the arguments vector
  InputArgv.insert(InputArgv.begin(), "jitted_program");

  // run the main function
  J->runAsMain(ExecutorAddr(mainSymb.getAddress()), InputArgv);

  // print opportunities after evaluation ranked by their score
  if (clPrintRankedOpportunities)
    J->printRankedOpportunities();

#ifndef DEBUG
  // run destructors
  ExitOnErr(J->deinitializePlatform());
#endif
  return 0;
}
