#pragma once
#include <map>

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"

#define IPO_PIPELINE                                                           \
  "annotation2metadata,forceattrs,inferattrs,coro-early,function<eager-inv>("  \
  "lower-expect,sroa<modify-cfg>,early-cse<>,callsite-splitting),openmp-opt,"  \
  "ipsccp,called-value-propagation,globalopt,function(mem2reg),require<"       \
  "globals-aa>,function(invalidate<aa>),"                                      \
  "require<profile-summary>,cgscc(devirt<4>(inline<only-mandatory>,inline,"    \
  "function-attrs,argpromotion,openmp-opt-cgscc,function<eager-inv>(sroa<"     \
  "modify-cfg>,early-cse<memssa>,speculative-execution,correlated-"            \
  "propagation,libcalls-shrinkwrap,"                                           \
  "tailcallelim,reassociate,require<opt-remark-emit>,sroa<modify-"             \
  "cfg>,vector-combine,mldst-motion<no-split-footer-bb>,sccp,bdce,"            \
  "correlated-propagation,adce,memcpyopt,dse,coro-elide"                       \
  "),coro-split)),deadargelim,coro-cleanup,globalopt,globaldce,"               \
  "elim-avail-extern,rpo-function-attrs,recompute-globalsaa,function<eager-"   \
  "inv>(float2int,lower-constant-intrinsics,loop-distribute,inject-tli-"       \
  "mappings,slp-vectorizer,vector-combine,transform-"                          \
  "warning,sroa<preserve-cfg>,require<opt-remark-emit>,alignment-"             \
  "from-assumptions,loop-sink,instsimplify,div-rem-pairs,tailcallelim),"       \
  "globaldce,constmerge,cg-profile,rel-lookup-table-converter,function("       \
  "annotation-remarks),verify"

//#define IPO_PIPELINE                                                           \
//  "annotation2metadata,forceattrs,inferattrs,coro-early,function<eager-inv>("  \
//  "lower-expect,sroa<modify-cfg>,early-cse<>,callsite-splitting),openmp-opt,"  \
//  "ipsccp,called-value-propagation,globalopt,function(mem2reg),function<"      \
//  "eager-inv>(instcombine),require<globals-aa>,function(invalidate<aa>),"      \
//  "require<profile-summary>,cgscc(devirt<4>(inline<only-mandatory>,inline,"    \
//  "function-attrs,argpromotion,openmp-opt-cgscc,function<eager-inv>(sroa<"     \
//  "modify-cfg>,early-cse<memssa>,speculative-execution,correlated-"            \
//  "propagation,instcombine,aggressive-instcombine,libcalls-shrinkwrap,"        \
//  "tailcallelim,reassociate,require<opt-remark-emit>,instcombine,sroa<modify-" \
//  "cfg>,vector-combine,mldst-motion<no-split-footer-bb>,sccp,bdce,"            \
//  "instcombine,correlated-propagation,adce,memcpyopt,dse,coro-elide,"          \
//  "instcombine),coro-split)),deadargelim,coro-cleanup,globalopt,globaldce,"    \
//  "elim-avail-extern,rpo-function-attrs,recompute-globalsaa,function<eager-"   \
//  "inv>(float2int,lower-constant-intrinsics,loop-distribute,inject-tli-"       \
//  "mappings,instcombine,slp-vectorizer,vector-combine,instcombine,transform-"  \
//  "warning,sroa<preserve-cfg>,instcombine,require<opt-remark-emit>,alignment-" \
//  "from-assumptions,loop-sink,instsimplify,div-rem-pairs,tailcallelim),"       \
//  "globaldce,constmerge,cg-profile,rel-lookup-table-converter,function("       \
//  "annotation-remarks),verify"
enum approxTechnique { FAP, LPAR, LPERF, GAP };
enum TIER { low, medium, high };

// translate enum to string with technique name
static const std::map<approxTechnique, std::string> techniqueNameMap = {
    {FAP, "Function Approximation"},
    {LPAR, "Loop Parallelization"},
    {GAP, "GEMM Approximation"},
    {LPERF, "Loop Perforation"}};

// maximum approximate parameter per technique
static const std::map<approxTechnique, unsigned> maxParameter = {
    {FAP, 2}, {LPAR, 4}, {GAP, 5}, {LPERF, 16}};

using configurationPerTechniqueMap =
    std::map<const approxTechnique, std::vector<int>>;

using namespace llvm;

class EvaluationSystem;
class ForbiddenApproximations;

/**
 * This class receives the current application and error status and
 * approximation opportunities and outputs the desired combination to
 * approximate. This is an abstract class meant to be extended with desired
 * configuration selection logic.
 */
class ConfigurationEvaluation {
  friend EvaluationSystem;

public:
  ConfigurationEvaluation(double elim) : errorLimit(elim) {}
  /* create the data structure used for the heuristic to compute the optimal
   * approximation per technique
   */
  virtual void buildApproximationStructures(StringRef functionName,
                                            configurationPerTechniqueMap M) = 0;

  /* given a function, return the configuration suggested by the heuristic */
  virtual configurationPerTechniqueMap
  getSuggestedConfiguration(StringRef FunctionName) = 0;

  /* declare the destructor as virtual for inheritance purposes */
  virtual ~ConfigurationEvaluation() {}

  /* optional method used to inform if we are already at optimal configuration
   */
  bool achievedConvergence() { return foundOptimal; }

  /* Check if the current function has achieved convergence in each possible
   * configuration
   */
  virtual bool hasFoundOptimalConfiguration(std::string functionName) = 0;

  /* returns true if this function received an update in approximation on the
   * last evaluation iteration
   */
  virtual bool wasUpdatedInLastEvaluation(std::string functionName) = 0;

  virtual void unmarkAsUpdated(std::string functionName) = 0;

  /* print configurations sorted by score. This method will sort the original
   * list and must only be called after evaluation ends.
   */
  virtual std::string getRankedConfigurations(bool csv_format) = 0;

  /* returns a string containing a JSON object with current opportunity
   * configuration
   */
  virtual std::string getJSONConfiguration() = 0;

  /* open JSON file and restore configuration state
   * from it.
   */
  virtual void restoreStateFromJSON(StringRef functionName,
                                    std::string JSONFile) = 0;

  void setMonitorMemoryConsumption(bool shouldMonitor) {
    monitorMemoryConsumption = shouldMonitor;
  }

  bool monitorsMemoryConsumption() { return monitorMemoryConsumption; }

  class approximationQuality {
    double RoIpreciseTime = 0.0;
    double IterationPreciseTime = 0.0;
    std::optional<long> memoryConsumption = 0.0;
    // first -> second to last loop
    // second -> last loop
    std::pair<double, double> RoIspeedups;
    std::pair<double, double> errors;
    std::pair<double, double> iterationTimes;

  public:
    approximationQuality() {
      errors = {.0, .0};
      RoIspeedups = {1.0, 1.0};
      iterationTimes = {0.0, 0.0};
    }

    void updateRoIValues(double error, double time) {
      errors = {errors.second, error};
      RoIspeedups = {RoIspeedups.second, RoIpreciseTime / time};
    }

    void updateIterationTime(double time) {
      iterationTimes = {iterationTimes.second, time};
    }

    void updateMemoryConsumption(long memory) { memoryConsumption = memory; }

    llvm::Expected<long> getMemoryConsumption() {
      if (memoryConsumption.has_value())
        return memoryConsumption.value();

      return make_error<StringError>(
          "There's no memory consumption. Maybe re-run with --memory-conscious",
          inconvertibleErrorCode());
    }

    const double getAvgSpeedup() {
      return (RoIspeedups.second + RoIspeedups.first) / 2;
    }

    /* update precise time in increments (both iteration and RoI) */
    void incrementRoIPreciseTime(double time) { RoIpreciseTime += time; }
    void incrementIterationPreciseTime(double time) {
      IterationPreciseTime += time;
    }

    std::pair<double, double> getErrors() { return errors; }

    std::pair<double, double> getRoISpeedups() { return RoIspeedups; }

    std::pair<double, double> getIterationTimes() { return iterationTimes; }
    double getIterationPreciseTime() { return IterationPreciseTime; }
  } APQ;

  void setForbiddenApproxList(std::unique_ptr<ForbiddenApproximations> fap) {
    forbiddenApproxList = std::move(fap);
  }
  /* informs if we are already at optimal configuration */
  bool foundOptimal = false;

protected:
  std::unique_ptr<ForbiddenApproximations> forbiddenApproxList = nullptr;
  /* evaluates the score for each approximation opportunity for each function
   * and returns the choosen configuration set
   */
  virtual void updateSuggestedConfigurations() = 0;
  const double errorLimit;
  bool monitorMemoryConsumption;
};

/**
 * this class is the basis for the evaluation system, providing general
 * information on approximable configurations and controlling the
 * ConfigurationEvaluation subsystem
 */
class EvaluationSystem {
public:
  EvaluationSystem(std::unique_ptr<ConfigurationEvaluation> eval,
                   bool trainingMode = false, std::string programName = "");

  // add a function to our map
  void addFunction(const Function &F);

  // check if function is already added into analysis
  bool findFunction(const std::string FunctionName);
  bool findFunction(const Function &F);

  // checks if the function is approximable
  static bool isApproximable(const Function &F);
  static void printApproximationOpportunities(const Function &F);

  // call the evaluator so we can update the suggested configurations. returns a
  // set of configurations to reapproximate functions
  StringMap<configurationPerTechniqueMap> updateSuggestedConfigurations();

  configurationPerTechniqueMap
  getConfigurationForFunction(StringRef functionName);

  static std::string
  getCombinationFromConfiguration(configurationPerTechniqueMap configuration);

  void incrementRoIPreciseTime(double precTime);
  void incrementIterationPreciseTime(double precTime);

  void updateQualityValues(double error, double time, double iterationTime);

  void updateMemoryConsumption(long memory);

  double getLastSpeedup();
  double getLastError();
  double getErrorLimit();

  void setForbiddenApproxList(std::unique_ptr<ForbiddenApproximations> fap) {
    evaluator->setForbiddenApproxList(std::move(fap));
  }

  ConfigurationEvaluation *getConfigEvaluator() { return evaluator.get(); }

  /*
   * iterate over all approximate functions and print their opportunities sorted
   * by score
   */
  void printRankedOpportunities(bool csv_format = false);

  /*
   * Asks the evaluator if the function in question has found optimal
   * configuration, i.e. it won't be approximated anymore by the evaluator. The
   * JIT can use this information to discard non-used symbols and preserve
   * memory.
   */
  bool hasFoundOptimalConfiguration(std::string functionName);

  bool monitorsMemoryConsumption();

  void setMonitorMemoryConsumption(bool shouldMonitor);

private:
  std::unique_ptr<ConfigurationEvaluation> evaluator;
  // for each function, a configurationPerTechniqueMap
  StringMap<configurationPerTechniqueMap> functionOpportunityList;

  // iterates over the instructions inside the function to find approximation
  // opportunities for each technique.
  // returns a map that translates approxTechnique -> config vector
  configurationPerTechniqueMap populateApproximationRate(const Function &F);

  /*
   * Store last seen configuration into a JSON File
   */
  void storeConfigurationToFile();

  const std::string programName;
  const bool trainingMode;
};

/**
 * stores forbidden approximations for each technique to
 * avoid runtime errors
 */
class ForbiddenApproximations {
  using SkippableApprox = std::map<unsigned, unsigned>;
  using opportunitiesPerFunction = std::map<approxTechnique, SkippableApprox>;

public:
  // map for configuration -> max rate before skipping
  ForbiddenApproximations(StringRef jsonFile);

  opportunitiesPerFunction
  getForbiddenApproxForFunction(std::string functionName);

  static ErrorOr<std::unique_ptr<ForbiddenApproximations>>
  Create(StringRef jsonFileName);

private:
  // functionName -> skippableApprox
  std::map<std::string, opportunitiesPerFunction> skippableApprox;
  static const std::map<std::string, approxTechnique> stringToTechType;
};
