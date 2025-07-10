#include "Core.h"
#include "misc/VectorWrapper.h"
#include <limits>

/* This evaluator takes a greedy approach to approximation.
 * We test each possible configuration in isolation exactly once for their
 * smallest aggressiveness possible. We score configurations based on their
 * achieved speedup and error rates. We then combine approximations one-by-one
 * in order of score, adding a new approximation if it gives us speedups and
 * doesn't achieve errors above the fixed upper limit
 */
class SimpleEvaluator : public ConfigurationEvaluation {
public:
  SimpleEvaluator(double elim, TIER aggressiveness = low);

  // inherited methods
  void updateSuggestedConfigurations();
  void buildApproximationStructures(StringRef functionName,
                                    configurationPerTechniqueMap M);

  bool wasUpdatedInLastEvaluation(std::string functionName);
  void unmarkAsUpdated(std::string functionName);
  bool hasFoundOptimalConfiguration(std::string functionName);
  std::string getRankedConfigurations(bool csv_format);
  std::string getJSONConfiguration();
  void restoreStateFromJSON(StringRef functionName, std::string JSONFile);

private:
  configurationPerTechniqueMap
  getSuggestedConfiguration(StringRef functionName);

  struct OpportunitiesPerFunction;
  // stores information for a single approximation opportunity
  struct approximationInfo {
    unsigned index;
    // overall approximation score (higher is better)
    double score = 0.;
    // overall speedup achieved on the configuration
    double speedup = 0.;
    // Non-RoI iteration time (entire application)
    // This is only used as a fallback to ensure we are not
    // getting application slowdowns, even if we have speedups on RoI
    double iterationTime = 0.;
    // System memory usage during this process execution
    // We log this for memory-aware approaches where we don't want approximations to introduce memory leaks
    long memoryUsage = 0L;
    // <iteration_time, speedup, parameter> pair that indicates the parameter
    // which yields maximum speedup for this configuration
    struct idealCfgStruct {
      double speedup = 1.;
      unsigned parameter = 0;
      double iterationTime = 0.;
      long memoryUsage = 0L;
    };

    idealCfgStruct idealConfig;
    // parameter sent to pass for approximation level
    unsigned parameter = 0;
    unsigned maxParameter = 0;
    // found optimal approximation
    bool foundOptimal = false;
    approxTechnique AT;

    OpportunitiesPerFunction *parent;

    approximationInfo(unsigned idx, approxTechnique T, unsigned maxParameter,
                      OpportunitiesPerFunction *parent)
        : index(idx), maxParameter(maxParameter), AT(T), parent(parent) {}
  };
  // minimum required speedup for a new configuration to attain.
  // As we can't always get the last speedup as the ideal because new
  // configs are tried side-by-side, we store the min. expected speedup here.
  unsigned minRequiredSpeedup = 1.;
  double maxRequiredIterationTime = std::numeric_limits<double>::infinity();

  int iterationCount = 0;
  int heuristicalCount = 0;

  using opportunityList = std::vector<approximationInfo>;
  using techniqueToOpportunities = std::map<approxTechnique, opportunityList>;

  // stores list of opportunities per technique per function, also signals if a
  // function was recently changed by evaluation
  struct OpportunitiesPerFunction {
  public:
    OpportunitiesPerFunction(std::string fnName,
                             techniqueToOpportunities opportunitiesPerTechnique)
        : functionName(fnName),
          opportunitiesPerTechnique(opportunitiesPerTechnique) {}

    techniqueToOpportunities opportunitiesPerTechnique;
    bool updatedInLastEvaluation = false;

    std::string functionName;
  };

  StringMap<OpportunitiesPerFunction> oppPerFunctionMap;

  OpportunitiesPerFunction &
  getOpportunitiesForFunction(std::string functionName);

  // this is the opportunity we are checking in the current iteration
  approximationInfo *lastCheckedOpportunity = nullptr;
  // translates all the maps into a single vector so we can iterate without
  // major headaches
  VectorWrapper<approximationInfo> opportunitiesWrapper;

  double getScore();

  static std::map<approxTechnique, std::string> techToStrMap;

  // how aggressive do we want to value speedups instead of lower error rates?
  const TIER scoringAggressiveness;
};
