#include "Core.h"
#include "misc/VectorWrapper.h"

class SimpleEvaluator : public ConfigurationEvaluation {
public:
  SimpleEvaluator();

  // inherited methods
  void updateSuggestedConfigurations();
  void buildApproximationStructures(StringRef functionName,
                                    configurationPerTechniqueMap M);

  bool wasUpdatedInLastEvaluation(std::string functionName);
  void unmarkAsUpdated(std::string functionName);
  std::string getRankedConfigurations();

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
    // <speedup, parameter> pair that indicates the parameter which yields
    // maximum speedup for this configuration
    std::pair<double, unsigned> idealConfig = {1., 0};
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
};
