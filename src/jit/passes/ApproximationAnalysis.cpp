#include "Passes.h"

ApproximationAnalysis::ApproximationAnalysis(
    fnNameMapPair configs)
    : configurationMapPair(configs) {}

ApproximationAnalysis::ApproxAnalysisWrapperResult
ApproximationAnalysis ::run(Function &F, FunctionAnalysisManager &) {
  if (F.getName() == configurationMapPair.first)
    return ApproxAnalysisWrapperResult(configurationMapPair.second);
  else
    // should not happen, but a safeguard to send an empty map nonetheless
    return ApproxAnalysisWrapperResult(configurationPerTechniqueMap());
}

AnalysisKey ApproximationAnalysis::Key;
