#include "Core.h"
#include "SimpleEvaluator.h"
#include <iomanip>
#include <ios>
#include <sstream>

// number of repetitions required to test a single configuration
#define LOOPS_PER_CONFIG 3

bool isNan(double v) { return v != v; }

// empty constructor, we start building the attributes only when we need them
SimpleEvaluator::SimpleEvaluator() {}

void SimpleEvaluator::buildApproximationStructures(
    StringRef functionName, configurationPerTechniqueMap M) {
  auto &opPerFunction = getOpportunitiesForFunction(functionName.str());
  techniqueToOpportunities &evalMap = opPerFunction.opportunitiesPerTechnique;
  // iterate over all techniques
  for (auto MI : M) {
    auto technique = MI.first;
    // create opportunity list the size of the number of approximations
    opportunityList L;
    L.reserve(MI.second.size());

    for (size_t index = 0; index < MI.second.size(); index++) {
      auto oppInfo = approximationInfo(
          index, technique, maxParameter.at(technique), &opPerFunction);

      // if there is a forbidden approx list, set configurations as provided
      if (forbiddenApproxList) {
        auto forbiddenPerFunction =
            forbiddenApproxList->getForbiddenApproxForFunction(
                functionName.str());
        auto forbiddenPerTechnique = forbiddenPerFunction[technique];
        if (forbiddenPerTechnique.find(index + 1) !=
            forbiddenPerTechnique.end()) {
          oppInfo.maxParameter = forbiddenPerTechnique[index + 1];
          if (oppInfo.maxParameter == 0)
            oppInfo.foundOptimal = true;
        }
      }
      L.push_back(oppInfo);
    }
    // the eval map for this technique is the created list
    auto II = evalMap.insert(std::make_pair(technique, L));

    // add all arrays into our large vector
    opportunitiesWrapper.addVector(II.first->second);
  }

  oppPerFunctionMap.insert(std::make_pair(
      functionName, OpportunitiesPerFunction(functionName.str(), evalMap)));
}

// the desired one
void SimpleEvaluator::updateSuggestedConfigurations() {
  // the number of loops we will use to evaluate each configuration
  int evaluationLimit = opportunitiesWrapper.size() * LOOPS_PER_CONFIG;
  if (!foundOptimal) {
    //  test a new configuration from iterations [1, approxList.size() -1]
    if (iterationCount < evaluationLimit and
        iterationCount % LOOPS_PER_CONFIG == 0) {
      auto &opportunity =
          opportunitiesWrapper[iterationCount / LOOPS_PER_CONFIG];
      if (opportunity.foundOptimal == false) {
        opportunity.parameter = 1;
        opportunity.parent->updatedInLastEvaluation = true;
      }
    }

    // evaluate the last configuration tested from iterations [1,
    // approxSize()]
    if (iterationCount and iterationCount <= evaluationLimit) {
      auto &lastModifiedOp =
          opportunitiesWrapper[(iterationCount - 1) / LOOPS_PER_CONFIG];
      if (lastModifiedOp.foundOptimal == false) {
        lastModifiedOp.score += getScore() / LOOPS_PER_CONFIG;
        lastModifiedOp.speedup += APQ.getSpeedups().second / LOOPS_PER_CONFIG;
        if (iterationCount % LOOPS_PER_CONFIG == 0) {
          lastModifiedOp.parameter = 0;
          lastModifiedOp.parent->updatedInLastEvaluation = true;
          // do not touch opportunity again if error is above limit or error is
          // NaN
          auto lastError = APQ.getErrors().second;
          if (lastError > ERROR_LIMIT or isNan(lastError))
            lastModifiedOp.foundOptimal = true;
        }
      }
    }
    // start our combination heuristic evaluation
    if (iterationCount >= evaluationLimit) {
      // there is no last checked opportunity
      if (lastCheckedOpportunity == nullptr) {
        double maxScore = -std::numeric_limits<double>::infinity();
        for (auto &opportunity : opportunitiesWrapper) {
          // find the opportunity with max score that is still not optimal
          if (not opportunity.foundOptimal and opportunity.score > maxScore) {
            maxScore = opportunity.score;
            lastCheckedOpportunity = &opportunity;
          }
        }
      }
      // we still need to check if we found one opportunity after this search
      if (lastCheckedOpportunity) {
        // if we are evaluating this opportunity for the first time, our target
        // is at least the same speedup as before this opportunity showed up
        if (heuristicalCount == 0 and lastCheckedOpportunity->parameter == 0) {
          lastCheckedOpportunity->idealConfig.first = minRequiredSpeedup;
          lastCheckedOpportunity->speedup = .0;
          lastCheckedOpportunity->parameter = 1;
          lastCheckedOpportunity->parent->updatedInLastEvaluation = true;
        }
        // we want to thest each parameter for n loops
        else if (heuristicalCount > 0 and
                 heuristicalCount <= LOOPS_PER_CONFIG) {
          lastCheckedOpportunity->speedup += APQ.getSpeedups().second;
        }
        if (heuristicalCount >= LOOPS_PER_CONFIG) {
          // reset our count and speedup
          heuristicalCount = -1;
          lastCheckedOpportunity->speedup /= LOOPS_PER_CONFIG;

          auto lastError = APQ.getErrors().second;
          // if the current speedup is bigger than our ideal one, change the
          // propsed optimal configuration
          if (lastCheckedOpportunity->speedup >
                  lastCheckedOpportunity->idealConfig.first and
              lastError <= ERROR_LIMIT and not isNan(lastError)) {
            // update the optimal configuration
            lastCheckedOpportunity->idealConfig = {
                lastCheckedOpportunity->speedup,
                lastCheckedOpportunity->parameter};
          }
          // if we are still not at maximum parameter and still achieving
          // speedups, keep increasing the parameter
          if (lastCheckedOpportunity->parameter !=
              lastCheckedOpportunity->maxParameter) {
            lastCheckedOpportunity->parameter++;
            lastCheckedOpportunity->speedup = 0.;
            lastCheckedOpportunity->parent->updatedInLastEvaluation = true;
          } else { // else, set the opportunity as already checked and set its
                   // optimal parameter
            lastCheckedOpportunity->parameter =
                lastCheckedOpportunity->idealConfig.second;
            lastCheckedOpportunity->foundOptimal = true;
            lastCheckedOpportunity->parent->updatedInLastEvaluation = true;

            // the new minRequiredSpeedup is the one attained by the best rate
            // of this config
            minRequiredSpeedup = lastCheckedOpportunity->idealConfig.first;
            // reset the opportunity to null so we can search new ones
            lastCheckedOpportunity = nullptr;
          }
        }
        heuristicalCount++;
      } else {
        fprintf(stdout, "Converged after %d iterations!\n", iterationCount);
        this->foundOptimal = true;
      }
    }
  }
  // stop counting iterations after we did preprocessing
  // if (iterationCount <= evaluationLimit)
  iterationCount++;

  for (auto opportunity : opportunitiesWrapper)
    fprintf(stdout, "%d %d %d %lf |", opportunity.AT, opportunity.parameter,
            opportunity.foundOptimal, opportunity.score);
  fprintf(stdout, "\n");
}

configurationPerTechniqueMap
SimpleEvaluator::getSuggestedConfiguration(StringRef functionName) {
  configurationPerTechniqueMap M;
  for (auto pair : getOpportunitiesForFunction(functionName.str())
                       .opportunitiesPerTechnique) {
    auto configList = std::vector<int>(pair.second.size());
    // for each operation, extract only the parameter
    for (auto op : pair.second) {
      configList[op.index] = op.parameter;
    }
    // copy the configuration list into the map
    M[pair.first] = configList;
  }
  return M;
}

SimpleEvaluator::OpportunitiesPerFunction &
SimpleEvaluator::getOpportunitiesForFunction(std::string functionName) {
  auto II = oppPerFunctionMap.find(functionName);
  if (II == oppPerFunctionMap.end()) {
    II = oppPerFunctionMap
             .insert(std::make_pair(
                 functionName, OpportunitiesPerFunction(
                                   functionName, techniqueToOpportunities())))
             .first;
  }
  return II->second;
}

bool SimpleEvaluator::wasUpdatedInLastEvaluation(std::string functionName) {
  return oppPerFunctionMap.find(functionName)->second.updatedInLastEvaluation;
}

void SimpleEvaluator::unmarkAsUpdated(std::string functionName) {
  oppPerFunctionMap.find(functionName)->second.updatedInLastEvaluation = false;
}

double SimpleEvaluator::getScore() {
  std::pair<double, double> errors = APQ.getErrors();
  std::pair<double, double> speedups = APQ.getSpeedups();

  // we use max to avoid dividing by zero
  // score = (1 - s^-1)/error
  return (1 - 1 / speedups.second) / std::max(errors.second, 0.001);
}

template <typename T>
std::string formatText(const int width, const char fill, T value) {
  std::stringstream ss;

  ss << std::left << std::setw(width) << std::setfill(fill) << value;
  return ss.str();
}

std::string SimpleEvaluator::getRankedConfigurations() {
  std::sort(
      opportunitiesWrapper.begin(), opportunitiesWrapper.end(),
      [](SimpleEvaluator::approximationInfo a,
         SimpleEvaluator::approximationInfo b) { return a.score > b.score; });

  size_t maxFnSize = 0;
  for (auto &II : oppPerFunctionMap)
    maxFnSize = std::max(II.first().size(), maxFnSize);

  maxFnSize += 2;

  std::stringstream ss;
  const char fill = ' ';
  const int restWidth = 20;

  ss << std::endl;
  // BARS
  ss << " |=" << formatText(12, '=', "") << "=|=";
  ss << formatText(maxFnSize, '=', "") << "=|=";
  ss << formatText(25, '=', "") << "=|=";
  ss << formatText(9, '=', "") << "=|=";
  ss << formatText(9, '=', "") << "=| ";
  ss << std::endl;

  // Headers
  ss << " | " << formatText(12, fill, "SCORE") << " | ";
  ss << formatText(maxFnSize, fill, "FUNCTION") << " | ";
  ss << formatText(25, fill, "TECHNIQUE") << " | ";
  ss << formatText(9, fill, "PARAMETER") << " | ";
  ss << formatText(9, fill, "SPEEDUP") << " | ";
  ss << std::endl;

  // BARS
  ss << " |=" << formatText(12, '=', "") << "=|=";
  ss << formatText(maxFnSize, '=', "") << "=|=";
  ss << formatText(25, '=', "") << "=|=";
  ss << formatText(9, '=', "") << "=|=";
  ss << formatText(9, '=', "") << "=| ";
  ss << std::endl;

  // DATA
  for (auto &aa : opportunitiesWrapper) {
    ss << " | " << formatText(12, fill, aa.score) << " | ";
    ss << formatText(maxFnSize, fill, aa.parent->functionName) << " | ";
    ss << formatText(25, fill, techniqueNameMap.at(aa.AT)) << " | ";
    ss << formatText(9, fill, aa.parameter) << " | ";
    ss << formatText(9, fill, aa.speedup) << " | ";
    ss << std::endl;
  }

  // BARS
  ss << " |=" << formatText(12, '=', "") << "=|=";
  ss << formatText(maxFnSize, '=', "") << "=|=";
  ss << formatText(25, '=', "") << "=|=";
  ss << formatText(9, '=', "") << "=|=";
  ss << formatText(9, '=', "") << "=| ";
  ss << std::endl;

  return ss.str();
}
