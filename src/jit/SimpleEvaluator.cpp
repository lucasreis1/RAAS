#include "Core.h"
#include "SimpleEvaluator.h"
#include "misc/utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>

// number of repetitions required to test a single configuration
#define LOOPS_PER_CONFIG 3

#define DEBUG_TYPE "raas"

// empty constructor, we start building the attributes only when we need them
SimpleEvaluator::SimpleEvaluator(double elim, TIER aggressiveness)
    : ConfigurationEvaluation(elim), scoringAggressiveness(aggressiveness) {}

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
          LLVM_DEBUG(dbgs() << "[RAAS] Fixing parameter " << index + 1 << " as "
                            << forbiddenPerTechnique[index + 1]
                            << " for function " << functionName << "\n";);
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
          if (lastError > errorLimit or isNan(lastError))
            lastModifiedOp.foundOptimal = true;
        }
      } else
        iterationCount += LOOPS_PER_CONFIG - 1;
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
              lastError <= errorLimit and not isNan(lastError)) {
            // update the optimal configuration
            lastCheckedOpportunity->idealConfig = {
                lastCheckedOpportunity->speedup,
                lastCheckedOpportunity->parameter};
          }
          // if we are still not at maximum parameter and still achieving
          // speedups (or are approximating via GEMMs), keep increasing the
          // parameter
          if ((lastCheckedOpportunity->speedup >= 1. or
               lastCheckedOpportunity->AT == approxTechnique::GAP) and
              lastCheckedOpportunity->parameter !=
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
  LLVM_DEBUG(dbgs() << "[RAAS] Opportunities recomputed for iteration " << iterationCount << "\n");
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
  auto error = APQ.getErrors().second;
  auto speedup = APQ.getSpeedups().second;

  switch (scoringAggressiveness) {
  case low:
    // we use max to avoid dividing by zero
    // score = (1 - s^-1)/error
    return (1 - 1 / speedup) / std::max(error, 0.001);
    break;
  case medium:
    return speedup / std::max(error, 0.001);
    break;
  case high:
    return speedup * speedup * speedup / std::max(error, 0.001);
    break;
  }
}

std::string SimpleEvaluator::getJSONConfiguration() {
  auto getMAddNestedObject = [](json::Object &parent,
                                const std::string &key) -> json::Object & {
    auto it = parent.find(key);
    if (it == parent.end())
      it = parent.insert({key, json::Object()}).first;
    return *it->second.getAsObject();
  };

  json::Object lastConfigurationsObject;
  lastConfigurationsObject.insert({"iterations", iterationCount});

  auto &functionsObject =
      getMAddNestedObject(lastConfigurationsObject, "functions");
  for (auto &el : opportunitiesWrapper) {
    auto fn = el.parent->functionName;
    auto &obj = getMAddNestedObject(functionsObject, fn);
    auto &techObj = getMAddNestedObject(obj, std::to_string(el.AT));
    auto confArr =
        json::Array({json::Value(el.parameter), json::Value(el.score),
                     json::Value(el.foundOptimal)});
    techObj[std::to_string(el.index + 1)] = std::move(confArr);
  }

  json::Value jsonValue = std::move(lastConfigurationsObject);
  std::string jsonStr;
  llvm::raw_string_ostream jsonStream(jsonStr);
  jsonStream << jsonValue;
  jsonStream.flush();

  return jsonStr;
}

void SimpleEvaluator::restoreStateFromJSON(StringRef functionName,
                                           std::string JSONFilePath) {
  static llvm::ExitOnError exitOnErr;
  LLVM_DEBUG(dbgs() << "[RAAS] Restoring data from JSON file " << JSONFilePath
                    << "\n";);
  std::ifstream JSONFile(JSONFilePath);
  if (!JSONFile.is_open()) {
    LLVM_DEBUG(
        dbgs() << "[RAAS] File " << JSONFilePath
               << " not found! Continuning without restoring configuration\n");
    return;
  }

  std::stringstream buffer;
  buffer << JSONFile.rdbuf();
  std::string jsonStr = buffer.str();
  JSONFile.close();
  auto jsonObj = *exitOnErr(json::parse(jsonStr)).getAsObject();

  auto iterationsIt = jsonObj.find("iterations");
  assert(iterationsIt != jsonObj.end() &&
         "unable to find number of iterations on JSON config\n");
  auto iterationsOpt = iterationsIt->getSecond().getAsInteger();
  assert(iterationsOpt.has_value() &&
         "must have a value for iterations in the JSON\n");
  if (iterationsOpt.value() != iterationCount)
    iterationCount = iterationsOpt.value();

  auto FnObjectIt = jsonObj.find("functions");
  assert(FnObjectIt != jsonObj.end() &&
         "unable to find map of functions on JSON config\n");

  bool foundFn = false;
  for (auto &fnObject : *FnObjectIt->getSecond().getAsObject()) {
    std::string fnName = fnObject.getFirst().str();
    // to comply with per-function approximation granularity, we'll only load a
    // function of a time this should not provide significant performance impact
    // as reading a json is fairly quick and it's only done once per function
    if (fnName != functionName.str())
      continue;
    foundFn = true;

    auto &opPerFunction = getOpportunitiesForFunction(fnName);
    auto &opPerTechnique = opPerFunction.opportunitiesPerTechnique;
    auto techniques = fnObject.getSecond().getAsObject();
    for (auto &tech : *techniques) {
      auto AT = static_cast<approxTechnique>(std::stoi(tech.getFirst().str()));
      auto configArrIt = opPerTechnique.find(AT);
      assert(configArrIt != opPerTechnique.end() &&
             "Invalid JSON to load config from!\n");
      auto &evalMap = configArrIt->second;
      auto configurations = tech.getSecond().getAsObject();
      for (auto &config : *configurations) {
        auto idx = std::stoi(config.getFirst().str());

        // options are a pair [parameter (int) ,optimal (bool)]
        auto optionsInConfig = config.getSecond().getAsArray();
        auto optParam = (*optionsInConfig)[0].getAsInteger();
        auto optScore = (*optionsInConfig)[1].getAsNumber();
        auto optBool = (*optionsInConfig)[2].getAsBoolean();
        assert(optParam.has_value() && optBool.has_value() &&
               "Missing info for element in JSON");

        auto optParamValue = optParam.value();
        auto optBoolValue = optBool.value();

        // these may be set by the forbiddenapproxlist before we get here.
        // Make sure we comply to those limitations
        auto maxParameter = evalMap[idx - 1].maxParameter;
        auto foundOptimal = evalMap[idx - 1].foundOptimal;

        if (optParamValue > maxParameter)
          optParamValue = maxParameter;

        if (foundOptimal)
          optBoolValue = true;

        evalMap[idx - 1].parameter = optParamValue;
        evalMap[idx - 1].score = optScore.value();
        evalMap[idx - 1].foundOptimal = optBoolValue;
        LLVM_DEBUG(dbgs() << "[RAAS] Setting configuration "
                          << techniqueNameMap.at(AT) << "|" << idx
                          << " from function " << fnName << " to ["
                          << optParamValue << "|" << optScore.value() << "|"
                          << optBoolValue << "]\n";);
      }
    }
    opPerFunction.updatedInLastEvaluation = true;
  }
  if (!foundFn)
    LLVM_DEBUG(
        dbgs()
        << "[RAAS] " << functionName
        << " not found on JSON restoration file. Proceeding without it.\n");
  // assert(foundFn && "Function was not found in restore data JSON!\n");
}

template <typename T>
std::string formatText(const int width, const char fill, T value) {
  std::stringstream ss;

  ss << std::left << std::setw(width) << std::setfill(fill) << value;
  return ss.str();
}

std::string SimpleEvaluator::getRankedConfigurations(bool csv_format) {
  std::sort(
      opportunitiesWrapper.begin(), opportunitiesWrapper.end(),
      [](SimpleEvaluator::approximationInfo a,
         SimpleEvaluator::approximationInfo b) { return a.score > b.score; });

  std::stringstream ss;
  if (csv_format) {
    ss << "score,function,technique,parameter,speedup" << std::endl;
    for (auto &aa : opportunitiesWrapper)
      ss << aa.score << "," << aa.parent->functionName << ","
         << techniqueNameMap.at(aa.AT) << "," << aa.idealConfig.second << ","
         << aa.idealConfig.first << std::endl;

    return ss.str();
  }

  size_t maxFnSize = 0;
  for (auto &II : oppPerFunctionMap)
    maxFnSize = std::max(II.first().size(), maxFnSize);

  maxFnSize += 2;

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
    ss << formatText(9, fill, aa.idealConfig.second) << " | ";
    ss << formatText(9, fill, aa.idealConfig.first) << " | ";
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
