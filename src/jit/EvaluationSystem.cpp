#include "Core.h"
#include "passes/Passes.h"
#include <cxxabi.h>
#include <iostream>
#include <fstream>

EvaluationSystem::EvaluationSystem(
    std::unique_ptr<ConfigurationEvaluation> eval, bool trainingMode,
    std::string programName)
    : evaluator(std::move(eval)), trainingMode(trainingMode), programName(programName) {
  assert((trainingMode && programName != "" || !trainingMode) &&
         "Asked to store JSON but program name not informed!\n");
}

void EvaluationSystem::addFunction(const Function &F) {
  auto functionName = F.getName();
  if (functionOpportunityList.find(functionName) !=
      functionOpportunityList.end()) {
    llvm_unreachable("function added more than once!");
  }
  // create the approximation rate map
  functionOpportunityList.insert({functionName, populateApproximationRate(F)});
}

bool EvaluationSystem::findFunction(const std::string FunctionName) {
  return functionOpportunityList.find(FunctionName) !=
         functionOpportunityList.end();
}

bool EvaluationSystem::findFunction(const Function &F) {
  return findFunction(F.getName().str());
}

std::string EvaluationSystem::getCombinationFromConfiguration(
    configurationPerTechniqueMap configuration) {
  std::string combination = "";
  //  map function -> combination
  for (auto const &II : configuration) {
    for (auto const &el : II.second) {
      combination += std::to_string(II.first) + "/" + std::to_string(el) + ";";
    }
  }
  if (combination.size())
    combination.pop_back();
  return combination;
}

configurationPerTechniqueMap
EvaluationSystem::populateApproximationRate(const Function &F) {
  configurationPerTechniqueMap M;

  // build the approx list for each technique
  for (auto &apprPass : passlist::Passes) {
    unsigned tam = apprPass->searchApproximableCalls(F);
    if (tam)
      M[apprPass->T] = std::vector<int>(tam);
  }
  // create a map for each function with extra information inside the
  // evaluator class
  evaluator->buildApproximationStructures(F.getName(), M);

  if (trainingMode) {
    std::string fileName = programName + ".json";
    evaluator->restoreStateFromJSON(F.getName(), fileName);
  }

  return M;
}

// checks if the function is approximable, storing results on data
// structures
bool EvaluationSystem::isApproximable(const Function &F) {
  return passlist::isApproximable(F);
}

void EvaluationSystem::printApproximationOpportunities(const Function &F) {
  int status;
  const char *demangled =
      abi::__cxa_demangle(F.getName().str().c_str(), nullptr, nullptr, &status);
  if (status == -2)
    fprintf(stderr, "============ On function %s ============\n",
            F.getName().str().c_str());
  else
    fprintf(stderr, "============ On function %s ============\n", demangled);
  for (auto &apprPass : passlist::Passes)
    apprPass->printApproximationOpportunities(F);
}

void EvaluationSystem::incrementPreciseTime(double precTime) {
  evaluator->APQ.incrementPreciseTime(precTime);
}

void EvaluationSystem::updateQualityValues(double error, double time) {
  evaluator->APQ.updateValues(error, time);
}

StringMap<configurationPerTechniqueMap>
EvaluationSystem::updateSuggestedConfigurations() {
  StringMap<configurationPerTechniqueMap> toReapprox;
  // calls our evaluator to do its heuristic and update suggested configurations
  evaluator->updateSuggestedConfigurations();
  if (trainingMode)
    this->storeConfigurationToFile();

  // iterate over all approximable functions and ask the evaluator for
  // configurations that have changed in the last iteration
  for (auto &MI : functionOpportunityList) {
    auto functionName = MI.first().str();
    if (evaluator->wasUpdatedInLastEvaluation(functionName)) {
      evaluator->unmarkAsUpdated(functionName);
      auto configuration = evaluator->getSuggestedConfiguration(functionName);
      MI.second = configuration;
      // store the ones that have seen change in last evaluation
      toReapprox.insert(std::make_pair(functionName, configuration));
    }
  }

  return toReapprox;
}

configurationPerTechniqueMap
EvaluationSystem::getConfigurationForFunction(StringRef functionName) {
  auto II = functionOpportunityList.find(functionName);
  if (II == functionOpportunityList.end())
    llvm_unreachable("function not recognized by evaluation system");

  return II->second;
}

double EvaluationSystem::getLastSpeedup() {
  return evaluator->APQ.getSpeedups().second;
}

double EvaluationSystem::getLastError() {
  return evaluator->APQ.getErrors().second;
}

void EvaluationSystem::printRankedOpportunities(bool csv_format) {
  std::cout << evaluator->getRankedConfigurations(csv_format);
  return;
}

void EvaluationSystem::storeConfigurationToFile() {
  auto jsonStr = evaluator->getJSONConfiguration();
  std::string fileName = programName + ".json";
  std::ofstream f(fileName);
  if (f.is_open()) {
    f << jsonStr;
    f.close();
  }
  else {
    fprintf(stderr, "Unable to open file %s\n", fileName.c_str());
  }
}
