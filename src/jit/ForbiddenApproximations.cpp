#include "Core.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

ForbiddenApproximations::ForbiddenApproximations(StringRef jsonFile) {
  static llvm::ExitOnError exitOnErr;
  auto jsonStruct = exitOnErr(json::parse(jsonFile));

  // function -> tech[pair]
  for (auto &funSkippablePair : *jsonStruct.getAsObject()) {
    std::string functionName = funSkippablePair.getFirst().str();
    auto &skippableApproxForFunction = skippableApprox[functionName];
    // tech -> pair
    for (auto &techOppPair : *funSkippablePair.getSecond().getAsObject()) {
      auto technique = stringToTechType.at(techOppPair.getFirst().str());
      // first -> opportunity number
      // second -> max approx allowed
      for (auto &configPair : *techOppPair.getSecond().getAsObject()) {
        unsigned oppNumber = std::stoi(configPair.getFirst().str());
        unsigned maxAllowed = configPair.getSecond().getAsNumber().value();
        skippableApproxForFunction[technique].insert({oppNumber, maxAllowed});
      }
    }
  }
}

ForbiddenApproximations::opportunitiesPerFunction
ForbiddenApproximations::getForbiddenApproxForFunction(
    std::string functionName) {
  return skippableApprox[functionName];
}

ErrorOr<std::unique_ptr<ForbiddenApproximations>>
ForbiddenApproximations::Create(StringRef jsonFileName) {
  auto buffer = MemoryBuffer::getFileOrSTDIN(jsonFileName);
  if (!buffer)
    return buffer.getError();

  return std::make_unique<ForbiddenApproximations>(
      ForbiddenApproximations(buffer.get()->getBuffer()));
}

const std::map<std::string, approxTechnique>
    ForbiddenApproximations::stringToTechType = {
        {"lperf", LPERF}, {"fapp", FAP}, {"lpar", LPAR}};
