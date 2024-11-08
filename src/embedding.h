#include <string>

bool start_JIT(std::string evalBCFile, std::string appModulesFile,
               std::string preciseModulesFile = "",
               std::string forbiddenApproximationsFile = "");

void printOpportunities(bool csv_format = false);

int initializeEmbedding(int argc, char *argv[], int &i);
void endEmbedding();
