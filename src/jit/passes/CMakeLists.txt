# by default, LLVM is built without RTTI. Our structures must be, too
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

add_library(Passes SHARED Passes.cpp LoopPerforation.cpp FunctionApprox.cpp GEMMApproximation.cpp ApproximationAnalysis.cpp)

add_library(instrumentFn SHARED InstrumentApprox.cpp)
add_library(GEMMApprox SHARED GEMMApproximation.cpp)
add_library(ApproxOpportunityList SHARED GetApproximableOpportunities.cpp)
target_link_libraries(instrumentFn Passes LLVMSupport)
target_link_libraries(GEMMApprox Passes LLVMSupport)
target_link_libraries(ApproxOpportunityList Passes)
