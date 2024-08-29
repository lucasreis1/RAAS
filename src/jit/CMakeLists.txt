llvm_map_components_to_libnames(llvm_libs support core orcjit native irreader)

# by default, LLVM is built without RTTI. Our structures must be, too
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})


add_compile_definitions(RUNTIME_PATH="${LLVM_DIR}/../../clang/18/lib/x86_64-unknown-linux-gnu/liborc_rt.a")
add_compile_definitions(EMBEDDING)

add_library (JIT SHARED JIT.cpp ApproxLayer.cpp ForbiddenApproximations.cpp EvaluationSystem.cpp SimpleEvaluator.cpp misc/CustomDemangler.cpp)

#target_link_libraries(JIT ${llvm_libs} Passes) 
target_link_libraries(JIT ${llvm_libs} Passes "$<LINK_LIBRARY:WHOLE_ARCHIVE,approx_gemm>") 

add_subdirectory(passes)
