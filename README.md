# RAAS - Runtime Adaptive Approximation System

RAAS is a framework for runtime code approximation that works with LLVM IR modules, featuring adaptive control and evaluation-driven approximation tuning.
RAAS works by trying approximations at runtime and using an automatic evaluation system to select an optimal configuration based on heuristic evaluation.

## Key Features
- **LLVM IR-based**: Operates on compiled modules with precise/approximate partitions
- **Runtime Adaptation**: Fully Automatic runtime decision making for approximations

## Installation
An image container to be built with [apptainer](https://github.com/apptainer/apptainer.git) is provided for testing at 'containers/'

### Prerequisites
- [Custom LLVM fork (18.0)](https://github.com/lucasreis1/llvm-project/tree/pr/improve-symbol-mapping)
- Python 3.8+
- CMake ≥ 3.15
- Pybind11 (auto-fetched during build)

### Build & Install
```bash
git clone https://github.com/lucasreis1/RAAS.git
cd raas
mkdir build 
cmake -B build . -DLLVM_DIR=/path/to/llvm/cmake
cmake --build build
```

### User Requirements
The framework require users to provide an application-specific evaluation file that provides information on how to measure error rates. This file is an IR module that must contain two functions:
```llvm
; Stores reference outputs
define void @storeOriginal()

; Returns error metric 
define double @compare()
```

We also require light instrumentation on the application code that signals:
* Region-of-Interest demarcation: the region of interest that signals which sections are valid for approximation. These are often the computation-side of code (ignoring input and output processing)
* Evaluation time: the point in the code where it's safe to give context back to the framework to start the evaluation stage. This is often done after output processing

We provide a header file with implementation to be used for C/C++ applications ate `include/instrumentation.h`.

An example to apply to a generic application follows:
```c
#include "instrumentation.h"

int main() {
    for (int i = 0 ; i < n ; ++i) {
        read_inputs();
        RAAS_roi_begin();
        do_compute();
        RAAS_roi_end();
        output_processing();
        RAAS_evaluate_output();
    }
    return 0;
}
```

### Usage
```bash
    ./raas \
      --approx-modules <IR_file1.bc,IR_file2.bc> # Modules that will be approximated \
      --precise-modules <base_code.bc> # Required modules that will be compiled as-is \
      --evaluation-file <evaluator.bc> \
      --error-limit 0.1  # Max tolerable error (10%) \
      <application-args>
```
