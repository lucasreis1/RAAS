#include "jit/JIT.h"
#include "embedding.h"
#include <Python.h>
#include <cpython/initconfig.h>
#include <dlfcn.h>

PyStatus init_python(const char *program_name, int argc, char **argv) {
  PyStatus status;

  PyConfig config;
  PyConfig_InitPythonConfig(&config);

  status = PyConfig_SetBytesString(&config, &config.run_filename, program_name);
  if (PyStatus_Exception(status)) {
    PyConfig_Clear(&config);
    return status;
  }

  status = PyConfig_SetBytesArgv(&config, argc, argv);
  if (PyStatus_Exception(status)) {
    PyConfig_Clear(&config);
    return status;
  }

  // check if we are in a venv and set things accordingly
  auto env_path = getenv("VIRTUAL_ENV");
  if (env_path) {
    auto executable_path = std::string(env_path) + "/bin/python";
    PyConfig_SetBytesString(&config, &config.executable, executable_path.c_str());
  }

  status = Py_InitializeFromConfig(&config);
  return status;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "missing argument for script name!\n");
    fprintf(stderr,
            "Usage: ./a.out --eval_mod <evaluation_mod.bc> <script_file.py> <args...>\n");
    return 1;
  }

  auto *lib = dlopen("libtorch_cpu.so", RTLD_LAZY | RTLD_GLOBAL);
  if (lib == nullptr) {
    fprintf(stderr, "dlopen libtorch_cpu_approx failure: %s\n", dlerror());
    return 1;
  }

  int i;
  auto ret = initializeEmbedding(argc, argv, i);

  PyStatus status;
  char **argv_mod = nullptr;

  // if we could not start the JIT, try to work with all args
  if (not ret) {
    status = init_python(argv[1], argc, argv);
  } else {
    int lim = argc - (i - 1);
    argv_mod = new char *[lim];
    // skip argv[1] (evaluation module)
    argv_mod[0] = argv[0];
    for (int j = 1; j < lim; ++j, ++i)
      argv_mod[j] = argv[i];

    // auto status = init_python(argv[1], argc, argv);
    status = init_python(argv_mod[1], lim, argv_mod);
  }

  if (PyStatus_Exception(status)) {
    if (PyStatus_IsExit(status))
      return status.exitcode;

    Py_ExitStatusException(status);
  }

  PyObject *pName;

  if (argv_mod != nullptr)
    delete[] argv_mod;
  auto ret_val = Py_RunMain();

  endEmbedding();

  dlclose(lib);

  // if (ret)
  //   ExitOnErr(J->deinitializePlatform());
  return ret_val;
}
