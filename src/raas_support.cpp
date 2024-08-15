#include <pybind11/pybind11.h>

static double time_begin;
static inline void RAAS_roi_begin()  {
  struct timeval t;
  gettimeofday(&t, NULL);
  time_begin = (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

static inline void RAAS_roi_end()  {
  struct timeval t;
  gettimeofday(&t, NULL);
  double time_end = (double)t.tv_sec + (double)t.tv_usec * 1e-6;
  double delta = time_end - time_begin;
  FILE *f = fopen(".jit_time.txt", "w");
  fprintf(f, "%lf", delta);
  fclose(f);
}

extern "C" void $raas_evaluation() __attribute__((weak));

PYBIND11_MODULE(raas_support, m) {
  m.doc() = "RAAS support functions";

  m.def("RAAS_roi_begin", &RAAS_roi_begin, "Marks beggining of region-of-interest time");
  m.def("RAAS_roi_end", &RAAS_roi_end, "Marks end of region-of-interest time");
  m.def("RAAS_eval", &$raas_evaluation, "Gives context back to the JIT for evaluation");
}
