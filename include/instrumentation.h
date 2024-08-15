#include <stdio.h>
#include <sys/time.h>

#ifndef INSTRUMENTATION_H
#define INSTRUMENTATION_H

// function wrappers to hide the pointer parameter
#ifdef __cplusplus
extern "C" void $raas_evaluation();
#else
void $raas_evaluation();
#endif

inline __attribute__((always_inline)) void RAAS_evaluate_output() {
  $raas_evaluation();
}
#endif

static double RAAS_time_begin;

static inline void RAAS_roi_begin() {
  struct timeval t;
  gettimeofday(&t, NULL);
  RAAS_time_begin = (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

static inline void RAAS_roi_end() {
  struct timeval t;
  gettimeofday(&t, NULL);
  double time_end = (double)t.tv_sec + (double)t.tv_usec * 1e-6;
  double delta = time_end - RAAS_time_begin;

  FILE *f = fopen(".jit_time.txt", "w");
  fprintf(f, "%lf", delta);
  fclose(f);
}
