#include "common.h"
#include "mkl_blas.h"
#include <vector>

extern "C" {
  void approx_gemm_f16f16f32(char transa, char transb, int m, int n, int k, float alpha, 
                             const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    std::vector<MKL_F16> a_f16(m * k, 0);
    std::vector<MKL_F16> b_f16(k * n, 0);

    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < k ; ++j)
        a_f16[i * k + j] = f2h(a[i * k + j]);
    for (int i = 0 ; i < k ; ++i)
      for (int j = 0 ; j < n ; ++j)
        b_f16[i * n + j] = f2h(b[i * n + j]);

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;
    MKL_INT lda_ = lda;
    MKL_INT ldb_ = ldb;
    MKL_INT ldc_ = ldc;
    gemm_f16f16f32(&transa, &transb, &m_, &n_, &k_, &alpha, a_f16.data(), &lda_, b_f16.data(), &ldb_, &beta, c, &ldc_);
  }

  void approx_gemm_bf16bf16f32(char transa, char transb, int m, int n, int k, float alpha, 
                             const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    std::vector<MKL_BF16> a_bf16(m * k, 0);
    std::vector<MKL_BF16> b_bf16(k * n, 0);

    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < k ; ++j)
        a_bf16[i * k + j] = f2b(a[i * k + j]);
    for (int i = 0 ; i < k ; ++i)
      for (int j = 0 ; j < n ; ++j)
        b_bf16[i * n + j] = f2b(b[i * n + j]);

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;
    MKL_INT lda_ = lda;
    MKL_INT ldb_ = ldb;
    MKL_INT ldc_ = ldc;
    gemm_bf16bf16f32(&transa, &transb, &m_, &n_, &k_, &alpha, a_bf16.data(), &lda_, b_bf16.data(), &ldb_, &beta, c, &ldc_);
  }

  void approx_gemm_hgemm(char transa, char transb, int m, int n, int k, float alpha, 
                             const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    std::vector<MKL_F16> a_f16(m * k, 0);
    std::vector<MKL_F16> b_f16(k * n, 0);
    std::vector<MKL_F16> c_f16(m * n, 0);

    MKL_F16 alpha_f16 = f2h(alpha);
    MKL_F16 beta_f16 = f2h(beta);

    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < k ; ++j)
        a_f16[i * k + j] = f2h(a[i * k + j]);
    for (int i = 0 ; i < k ; ++i)
      for (int j = 0 ; j < n ; ++j)
        b_f16[i * n + j] = f2h(b[i * n + j]);
    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < n ; ++j)
        c_f16[i * n + j] = f2h(c[i * n + j]);

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;
    MKL_INT lda_ = lda;
    MKL_INT ldb_ = ldb;
    MKL_INT ldc_ = ldc;
    hgemm(&transa, &transb, &m_, &n_, &k_, &alpha_f16, a_f16.data(), &lda_, b_f16.data(), &ldb_, &beta_f16, c_f16.data(), &ldc_);

    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < n ; ++j)
        c[i * n + j] = h2f(c_f16[i * n + j]);
  }

  void approx_gemm_s16s16s32(char transa, char transb, int m, int n, int k, float alpha, 
                             const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    std::vector<MKL_INT16> a_s16(m * k, 0);
    std::vector<MKL_INT16> b_s16(k * n, 0);
    std::vector<MKL_INT32> c_s32(m * n, 0);

#define OFFSET 1
#define OFFSET_SQUARED 1
    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < k ; ++j)
        a_s16[i * k + j] = static_cast<MKL_INT16>(a[i * k + j] * OFFSET); 
    for (int i = 0 ; i < k ; ++i)
      for (int j = 0 ; j < n ; ++j)
        b_s16[i * n + j] = static_cast<MKL_INT16>(b[i * n + j] * OFFSET);
    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < n ; ++j)
        c_s32[i * n + j] = static_cast<MKL_INT32>(c[i * n + j] * OFFSET * OFFSET);

    char offsetc = 'F';
    MKL_INT16 ao = 0, bo = 0;
    MKL_INT32 co = 0;

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;
    MKL_INT lda_ = lda;
    MKL_INT ldb_ = ldb;
    MKL_INT ldc_ = ldc;
    gemm_s16s16s32(&transa, &transb, &offsetc, &m_, &n_, &k_, 
                   &alpha, a_s16.data(), &lda_, &ao, b_s16.data(), &ldb_, &bo, 
                   &beta, c_s32.data(), &ldc_, &co);

    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < n ; ++j)
        c[i * n + j] = static_cast<float>(c_s32[i * n + j])/ OFFSET_SQUARED;
  }

  void approx_gemm_s8u8s32(char transa, char transb, int m, int n, int k, float alpha, 
                             const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    std::vector<MKL_INT8> a_s8(m * k, 0);
    std::vector<MKL_UINT8> b_u8(k * n, 0);
    std::vector<MKL_INT32> c_s32(m * n, 0);

#define OFFSET_8 1
#define OFFSET_8_SQUARED  OFFSET_8 * OFFSET_8
    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < k ; ++j)
        a_s8[i * k + j] = static_cast<MKL_INT8>(a[i * k + j] * OFFSET_8);
    for (int i = 0 ; i < k ; ++i)
      for (int j = 0 ; j < n ; ++j)
        b_u8[i * n + j] = static_cast<MKL_UINT8>(b[i * n + j] * OFFSET_8);
    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < n ; ++j)
        c_s32[i * n + j] = static_cast<MKL_INT32>(c[i * n + j] * OFFSET_8_SQUARED);

    char offsetc = 'F';
    MKL_INT8 ao = 0, bo = 0;
    MKL_INT32 co = 0;
    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;
    MKL_INT lda_ = lda;
    MKL_INT ldb_ = ldb;
    MKL_INT ldc_ = ldc;
    gemm_s8u8s32(&transa, &transb, &offsetc, &m_, &n_, &k_, 
                   &alpha, a_s8.data(), &lda_, &ao, b_u8.data(), &ldb_, &bo, 
                   &beta, c_s32.data(), &ldc_, &co);

    for (int i = 0 ; i < m ; ++i)
      for (int j = 0 ; j < n ; ++j)
        c[i * n + j] = static_cast<float>(c_s32[i * n + j])/OFFSET_8_SQUARED;
  }
}
