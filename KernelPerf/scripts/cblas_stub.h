#pragma once

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;
#ifdef __cplusplus
extern "C" {
#endif
void cblas_sgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
    float, const float*, int, const float*, int, float, float*, int);
void cblas_dgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
    double, const double*, int, const double*, int, double, double*, int);
void cblas_cgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
    const void*, const void*, int, const void*, int, const void*, void*, int);
void cblas_zgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
    const void*, const void*, int, const void*, int, const void*, void*, int);
#ifdef __cplusplus
}
#endif
