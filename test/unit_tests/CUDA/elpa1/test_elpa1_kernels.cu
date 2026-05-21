//    Copyright 2026, MPCDF
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>

// Unit tests for kernels in src/elpa1/GPU/CUDA/elpa1_cuda.cu:
//   cuda_copy_real_part_to_q_*_complex_kernel  (double_complex, float_complex)
//   cuda_zero_skewsymmetric_q_*_kernel          (double, float)
//   cuda_copy_skewsymmetric_second_half_q_*     (double, float; plus and minus)
//   cuda_copy_skewsymmetric_first_half_q_*      (double, float)
//   cuda_get_skewsymmetric_second_half_q_*      (double, float)
//   cuda_put_skewsymmetric_second_half_q_*      (double, float)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <type_traits>
#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

#include "../../../../src/elpa1/GPU/CUDA/elpa1_cuda.cu"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---- dispatcher overloads for cuda_zero_skewsymmetric_q ----

static void call_zero_skewsymmetric_q(double *q, int *mR, int *mC, cudaStream_t s)
{ cuda_zero_skewsymmetric_q_double_FromC(q, mR, mC, s); }

static void call_zero_skewsymmetric_q(float *q, int *mR, int *mC, cudaStream_t s)
{ cuda_zero_skewsymmetric_q_float_FromC(q, mR, mC, s); }

#ifdef WANT_HALF_PRECISION_REAL
static void call_zero_skewsymmetric_q(__half *q, int *mR, int *mC, cudaStream_t s)
{ cuda_zero_skewsymmetric_q_half_FromC(q, mR, mC, s); }
#endif

// ---- dispatcher overloads for cuda_copy_skewsymmetric_second_half_q ----

static void call_copy_second_half_q(double *q, int *i, int *mR, int *mC, int *sign, cudaStream_t s)
{ cuda_copy_skewsymmetric_second_half_q_double_FromC(q, i, mR, mC, sign, s); }

static void call_copy_second_half_q(float *q, int *i, int *mR, int *mC, int *sign, cudaStream_t s)
{ cuda_copy_skewsymmetric_second_half_q_float_FromC(q, i, mR, mC, sign, s); }

#ifdef WANT_HALF_PRECISION_REAL
static void call_copy_second_half_q(__half *q, int *i, int *mR, int *mC, int *sign, cudaStream_t s)
{ cuda_copy_skewsymmetric_second_half_q_half_FromC(q, i, mR, mC, sign, s); }
#endif

// ---- dispatcher overloads for cuda_copy_skewsymmetric_first_half_q ----

static void call_copy_first_half_q(double *q, int *i, int *mR, int *mC, int *sign, cudaStream_t s)
{ cuda_copy_skewsymmetric_first_half_q_double_FromC(q, i, mR, mC, sign, s); }

static void call_copy_first_half_q(float *q, int *i, int *mR, int *mC, int *sign, cudaStream_t s)
{ cuda_copy_skewsymmetric_first_half_q_float_FromC(q, i, mR, mC, sign, s); }

#ifdef WANT_HALF_PRECISION_REAL
static void call_copy_first_half_q(__half *q, int *i, int *mR, int *mC, int *sign, cudaStream_t s)
{ cuda_copy_skewsymmetric_first_half_q_half_FromC(q, i, mR, mC, sign, s); }
#endif

// ---- dispatcher overloads for cuda_get_skewsymmetric_second_half_q ----

static void call_get_second_half_q(double *q, double *q2, int *mR, int *mC, cudaStream_t s)
{ cuda_get_skewsymmetric_second_half_q_double_FromC(q, q2, mR, mC, s); }

static void call_get_second_half_q(float *q, float *q2, int *mR, int *mC, cudaStream_t s)
{ cuda_get_skewsymmetric_second_half_q_float_FromC(q, q2, mR, mC, s); }

#ifdef WANT_HALF_PRECISION_REAL
static void call_get_second_half_q(__half *q, __half *q2, int *mR, int *mC, cudaStream_t s)
{ cuda_get_skewsymmetric_second_half_q_half_FromC(q, q2, mR, mC, s); }
#endif

// ---- dispatcher overloads for cuda_put_skewsymmetric_second_half_q ----

static void call_put_second_half_q(double *q, double *q2, int *mR, int *mC, cudaStream_t s)
{ cuda_put_skewsymmetric_second_half_q_double_FromC(q, q2, mR, mC, s); }

static void call_put_second_half_q(float *q, float *q2, int *mR, int *mC, cudaStream_t s)
{ cuda_put_skewsymmetric_second_half_q_float_FromC(q, q2, mR, mC, s); }

#ifdef WANT_HALF_PRECISION_REAL
static void call_put_second_half_q(__half *q, __half *q2, int *mR, int *mC, cudaStream_t s)
{ cuda_put_skewsymmetric_second_half_q_half_FromC(q, q2, mR, mC, s); }
#endif

// ============================================================
// Test: cuda_copy_real_part_to_q_*_complex_kernel
//
// Parameters: matrixRows=4, l_rows=3, l_cols_nev=2
// q_real[k] = k+1  (8 elements, col-major with leading dim matrixRows=4)
// q initialised to sentinel {99, 99}
//
// For valid (row < l_rows=3, col < l_cols_nev=2):
//   index = row + 4*col
//   q[index].x = q_real[index]   q[index].y = 0
// Row 3 (index 3 and 7) untouched.
// ============================================================
template <typename TC>
static int run_copy_real_part_to_q_test(const char *type_name)
{
    using TR = typename cuda_real_type<TC>::type;

    int matrixRows = 4, l_rows = 3, l_cols_nev = 2;
    const int n = matrixRows * l_cols_nev;  // 8

    TR  h_qreal[8];
    TC  h_q[8];
    for (int k = 0; k < n; k++) {
        h_qreal[k] = (TR)(k + 1);
        h_q[k].x   = (TR)99;
        h_q[k].y   = (TR)99;
    }

    TR *d_qreal;  TC *d_q;
    CUDA_CHECK(cudaMalloc(&d_qreal, n * sizeof(TR)));
    CUDA_CHECK(cudaMalloc(&d_q,     n * sizeof(TC)));
    CUDA_CHECK(cudaMemcpy(d_qreal, h_qreal, n * sizeof(TR), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q,     h_q,     n * sizeof(TC), cudaMemcpyHostToDevice));

    // Launchers use C99 _Complex pointers; cast from cuComplex is safe (same layout).
    if constexpr (std::is_same_v<TC, cuDoubleComplex>)
        cuda_copy_real_part_to_q_double_complex_FromC(
            (double _Complex*)d_q, (double*)d_qreal, &matrixRows, &l_rows, &l_cols_nev, nullptr);
    else
        cuda_copy_real_part_to_q_float_complex_FromC(
            (float _Complex*)d_q, (float*)d_qreal, &matrixRows, &l_rows, &l_cols_nev, nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_q, d_q, n * sizeof(TC), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_qreal));
    CUDA_CHECK(cudaFree(d_q));

    int failures = 0;
    for (int col = 0; col < l_cols_nev; col++) {
        for (int row = 0; row < matrixRows; row++) {
            int idx = row + matrixRows * col;
            if (row < l_rows) {
                if (h_q[idx].x != h_qreal[idx] || h_q[idx].y != (TR)0) failures++;
            } else {
                // sentinel must be untouched
                if (h_q[idx].x != (TR)99 || h_q[idx].y != (TR)99) failures++;
            }
        }
    }

    printf("  cuda_copy_real_part_to_q_complex_kernel <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: cuda_zero_skewsymmetric_q_*_kernel
//
// Parameters: matrixRows=3, matrixCols=2
// q has 3 * 2*matrixCols = 12 elements, filled 1..12.
// Kernel zeroes q[row + 3*(col + 2)] for row<3, col<2
//   → indices 6,7,8 (col=0) and 9,10,11 (col=1).
// Lower half q[0..5] must be unchanged.
// ============================================================
template <typename T>
static int run_zero_skewsymmetric_q_test(const char *type_name)
{
    int matrixRows = 3, matrixCols = 2;
    const int n = matrixRows * 2 * matrixCols;  // 12

    T h_q[12], h_result[12];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q;
    CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, n * sizeof(T), cudaMemcpyHostToDevice));

    call_zero_skewsymmetric_q(d_q, &matrixRows, &matrixCols, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_q, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));

    // Build expected: upper half zeroed, lower half unchanged
    T expected[12];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * (col + matrixCols)] = (T)0;

    int failures = 0;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  cuda_zero_skewsymmetric_q_kernel        <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: cuda_copy_skewsymmetric_second_half_q_*  (plus and minus)
//
// Parameters: matrixRows=4, matrixCols=3, i=2 (1-based row)
// q has 4 * 2*3 = 24 elements, filled 1..24.
// Kernel operates on row (i-1)=1 (0-based), col in [3,6):
//   index    = 1 + 4*col
//   indexLow = 1 + 4*(col - 3)
//   _plus:  q[index] =  q[indexLow];  q[indexLow] = 0
//   _minus: q[index] = -q[indexLow];  q[indexLow] = 0
// col=3 → index 13, indexLow 1
// col=4 → index 17, indexLow 5
// col=5 → index 21, indexLow 9
// ============================================================
template <typename T>
static int run_copy_second_half_q_one(const char *type_name, int sign)
{
    int matrixRows = 4, matrixCols = 3, i = 2;
    const int n = matrixRows * 2 * matrixCols;  // 24

    T h_q[24];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q;
    CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_second_half_q(d_q, &i, &matrixRows, &matrixCols, &sign, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[24];
    CUDA_CHECK(cudaMemcpy(h_result, d_q, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));

    // Build expected
    T expected[24];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = matrixCols; col < 2 * matrixCols; col++) {
        int idx     = (i-1) + matrixRows * col;
        int idxLow  = (i-1) + matrixRows * (col - matrixCols);
        expected[idx]    = (sign == 1) ? h_q[idxLow] : -h_q[idxLow];
        expected[idxLow] = (T)0;
    }

    int failures = 0;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  cuda_copy_second_half_q_kernel          <%s> sign=%d: %s\n",
           type_name, sign, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

template <typename T>
static int run_copy_second_half_q_test(const char *type_name)
{
    int failures = 0;
    failures += run_copy_second_half_q_one<T>(type_name, 1);   // _plus
    failures += run_copy_second_half_q_one<T>(type_name, 0);   // _minus
    return failures;
}

// ============================================================
// Test: cuda_copy_skewsymmetric_first_half_q_*  (minus only)
//
// Parameters: matrixRows=4, matrixCols=3, i=2 (1-based row)
// q has 4 * 3 = 12 elements, filled 1..12.
// Kernel negates row (i-1)=1 (0-based) for col in [0,3):
//   index = 1 + 4*col   → indices 1, 5, 9
//   q[index] = -q[index]
// All other elements unchanged.
// ============================================================
template <typename T>
static int run_copy_first_half_q_test(const char *type_name)
{
    int matrixRows = 4, matrixCols = 3, i = 2;
    const int n = matrixRows * matrixCols;  // 12
    int sign = 0;  // only minus variant exists; value is unused by launcher

    T h_q[12];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q;
    CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_first_half_q(d_q, &i, &matrixRows, &matrixCols, &sign, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[12];
    CUDA_CHECK(cudaMemcpy(h_result, d_q, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));

    // Build expected: negate row i-1 across all cols
    T expected[12];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        expected[(i-1) + matrixRows * col] = -h_q[(i-1) + matrixRows * col];

    int failures = 0;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  cuda_copy_first_half_q_kernel           <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: cuda_get_skewsymmetric_second_half_q_*
//
// Parameters: matrixRows=3, matrixCols=2
// q[k] = k+1 (12 elements)
// q_2 should receive:  q_2[row + 3*col] = q[row + 3*(col+2)]
//   q_2[0..2] = q[6..8] = 7,8,9
//   q_2[3..5] = q[9..11] = 10,11,12
// ============================================================
template <typename T>
static int run_get_second_half_q_test(const char *type_name)
{
    int matrixRows = 3, matrixCols = 2;
    const int n_q  = matrixRows * 2 * matrixCols;  // 12
    const int n_q2 = matrixRows * matrixCols;       // 6

    T h_q[12];
    for (int k = 0; k < n_q; k++) h_q[k] = (T)(k + 1);

    T *d_q, *d_q2;
    CUDA_CHECK(cudaMalloc(&d_q,  n_q  * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_q2, n_q2 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, n_q * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_q2, 0, n_q2 * sizeof(T)));

    call_get_second_half_q(d_q, d_q2, &matrixRows, &matrixCols, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_q2[6];
    CUDA_CHECK(cudaMemcpy(h_q2, d_q2, n_q2 * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_q2));

    // Build expected: q_2[row + 3*col] = q[row + 3*(col+2)]
    T expected[6];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * col] = h_q[row + matrixRows * (col + matrixCols)];

    int failures = 0;
    for (int k = 0; k < n_q2; k++)
        if (h_q2[k] != expected[k]) failures++;

    printf("  cuda_get_second_half_q_kernel           <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: cuda_put_skewsymmetric_second_half_q_*
//
// Parameters: matrixRows=3, matrixCols=2
// q[k] = k+1, q_2[k] = k+101
// After kernel: q[row + 3*(col+2)] = q_2[row + 3*col]
//   q[6..8]  = q_2[0..2] = 101,102,103
//   q[9..11] = q_2[3..5] = 104,105,106
// Lower half q[0..5] unchanged.
// ============================================================
template <typename T>
static int run_put_second_half_q_test(const char *type_name)
{
    int matrixRows = 3, matrixCols = 2;
    const int n_q  = matrixRows * 2 * matrixCols;  // 12
    const int n_q2 = matrixRows * matrixCols;       // 6

    T h_q[12], h_q2[6];
    for (int k = 0; k < n_q;  k++) h_q[k]  = (T)(k + 1);
    for (int k = 0; k < n_q2; k++) h_q2[k] = (T)(k + 101);

    T *d_q, *d_q2;
    CUDA_CHECK(cudaMalloc(&d_q,  n_q  * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_q2, n_q2 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q,  h_q,  n_q  * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q2, h_q2, n_q2 * sizeof(T), cudaMemcpyHostToDevice));

    call_put_second_half_q(d_q, d_q2, &matrixRows, &matrixCols, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[12];
    CUDA_CHECK(cudaMemcpy(h_result, d_q, n_q * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_q2));

    // Build expected: upper half replaced by q_2, lower half unchanged
    T expected[12];
    for (int k = 0; k < n_q; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * (col + matrixCols)] = h_q2[row + matrixRows * col];

    int failures = 0;
    for (int k = 0; k < n_q; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  cuda_put_second_half_q_kernel           <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
int main(void)
{
    int failures = 0;

    printf("=== Unit tests for elpa1_cuda.cu kernels ===\n\n");
    printf("Testing cuda_copy_real_part_to_q_complex_kernel:\n");
    failures += run_copy_real_part_to_q_test<cuDoubleComplex>("cuDoubleComplex");
    failures += run_copy_real_part_to_q_test<cuFloatComplex> ("cuFloatComplex");

    printf("\nTesting cuda_zero_skewsymmetric_q_kernel:\n");
    failures += run_zero_skewsymmetric_q_test<double>("double");
    failures += run_zero_skewsymmetric_q_test<float> ("float");

    printf("\nTesting cuda_copy_skewsymmetric_second_half_q_kernel:\n");
    failures += run_copy_second_half_q_test<double>("double");
    failures += run_copy_second_half_q_test<float> ("float");

    printf("\nTesting cuda_copy_skewsymmetric_first_half_q_kernel:\n");
    failures += run_copy_first_half_q_test<double>("double");
    failures += run_copy_first_half_q_test<float> ("float");

    printf("\nTesting cuda_get_skewsymmetric_second_half_q_kernel:\n");
    failures += run_get_second_half_q_test<double>("double");
    failures += run_get_second_half_q_test<float> ("float");

    printf("\nTesting cuda_put_skewsymmetric_second_half_q_kernel:\n");
    failures += run_put_second_half_q_test<double>("double");
    failures += run_put_second_half_q_test<float> ("float");

#ifdef WANT_HALF_PRECISION_REAL
    printf("\nTesting cuda_zero_skewsymmetric_q_kernel <__half>:\n");
    failures += run_zero_skewsymmetric_q_test<__half>("__half");

    printf("\nTesting cuda_copy_skewsymmetric_second_half_q_kernel <__half>:\n");
    failures += run_copy_second_half_q_test<__half>("__half");

    printf("\nTesting cuda_copy_skewsymmetric_first_half_q_kernel <__half>:\n");
    failures += run_copy_first_half_q_test<__half>("__half");

    printf("\nTesting cuda_get_skewsymmetric_second_half_q_kernel <__half>:\n");
    failures += run_get_second_half_q_test<__half>("__half");

    printf("\nTesting cuda_put_skewsymmetric_second_half_q_kernel <__half>:\n");
    failures += run_put_second_half_q_test<__half>("__half");
#endif /* WANT_HALF_PRECISION_REAL */

    printf("\n=== Summary: %d failure(s) ===\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
    fprintf(stderr, "Error: this test should only be called when the ELPA build is with NVIDIA GPU version enabled\n");
    abort();
}

#endif /* WITH_NVIDIA_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
