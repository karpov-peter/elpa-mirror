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

// Unit tests for kernels in src/cholesky/GPU/CUDA/elpa_cholesky_cuda.cu:
//   cuda_check_device_info_kernel
//   cuda_accumulate_device_info_kernel
//   cuda_copy_a_tmatc_kernel<T>       (all four data types)
//   cuda_set_a_lower_to_zero_kernel<T> (all four data types)

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

#include "../../../../src/cholesky/GPU/CUDA/elpa_cholesky_cuda.cu"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---- host-side type helpers ----

template <typename T>
static T make_val(int i)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>)
        return make_cuDoubleComplex((double)i, (double)(i * 2));
    else if constexpr (std::is_same_v<T, cuFloatComplex>)
        return make_cuFloatComplex((float)i, (float)(i * 2));
    else
        return (T)i;
}

template <typename T>
static T zero_val()
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>) return make_cuDoubleComplex(0.0, 0.0);
    else if constexpr (std::is_same_v<T, cuFloatComplex>) return make_cuFloatComplex(0.0f, 0.0f);
    else return (T)0;
}

template <typename T>
static T conj_val(T v)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>)
        return make_cuDoubleComplex(v.x, -v.y);
    else if constexpr (std::is_same_v<T, cuFloatComplex>)
        return make_cuFloatComplex(v.x, -v.y);
    else
        return v;
}

template <typename T>
static bool vals_equal(T a, T b)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

// ---- per-type char tag for cuda_set_a_lower_to_zero_FromC ----

template <typename T> static char type_char();
template <> char type_char<double>()          { return 'D'; }
template <> char type_char<float>()           { return 'S'; }
template <> char type_char<cuDoubleComplex>() { return 'Z'; }
template <> char type_char<cuFloatComplex>()  { return 'C'; }

// ---- dispatcher overloads for cuda_copy_a_tmatc ----

static void call_copy_a_tmatc(double *a, double *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               cudaStream_t s)
{ cuda_copy_double_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

static void call_copy_a_tmatc(float *a, float *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               cudaStream_t s)
{ cuda_copy_float_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

static void call_copy_a_tmatc(cuDoubleComplex *a, cuDoubleComplex *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               cudaStream_t s)
{ cuda_copy_double_complex_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

static void call_copy_a_tmatc(cuFloatComplex *a, cuFloatComplex *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               cudaStream_t s)
{ cuda_copy_float_complex_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

// ============================================================
// Test: cuda_check_device_info_kernel
//
// info_dev = 0  →  assertion passes, kernel completes normally.
// ============================================================
static int run_check_device_info_test()
{
    int h_info = 0;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_info, &h_info, sizeof(int), cudaMemcpyHostToDevice));

    cuda_check_device_info_FromC(d_info, nullptr);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_info));
    printf("  cuda_check_device_info_kernel   (info=0): PASS\n");
    return 0;
}

// ============================================================
// Test: cuda_accumulate_device_info_kernel
//
// Start with info_abs=5, then add abs(-3)=3, abs(0)=0, abs(7)=7.
// Expected final value: 5 + 3 + 0 + 7 = 15.
// ============================================================
static int run_accumulate_device_info_test()
{
    int h_abs = 5;
    int h_new;
    int *d_abs, *d_new;
    CUDA_CHECK(cudaMalloc(&d_abs, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_abs, &h_abs, sizeof(int), cudaMemcpyHostToDevice));

    int deltas[] = { -3, 0, 7 };
    for (int i = 0; i < 3; i++) {
        h_new = deltas[i];
        CUDA_CHECK(cudaMemcpy(d_new, &h_new, sizeof(int), cudaMemcpyHostToDevice));
        cuda_accumulate_device_info_FromC(d_abs, d_new, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(&h_abs, d_abs, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_abs));
    CUDA_CHECK(cudaFree(d_new));

    int failures = (h_abs != 15) ? 1 : 0;
    printf("  cuda_accumulate_device_info_kernel (expected 15, got %d): %s\n",
           h_abs, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: cuda_copy_a_tmatc_kernel<T>
//
// Parameters: nblk=2, matrixRows=3, l_cols=3, l_colx=1, l_row1=1
//
// a_dev is col-major with leading dimension matrixRows=3.
// The kernel reads rows [l_row1, l_row1+nblk-1] = [1,2] of all
// l_cols=3 columns and writes them into tmatc with conjugate
// transpose:
//   tmatc[(jj-1) + (ii-1)*l_cols] = conj(a[(ii-1) + (jj-1)*matrixRows])
//
// a_dev  size: matrixRows * l_cols = 9   (filled with make_val(1..9))
// tmatc  size: l_cols * nblk        = 6
// ============================================================
template <typename T>
static int run_copy_a_tmatc_test(const char *type_name)
{
    int nblk = 2, matrixRows = 3, l_cols = 3, l_colx = 1, l_row1 = 1;
    const int a_n     = matrixRows * l_cols;  // 9
    const int tmatc_n = l_cols * nblk;        // 6

    T h_a[9];
    for (int i = 0; i < a_n; i++) h_a[i] = make_val<T>(i + 1);

    T *d_a, *d_tmatc;
    CUDA_CHECK(cudaMalloc(&d_a,     a_n     * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmatc, tmatc_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tmatc, 0, tmatc_n * sizeof(T)));

    call_copy_a_tmatc(d_a, d_tmatc, &nblk, &matrixRows, &l_cols, &l_colx, &l_row1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_tmatc[6];
    CUDA_CHECK(cudaMemcpy(h_tmatc, d_tmatc, tmatc_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_tmatc));

    // Build expected: for ii∈[1,nblk], jj∈[1,l_cols]:
    //   tmatc[(jj-1) + (ii-1)*l_cols] = conj(a[(ii-1) + (jj-1)*matrixRows])
    T expected[6];
    for (int ii = 1; ii <= nblk; ii++)
        for (int jj = 1; jj <= l_cols; jj++)
            expected[(jj-1) + (ii-1)*l_cols] = conj_val(h_a[(ii-1) + (jj-1)*matrixRows]);

    int failures = 0;
    for (int i = 0; i < tmatc_n; i++)
        if (!vals_equal(h_tmatc[i], expected[i])) failures++;

    printf("  cuda_copy_a_tmatc_kernel        <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: cuda_set_a_lower_to_zero_kernel<T>
//
// Single-process layout: np_cols=1, np_rows=1, my_pcol=0, my_prow=0
// na=3, matrixRows=3, nblk=1
//
// Every global column J_gl belongs to the local process.
// l_col1 = J_gl, l_row1 = J_gl+1.
//
// Sub-diagonal elements zeroed in a 3×3 col-major matrix:
//   J_gl=1 → col 0, rows 1..2 → a[1], a[2]
//   J_gl=2 → col 1, row  2    → a[5]
//   J_gl=3 → col 2, l_row1=4 > matrixRows=3 → nothing
// ============================================================
template <typename T>
static int run_set_a_lower_to_zero_test(const char *type_name)
{
    int na = 3, matrixRows = 3, my_pcol = 0, np_cols = 1;
    int my_prow = 0, np_rows = 1, nblk = 1, wantDebug = 1;
    const int a_n = matrixRows * na;  // 9

    T h_a[9];
    for (int i = 0; i < a_n; i++) h_a[i] = make_val<T>(i + 1);

    T *d_a;
    CUDA_CHECK(cudaMalloc(&d_a, a_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_n * sizeof(T), cudaMemcpyHostToDevice));

    cuda_set_a_lower_to_zero_FromC(type_char<T>(), (intptr_t)d_a,
                                   &na, &matrixRows, &my_pcol, &np_cols,
                                   &my_prow, &np_rows, &nblk, &wantDebug, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[9];
    CUDA_CHECK(cudaMemcpy(h_result, d_a, a_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));

    // Build expected: start from original, zero the sub-diagonal entries
    T expected[9];
    for (int i = 0; i < a_n; i++) expected[i] = h_a[i];
    expected[1] = zero_val<T>();  // col 0, row 1
    expected[2] = zero_val<T>();  // col 0, row 2
    expected[5] = zero_val<T>();  // col 1, row 2

    int failures = 0;
    for (int i = 0; i < a_n; i++)
        if (!vals_equal(h_result[i], expected[i])) failures++;

    printf("  cuda_set_a_lower_to_zero_kernel <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
int main(void)
{
    int failures = 0;

    printf("Testing cuda_check_device_info_kernel:\n");
    failures += run_check_device_info_test();

    printf("\nTesting cuda_accumulate_device_info_kernel:\n");
    failures += run_accumulate_device_info_test();

    printf("\nTesting cuda_copy_a_tmatc_kernel:\n");
    failures += run_copy_a_tmatc_test<double>         ("double");
    failures += run_copy_a_tmatc_test<float>          ("float");
    failures += run_copy_a_tmatc_test<cuDoubleComplex>("cuDoubleComplex");
    failures += run_copy_a_tmatc_test<cuFloatComplex> ("cuFloatComplex");

    printf("\nTesting cuda_set_a_lower_to_zero_kernel:\n");
    failures += run_set_a_lower_to_zero_test<double>         ("double");
    failures += run_set_a_lower_to_zero_test<float>          ("float");
    failures += run_set_a_lower_to_zero_test<cuDoubleComplex>("cuDoubleComplex");
    failures += run_set_a_lower_to_zero_test<cuFloatComplex> ("cuFloatComplex");

    if (failures == 0)
        printf("\nAll tests passed.\n");
    else
        printf("\n%d test(s) FAILED.\n", failures);

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
