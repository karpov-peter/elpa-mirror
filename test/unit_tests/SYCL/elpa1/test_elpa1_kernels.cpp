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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <type_traits>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_SYCL_GPU_VERSION

#include <sycl/sycl.hpp>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-72s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

// ============================================================
// Kernel bodies (self-contained, no syclCommon.hpp dependency)
// ============================================================

// Kernel 1: copy real part to complex q
template <typename T>
static void sycl_copy_real_part_to_q_complex_kernel(
    std::complex<T> *q_dev, T *q_real_dev,
    int matrixRows, int l_rows, int l_cols_nev,
    const sycl::nd_item<2> &it)
{
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (row < l_rows && col < l_cols_nev) {
        int index = row + matrixRows * col;
        q_dev[index] = std::complex<T>(q_real_dev[index], 0.0);
    }
}

// Kernel 2: zero skewsymmetric q upper half
template <typename T>
static void sycl_zero_skewsymmetric_q_kernel(
    T *q_dev, int matrixRows, int matrixCols,
    const sycl::nd_item<2> &it)
{
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (row < matrixRows && col < matrixCols) {
        int index = row + matrixRows * (col + matrixCols);
        q_dev[index] = 0.0;
    }
}

// Kernel 3: copy skewsymmetric second half (template bool isPlus)
template <typename T, bool isPlus>
static void sycl_copy_skewsymmetric_second_half_q_kernel(
    T *q_dev, const int i, const int matrixRows, const int matrixCols,
    const sycl::nd_item<1> &it)
{
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (col >= matrixCols && col < 2 * matrixCols) {
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        if constexpr (isPlus) {
            q_dev[index] = q_dev[indexLow];
        } else {
            q_dev[index] = -q_dev[indexLow];
        }
        q_dev[indexLow] = 0.0;
    }
}

// Kernel 4: copy skewsymmetric first half (negate row i-1)
template <typename T>
static void sycl_copy_skewsymmetric_first_half_q_kernel(
    T *q_dev, const int i, const int matrixRows, const int matrixCols,
    const sycl::nd_item<1> &it)
{
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (col < matrixCols) {
        int index = (i-1) + matrixRows * col;
        q_dev[index] = -q_dev[index];
    }
}

// Kernel 5: get skewsymmetric second half (q_2 <- upper half of q)
template <typename T>
static void sycl_get_skewsymmetric_second_half_q_kernel(
    T *q_dev, T *q_2_dev, int matrixRows, int matrixCols,
    const sycl::nd_item<2> &it)
{
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (row < matrixRows && col < matrixCols) {
        int index  = row + matrixRows * col;
        int index2 = row + matrixRows * (col + matrixCols);
        q_2_dev[index] = q_dev[index2];
    }
}

// Kernel 6: put skewsymmetric second half (upper half of q <- q2)
template <typename T>
static void sycl_put_skewsymmetric_second_half_q_kernel(
    T *q_dev, T *q2_dev, int matrixRows, int matrixCols,
    const sycl::nd_item<2> &it)
{
    int row = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
    int col = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    if (row < matrixRows && col < matrixCols) {
        int index  = row + matrixRows * col;
        int index2 = row + matrixRows * (col + matrixCols);
        q_dev[index2] = q2_dev[index];
    }
}

// ============================================================
// Test 1: sycl_copy_real_part_to_q_complex
//
// matrixRows=4, l_rows=2, l_cols_nev=3 (12 elements)
// q_real_dev[i] = i+1
// Expected: q_dev[row + matrixRows*col] = {q_real_dev[row + matrixRows*col], 0}
//           for row<2, col<3; positions with row>=2 untouched (sentinel {99,99}).
// ============================================================
template <typename T>
static void test_copy_real_part_to_q(sycl::queue &q, const char *tname)
{
    printf("\nsycl_copy_real_part_to_q_complex<%s>:\n", tname);

    const int matrixRows = 4, l_rows = 2, l_cols_nev = 3;
    const int n = matrixRows * l_cols_nev;  // 12

    T h_qreal[12];
    std::complex<T> h_q[12];
    for (int k = 0; k < n; k++) {
        h_qreal[k] = (T)(k + 1);
        h_q[k] = std::complex<T>((T)99, (T)99);
    }

    T              *d_qreal = sycl::malloc_device<T>(n, q);
    std::complex<T> *d_q   = sycl::malloc_device<std::complex<T>>(n, q);
    q.memcpy(d_qreal, h_qreal, n * sizeof(T)).wait();
    q.memcpy(d_q, h_q, n * sizeof(std::complex<T>)).wait();

    sycl::range<2> threadsPerBlock = {4, 4};
    int blk_col = (l_cols_nev + (int)threadsPerBlock[0] - 1) / (int)threadsPerBlock[0];
    int blk_row = (l_rows     + (int)threadsPerBlock[1] - 1) / (int)threadsPerBlock[1];
    sycl::range<2> blocks = {(size_t)blk_col, (size_t)blk_row};

    q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock),
                   [=](sycl::nd_item<2> it) {
        sycl_copy_real_part_to_q_complex_kernel(d_q, d_qreal, matrixRows, l_rows, l_cols_nev, it);
    });
    q.wait_and_throw();

    q.memcpy(h_q, d_q, n * sizeof(std::complex<T>)).wait();
    sycl::free(d_qreal, q);
    sycl::free(d_q, q);

    bool ok_valid   = true;
    bool ok_sentinel = true;
    for (int col = 0; col < l_cols_nev; col++) {
        for (int row = 0; row < matrixRows; row++) {
            int idx = row + matrixRows * col;
            if (row < l_rows) {
                if (h_q[idx].real() != h_qreal[idx] || h_q[idx].imag() != (T)0)
                    ok_valid = false;
            } else {
                if (h_q[idx].real() != (T)99 || h_q[idx].imag() != (T)99)
                    ok_sentinel = false;
            }
        }
    }

    char lbl1[128], lbl2[128];
    snprintf(lbl1, sizeof(lbl1), "%s copy_real_to_q: valid cells correct (real=qreal, imag=0)", tname);
    snprintf(lbl2, sizeof(lbl2), "%s copy_real_to_q: out-of-bound rows untouched (sentinel)", tname);
    REPORT(lbl1, ok_valid);
    REPORT(lbl2, ok_sentinel);
}

// ============================================================
// Test 2: sycl_zero_skewsymmetric_q
//
// matrixRows=2, matrixCols=2
// Allocate q_dev[8], fill all with 1.0.
// After kernel: indices [4..7] (upper half) = 0, indices [0..3] unchanged.
// ============================================================
template <typename T>
static void test_zero_skewsymmetric_q(sycl::queue &q, const char *tname)
{
    printf("\nsycl_zero_skewsymmetric_q<%s>:\n", tname);

    const int matrixRows = 2, matrixCols = 2;
    const int n = matrixRows * 2 * matrixCols;  // 8

    T h_q[8];
    for (int k = 0; k < n; k++) h_q[k] = (T)1;

    T *d_q = sycl::malloc_device<T>(n, q);
    q.memcpy(d_q, h_q, n * sizeof(T)).wait();

    sycl::range<2> threadsPerBlock = {4, 4};
    int blk_col = (matrixCols + (int)threadsPerBlock[0] - 1) / (int)threadsPerBlock[0];
    int blk_row = (matrixRows + (int)threadsPerBlock[1] - 1) / (int)threadsPerBlock[1];
    sycl::range<2> blocks = {(size_t)blk_col, (size_t)blk_row};

    q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock),
                   [=](sycl::nd_item<2> it) {
        sycl_zero_skewsymmetric_q_kernel(d_q, matrixRows, matrixCols, it);
    });
    q.wait_and_throw();

    T h_result[8];
    q.memcpy(h_result, d_q, n * sizeof(T)).wait();
    sycl::free(d_q, q);

    // Build expected: lower half [0..3] unchanged (1.0), upper half [4..7] = 0.0
    T expected[8];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * (col + matrixCols)] = (T)0;

    bool ok_lower = true, ok_upper = true;
    for (int k = 0; k < n / 2; k++)
        if (h_result[k] != expected[k]) ok_lower = false;
    for (int k = n / 2; k < n; k++)
        if (h_result[k] != expected[k]) ok_upper = false;

    char lbl1[128], lbl2[128];
    snprintf(lbl1, sizeof(lbl1), "%s zero_skewsymmetric_q: lower half unchanged", tname);
    snprintf(lbl2, sizeof(lbl2), "%s zero_skewsymmetric_q: upper half zeroed", tname);
    REPORT(lbl1, ok_lower);
    REPORT(lbl2, ok_upper);
}

// ============================================================
// Test 3: sycl_copy_skewsymmetric_second_half_q_kernel
//
// matrixRows=2, matrixCols=2, i=1
// q_dev[8]: set q_dev[0]=5.0, q_dev[2]=7.0 (positions (i-1)+matrixRows*col for col=0,1)
// isPlus=true:  q_dev[4]=5, q_dev[6]=7; q_dev[0]=0, q_dev[2]=0
// isPlus=false: q_dev[4]=-5, q_dev[6]=-7; q_dev[0]=0, q_dev[2]=0
// Launch 1D: threadsPerBlock=32, blocks=ceil(2*matrixCols/32)=1
// ============================================================
template <typename T>
static void test_copy_second_half_q_one(sycl::queue &q, const char *tname, bool isPlus)
{
    const int matrixRows = 2, matrixCols = 2, i_val = 1;
    const int n = 4 * matrixRows;  // 8

    T h_q[8] = {0};
    h_q[0] = (T)5;  // (i-1) + matrixRows*0
    h_q[2] = (T)7;  // (i-1) + matrixRows*1

    T *d_q = sycl::malloc_device<T>(n, q);
    q.memcpy(d_q, h_q, n * sizeof(T)).wait();

    const int threadsPerBlock = 32;
    int blocks = (2 * matrixCols + threadsPerBlock - 1) / threadsPerBlock;

    if (isPlus) {
        q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_skewsymmetric_second_half_q_kernel<T, true>(d_q, i_val, matrixRows, matrixCols, it);
        });
    } else {
        q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_skewsymmetric_second_half_q_kernel<T, false>(d_q, i_val, matrixRows, matrixCols, it);
        });
    }
    q.wait_and_throw();

    T h_result[8] = {0};
    q.memcpy(h_result, d_q, n * sizeof(T)).wait();
    sycl::free(d_q, q);

    // Check: source positions zeroed
    bool ok_src_zero = (h_result[0] == (T)0 && h_result[2] == (T)0);
    // Check: destination positions correctly set
    T exp4 = isPlus ? (T)5  : (T)-5;
    T exp6 = isPlus ? (T)7  : (T)-7;
    bool ok_dst = (h_result[4] == exp4 && h_result[6] == exp6);

    char lbl1[128], lbl2[128];
    snprintf(lbl1, sizeof(lbl1), "%s copy_second_half_q isPlus=%s: source zeroed",
             tname, isPlus ? "true" : "false");
    snprintf(lbl2, sizeof(lbl2), "%s copy_second_half_q isPlus=%s: destination correct",
             tname, isPlus ? "true" : "false");
    REPORT(lbl1, ok_src_zero);
    REPORT(lbl2, ok_dst);
}

template <typename T>
static void test_copy_second_half_q(sycl::queue &q, const char *tname)
{
    printf("\nsycl_copy_skewsymmetric_second_half_q_kernel<%s>:\n", tname);
    test_copy_second_half_q_one<T>(q, tname, true);   // isPlus=true
    test_copy_second_half_q_one<T>(q, tname, false);  // isPlus=false
}

// ============================================================
// Test 4: sycl_copy_skewsymmetric_first_half_q
//
// matrixRows=2, matrixCols=3, i=1
// h_q = {1,2,3,4,5,6}: layout q[row + matrixRows*col]
// After kernel: q_dev[(i-1)+matrixRows*col] = -q_dev[(i-1)+matrixRows*col] for col<3
// That is: q[0]=-1, q[2]=-3, q[4]=-5; q[1],q[3],q[5] unchanged.
// ============================================================
template <typename T>
static void test_copy_first_half_q(sycl::queue &q, const char *tname)
{
    printf("\nsycl_copy_skewsymmetric_first_half_q<%s>:\n", tname);

    const int matrixRows = 2, matrixCols = 3, i_val = 1;
    const int n = matrixRows * matrixCols;  // 6

    T h_q[6];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q = sycl::malloc_device<T>(n, q);
    q.memcpy(d_q, h_q, n * sizeof(T)).wait();

    const int threadsPerBlock = 32;
    int blocks = (matrixCols + threadsPerBlock - 1) / threadsPerBlock;

    q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock),
                   [=](sycl::nd_item<1> it) {
        sycl_copy_skewsymmetric_first_half_q_kernel(d_q, i_val, matrixRows, matrixCols, it);
    });
    q.wait_and_throw();

    T h_result[6];
    q.memcpy(h_result, d_q, n * sizeof(T)).wait();
    sycl::free(d_q, q);

    // Build expected: negate row (i-1)=0 for each col
    T expected[6];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        expected[(i_val-1) + matrixRows * col] = -h_q[(i_val-1) + matrixRows * col];

    bool ok = true;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) ok = false;

    char lbl[128];
    snprintf(lbl, sizeof(lbl), "%s copy_first_half_q: row 0 negated, others unchanged", tname);
    REPORT(lbl, ok);
}

// ============================================================
// Test 5: sycl_get_skewsymmetric_second_half_q
//
// matrixRows=2, matrixCols=2
// q_dev[8]: upper half (indices [4..7]) = {10,20,30,40}
// After kernel: q_2_dev[0..3] = {10,20,30,40}
// ============================================================
template <typename T>
static void test_get_second_half_q(sycl::queue &q, const char *tname)
{
    printf("\nsycl_get_skewsymmetric_second_half_q<%s>:\n", tname);

    const int matrixRows = 2, matrixCols = 2;
    const int n_q  = matrixRows * 2 * matrixCols;  // 8
    const int n_q2 = matrixRows * matrixCols;       // 4

    T h_q[8] = {0};
    // Set upper half: q[row + matrixRows*(col+matrixCols)] for row<2, col<2
    h_q[4] = (T)10; h_q[5] = (T)20; h_q[6] = (T)30; h_q[7] = (T)40;

    T *d_q  = sycl::malloc_device<T>(n_q,  q);
    T *d_q2 = sycl::malloc_device<T>(n_q2, q);
    q.memcpy(d_q, h_q, n_q * sizeof(T)).wait();
    q.memset(d_q2, 0, n_q2 * sizeof(T)).wait();

    sycl::range<2> threadsPerBlock = {4, 4};
    int blk_col = (matrixCols + (int)threadsPerBlock[0] - 1) / (int)threadsPerBlock[0];
    int blk_row = (matrixRows + (int)threadsPerBlock[1] - 1) / (int)threadsPerBlock[1];
    sycl::range<2> blocks = {(size_t)blk_col, (size_t)blk_row};

    q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock),
                   [=](sycl::nd_item<2> it) {
        sycl_get_skewsymmetric_second_half_q_kernel(d_q, d_q2, matrixRows, matrixCols, it);
    });
    q.wait_and_throw();

    T h_q2[4];
    q.memcpy(h_q2, d_q2, n_q2 * sizeof(T)).wait();
    sycl::free(d_q, q);
    sycl::free(d_q2, q);

    // Expected: q_2[row + matrixRows*col] = q[row + matrixRows*(col+matrixCols)]
    T expected[4];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * col] = h_q[row + matrixRows * (col + matrixCols)];

    bool ok = true;
    for (int k = 0; k < n_q2; k++)
        if (h_q2[k] != expected[k]) ok = false;

    char lbl[128];
    snprintf(lbl, sizeof(lbl), "%s get_second_half_q: q_2 = upper half of q", tname);
    REPORT(lbl, ok);
}

// ============================================================
// Test 6: sycl_put_skewsymmetric_second_half_q
//
// matrixRows=2, matrixCols=2
// q2_dev[0..3] = {10,20,30,40}
// After kernel: q_dev[4..7] = {10,20,30,40}, q_dev[0..3] unchanged (0.0).
// ============================================================
template <typename T>
static void test_put_second_half_q(sycl::queue &q, const char *tname)
{
    printf("\nsycl_put_skewsymmetric_second_half_q<%s>:\n", tname);

    const int matrixRows = 2, matrixCols = 2;
    const int n_q  = matrixRows * 2 * matrixCols;  // 8
    const int n_q2 = matrixRows * matrixCols;       // 4

    T h_q2[4]  = {(T)10, (T)20, (T)30, (T)40};

    T *d_q  = sycl::malloc_device<T>(n_q,  q);
    T *d_q2 = sycl::malloc_device<T>(n_q2, q);
    q.memset(d_q, 0, n_q * sizeof(T)).wait();
    q.memcpy(d_q2, h_q2, n_q2 * sizeof(T)).wait();

    sycl::range<2> threadsPerBlock = {4, 4};
    int blk_col = (matrixCols + (int)threadsPerBlock[0] - 1) / (int)threadsPerBlock[0];
    int blk_row = (matrixRows + (int)threadsPerBlock[1] - 1) / (int)threadsPerBlock[1];
    sycl::range<2> blocks = {(size_t)blk_col, (size_t)blk_row};

    q.parallel_for(sycl::nd_range<2>(blocks * threadsPerBlock, threadsPerBlock),
                   [=](sycl::nd_item<2> it) {
        sycl_put_skewsymmetric_second_half_q_kernel(d_q, d_q2, matrixRows, matrixCols, it);
    });
    q.wait_and_throw();

    T h_result[8];
    q.memcpy(h_result, d_q, n_q * sizeof(T)).wait();
    sycl::free(d_q, q);
    sycl::free(d_q2, q);

    // Expected: lower half unchanged (0.0); upper half = h_q2
    bool ok_lower = true, ok_upper = true;
    for (int k = 0; k < n_q2; k++)
        if (h_result[k] != (T)0) ok_lower = false;
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            if (h_result[row + matrixRows * (col + matrixCols)] != h_q2[row + matrixRows * col])
                ok_upper = false;

    char lbl1[128], lbl2[128];
    snprintf(lbl1, sizeof(lbl1), "%s put_second_half_q: lower half unchanged (zeros)", tname);
    snprintf(lbl2, sizeof(lbl2), "%s put_second_half_q: upper half = q2", tname);
    REPORT(lbl1, ok_lower);
    REPORT(lbl2, ok_upper);
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for elpa1 kernels (SYCL) ===\n");

    sycl::queue q(sycl::default_selector_v);

    test_copy_real_part_to_q<double>(q, "double");
    test_copy_real_part_to_q<float> (q, "float");

    test_zero_skewsymmetric_q<double>(q, "double");
    test_zero_skewsymmetric_q<float> (q, "float");

    test_copy_second_half_q<double>(q, "double");
    test_copy_second_half_q<float> (q, "float");

    test_copy_first_half_q<double>(q, "double");
    test_copy_first_half_q<float> (q, "float");

    test_get_second_half_q<double>(q, "double");
    test_get_second_half_q<float> (q, "float");

    test_put_second_half_q<double>(q, "double");
    test_put_second_half_q<float> (q, "float");

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
    fprintf(stderr, "Error: this test requires WITH_SYCL_GPU_VERSION\n");
    abort();
}

#endif /* WITH_SYCL_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
