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

// Unit tests for kernels in
// src/invert_trm/GPU/CUDA/elpa_invert_trm_cuda.cu:
//
//   cuda_copy_a_tmat2     – copy a stripe of a_dev into tmat2_dev
//   cuda_copy_tmp2_tmat2  – copy nb columns of tmp2 into tmat2
//   cuda_copy_a_tmat1     – copy a block of a_dev into tmat1_dev and zero source
//   cuda_copy_tmp1_tmp2   – scatter packed lower-triangular tmp1 into tmp2
//   cuda_copy_a_tmp1      – gather lower-triangular block from a_dev into packed tmp1
//
// All tests use small integer values so equality comparisons are exact for
// both real and complex types in both double and float precision.
//
// Index derivations for each test are given in the per-test comments.

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

#include "../../../../src/invert_trm/GPU/CUDA/elpa_invert_trm_cuda.cu"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-64s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

// ---- type helpers ----

template <typename T> static T make_val(int n);
template <> double          make_val<double>(int n)          { return (double)n; }
template <> float           make_val<float>(int n)           { return (float)n; }
template <> cuDoubleComplex make_val<cuDoubleComplex>(int n)  { return make_cuDoubleComplex((double)n, (double)(n + 100)); }
template <> cuFloatComplex  make_val<cuFloatComplex>(int n)   { return make_cuFloatComplex((float)n, (float)(n + 100)); }

template <typename T>
static bool vals_eq(T a, T b)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

template <typename T>
static bool is_zero(T v)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        return v.x == 0 && v.y == 0;
    else
        return v == (T)0;
}

// ---- dispatchers: cuda_copy_a_tmat2 ----

static void call_copy_a_tmat2(double *a, double *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, cudaStream_t s)
{ cuda_copy_double_a_tmat2_FromC(a, tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

static void call_copy_a_tmat2(float *a, float *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, cudaStream_t s)
{ cuda_copy_float_a_tmat2_FromC(a, tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

static void call_copy_a_tmat2(cuDoubleComplex *a, cuDoubleComplex *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, cudaStream_t s)
{ cuda_copy_double_complex_a_tmat2_FromC((double _Complex *)a, (double _Complex *)tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

static void call_copy_a_tmat2(cuFloatComplex *a, cuFloatComplex *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, cudaStream_t s)
{ cuda_copy_float_complex_a_tmat2_FromC((float _Complex *)a, (float _Complex *)tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

// ---- dispatchers: cuda_copy_tmp2_tmat2 ----

static void call_copy_tmp2_tmat2(double *tmp2, double *tmat2,
    int *nblk, int *l_col1, int *nb, cudaStream_t s)
{ cuda_copy_double_tmp2_tmat2_FromC(tmp2, tmat2, nblk, l_col1, nb, s); }

static void call_copy_tmp2_tmat2(float *tmp2, float *tmat2,
    int *nblk, int *l_col1, int *nb, cudaStream_t s)
{ cuda_copy_float_tmp2_tmat2_FromC(tmp2, tmat2, nblk, l_col1, nb, s); }

static void call_copy_tmp2_tmat2(cuDoubleComplex *tmp2, cuDoubleComplex *tmat2,
    int *nblk, int *l_col1, int *nb, cudaStream_t s)
{ cuda_copy_double_complex_tmp2_tmat2_FromC((double _Complex *)tmp2, (double _Complex *)tmat2, nblk, l_col1, nb, s); }

static void call_copy_tmp2_tmat2(cuFloatComplex *tmp2, cuFloatComplex *tmat2,
    int *nblk, int *l_col1, int *nb, cudaStream_t s)
{ cuda_copy_float_complex_tmp2_tmat2_FromC((float _Complex *)tmp2, (float _Complex *)tmat2, nblk, l_col1, nb, s); }

// ---- dispatchers: cuda_copy_a_tmat1 ----

static void call_copy_a_tmat1(double *a, double *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, cudaStream_t s)
{ cuda_copy_double_a_tmat1_FromC(a, tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

static void call_copy_a_tmat1(float *a, float *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, cudaStream_t s)
{ cuda_copy_float_a_tmat1_FromC(a, tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

static void call_copy_a_tmat1(cuDoubleComplex *a, cuDoubleComplex *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, cudaStream_t s)
{ cuda_copy_double_complex_a_tmat1_FromC((double _Complex *)a, (double _Complex *)tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

static void call_copy_a_tmat1(cuFloatComplex *a, cuFloatComplex *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, cudaStream_t s)
{ cuda_copy_float_complex_a_tmat1_FromC((float _Complex *)a, (float _Complex *)tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

// ---- dispatchers: cuda_copy_tmp1_tmp2 ----

static void call_copy_tmp1_tmp2(double *tmp1, double *tmp2,
    int *nblk, int *nb, cudaStream_t s)
{ cuda_copy_double_tmp1_tmp2_FromC(tmp1, tmp2, nblk, nb, s); }

static void call_copy_tmp1_tmp2(float *tmp1, float *tmp2,
    int *nblk, int *nb, cudaStream_t s)
{ cuda_copy_float_tmp1_tmp2_FromC(tmp1, tmp2, nblk, nb, s); }

static void call_copy_tmp1_tmp2(cuDoubleComplex *tmp1, cuDoubleComplex *tmp2,
    int *nblk, int *nb, cudaStream_t s)
{ cuda_copy_double_complex_tmp1_tmp2_FromC((double _Complex *)tmp1, (double _Complex *)tmp2, nblk, nb, s); }

static void call_copy_tmp1_tmp2(cuFloatComplex *tmp1, cuFloatComplex *tmp2,
    int *nblk, int *nb, cudaStream_t s)
{ cuda_copy_float_complex_tmp1_tmp2_FromC((float _Complex *)tmp1, (float _Complex *)tmp2, nblk, nb, s); }

// ---- dispatchers: cuda_copy_a_tmp1 ----

static void call_copy_a_tmp1(double *a, double *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, cudaStream_t s)
{ cuda_copy_double_a_tmp1_FromC(a, tmp1, l_row1, l_col1, mR, nb, s); }

static void call_copy_a_tmp1(float *a, float *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, cudaStream_t s)
{ cuda_copy_float_a_tmp1_FromC(a, tmp1, l_row1, l_col1, mR, nb, s); }

static void call_copy_a_tmp1(cuDoubleComplex *a, cuDoubleComplex *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, cudaStream_t s)
{ cuda_copy_double_complex_a_tmp1_FromC((double _Complex *)a, (double _Complex *)tmp1, l_row1, l_col1, mR, nb, s); }

static void call_copy_a_tmp1(cuFloatComplex *a, cuFloatComplex *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, cudaStream_t s)
{ cuda_copy_float_complex_a_tmp1_FromC((float _Complex *)a, (float _Complex *)tmp1, l_row1, l_col1, mR, nb, s); }

// ============================================================
// Test: cuda_copy_a_tmat2
//
// nblk=2, matrixRows=4, l_cols=3, l_colx=2, l_row1=1, nb=2
// blocks = l_cols-l_colx+1 = 2, threads = nb = 2
//
// Kernel writes (0-based ni=threadIdx.x, ci=blockIdx.x):
//   tmat2[ni + (l_colx-1 + ci)*nblk] = a[(l_row1-1 + ni) + (l_colx-1 + ci)*matrixRows]
//   → tmat2[2]=a[4], tmat2[3]=a[5], tmat2[4]=a[8], tmat2[5]=a[9]
//
// a[k] = make_val(k+1), sentinel = make_val(-999)
// Expected: tmat2[2]=make_val(5), tmat2[3]=make_val(6),
//           tmat2[4]=make_val(9), tmat2[5]=make_val(10)
//           tmat2[0],tmat2[1] unchanged (sentinel)
// ============================================================
template <typename T>
static void run_copy_a_tmat2_test(const char *type_name)
{
    int nblk = 2, matrixRows = 4, l_cols = 3, l_colx = 2, l_row1 = 1, nb = 2;
    const int a_n     = matrixRows * l_cols;  // 12
    const int tmat2_n = nblk * l_cols;        // 6

    T h_a[12], h_tmat2[6];
    for (int k = 0; k < a_n; k++)     h_a[k]     = make_val<T>(k + 1);
    for (int k = 0; k < tmat2_n; k++) h_tmat2[k] = make_val<T>(-999);

    T *d_a, *d_tmat2;
    CUDA_CHECK(cudaMalloc(&d_a,     a_n     * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmat2, tmat2_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,     h_a,     a_n     * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmat2, h_tmat2, tmat2_n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_a_tmat2(d_a, d_tmat2, &nblk, &matrixRows, &l_cols, &l_colx, &l_row1, &nb,
                      (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[6];
    CUDA_CHECK(cudaMemcpy(h_res, d_tmat2, tmat2_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_tmat2));

    // Written positions and the a-index they come from
    const int tmat2_idx[] = {2, 3, 4, 5};
    const int a_idx[]     = {4, 5, 8, 9};

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 4; k++) {
        T exp = make_val<T>(a_idx[k] + 1);
        if (!vals_eq(h_res[tmat2_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_a_tmat2 <%s> tmat2[%d]", type_name, tmat2_idx[k]);
            REPORT(buf, false);
        }
    }
    // Verify sentinels at unwritten positions 0,1
    for (int k = 0; k < 2; k++) {
        if (!vals_eq(h_res[k], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_a_tmat2 <%s> tmat2[%d] sentinel clobbered", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf),
                 "copy_a_tmat2 <%s> 4 writes + 2 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: cuda_copy_tmp2_tmat2
//
// nblk=2, nb=2, l_col1=3
// blocks = nb = 2, threads = nb = 2
//
// Kernel writes (0-based ni=threadIdx.x, ci=blockIdx.x):
//   tmat2[ni + (l_col1-1 + ci)*nblk] = tmp2[ni + ci*nblk]
//   → tmat2[4]=tmp2[0], tmat2[5]=tmp2[1], tmat2[6]=tmp2[2], tmat2[7]=tmp2[3]
//
// tmp2[k] = make_val(k+1), tmat2 sentinel = make_val(-999)
// Expected: tmat2[4..7] = make_val(1..4); tmat2[0..3] unchanged
// ============================================================
template <typename T>
static void run_copy_tmp2_tmat2_test(const char *type_name)
{
    int nblk = 2, nb = 2, l_col1 = 3;
    const int tmp2_n  = nblk * nb;                   // 4
    const int tmat2_n = nblk * (l_col1 - 1 + nb);   // 8

    T h_tmp2[4], h_tmat2[8];
    for (int k = 0; k < tmp2_n; k++)  h_tmp2[k]  = make_val<T>(k + 1);
    for (int k = 0; k < tmat2_n; k++) h_tmat2[k] = make_val<T>(-999);

    T *d_tmp2, *d_tmat2;
    CUDA_CHECK(cudaMalloc(&d_tmp2,  tmp2_n  * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmat2, tmat2_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_tmp2,  h_tmp2,  tmp2_n  * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmat2, h_tmat2, tmat2_n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_tmp2_tmat2(d_tmp2, d_tmat2, &nblk, &l_col1, &nb, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[8];
    CUDA_CHECK(cudaMemcpy(h_res, d_tmat2, tmat2_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp2));
    CUDA_CHECK(cudaFree(d_tmat2));

    char buf[128];
    bool all_ok = true;
    // Written: tmat2[4..7] = tmp2[0..3] = make_val(1..4)
    for (int k = 0; k < 4; k++) {
        T exp = make_val<T>(k + 1);
        if (!vals_eq(h_res[4 + k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_tmp2_tmat2 <%s> tmat2[%d]", type_name, 4 + k);
            REPORT(buf, false);
        }
    }
    // Sentinels at tmat2[0..3]
    for (int k = 0; k < 4; k++) {
        if (!vals_eq(h_res[k], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_tmp2_tmat2 <%s> tmat2[%d] sentinel clobbered", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf),
                 "copy_tmp2_tmat2 <%s> 4 writes + 4 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: cuda_copy_a_tmat1
//
// l_rows=3, matrixRows=4, nb=2, l_row1=3, l_col1=2
// blocks = l_row1-1 = 2, threads = nb = 2
//
// Kernel writes (0-based ri=blockIdx.x, ni=threadIdx.x):
//   tmat1[ri + ni*l_rows] = a[ri + (l_col1-1 + ni)*matrixRows]
//   set_to_zero(a[ri + (l_col1-1 + ni)*matrixRows])
//   → tmat1[0]=a[4], tmat1[1]=a[5], tmat1[3]=a[8], tmat1[4]=a[9]
//     a[4]=a[5]=a[8]=a[9]=0
//
// a[k] = make_val(k+1), sentinel = make_val(-999)
// Expected tmat1: [0]=make_val(5), [1]=make_val(6), [3]=make_val(9), [4]=make_val(10)
//          tmat1[2],tmat1[5] unchanged (sentinel)
//          a[4]=a[5]=a[8]=a[9]=zero
// ============================================================
template <typename T>
static void run_copy_a_tmat1_test(const char *type_name)
{
    int l_rows = 3, matrixRows = 4, nb = 2, l_row1 = 3, l_col1 = 2;
    const int a_n     = matrixRows * (l_col1 - 1 + nb);  // 12
    const int tmat1_n = l_rows * nb;                      // 6

    T h_a[12], h_tmat1[6];
    for (int k = 0; k < a_n; k++)     h_a[k]     = make_val<T>(k + 1);
    for (int k = 0; k < tmat1_n; k++) h_tmat1[k] = make_val<T>(-999);

    T *d_a, *d_tmat1;
    CUDA_CHECK(cudaMalloc(&d_a,     a_n     * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmat1, tmat1_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,     h_a,     a_n     * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmat1, h_tmat1, tmat1_n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_a_tmat1(d_a, d_tmat1, &l_rows, &matrixRows, &nb, &l_row1, &l_col1,
                      (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res_tmat1[6], h_res_a[12];
    CUDA_CHECK(cudaMemcpy(h_res_tmat1, d_tmat1, tmat1_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_a,     d_a,     a_n     * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_tmat1));

    char buf[128];
    bool all_ok = true;

    // Written tmat1 positions and source a indices
    const int tmat1_idx[] = {0, 1, 3, 4};
    const int a_idx[]     = {4, 5, 8, 9};
    for (int k = 0; k < 4; k++) {
        T exp = make_val<T>(a_idx[k] + 1);
        if (!vals_eq(h_res_tmat1[tmat1_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_a_tmat1 <%s> tmat1[%d]", type_name, tmat1_idx[k]);
            REPORT(buf, false);
        }
    }
    // Sentinels at tmat1[2],tmat1[5]
    const int sent_idx[] = {2, 5};
    for (int k = 0; k < 2; k++) {
        if (!vals_eq(h_res_tmat1[sent_idx[k]], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_a_tmat1 <%s> tmat1[%d] sentinel clobbered", type_name, sent_idx[k]);
            REPORT(buf, false);
        }
    }
    // Zeroed positions in a
    for (int k = 0; k < 4; k++) {
        if (!is_zero(h_res_a[a_idx[k]])) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_a_tmat1 <%s> a[%d] not zeroed", type_name, a_idx[k]);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf),
                 "copy_a_tmat1 <%s> 4 writes + 2 sentinels + 4 zeros correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: cuda_copy_tmp1_tmp2
//
// nb=3, nblk=3
// blocks = nb = 3 (j_index), threads = nb = 3 (i_index)
// tmp1 holds nb*(nb+1)/2 = 6 packed lower-triangular elements.
//
// Kernel: if (j_index <= i_index):
//   tmp2[j-1 + (i-1)*nblk] = tmp1[i*(i-1)/2 + j - 1]
//
//   (i=1,j=1): tmp2[0] = tmp1[0]
//   (i=2,j=1): tmp2[3] = tmp1[1]   (i=2,j=2): tmp2[4] = tmp1[2]
//   (i=3,j=1): tmp2[6] = tmp1[3]   (i=3,j=2): tmp2[7] = tmp1[4]   (i=3,j=3): tmp2[8] = tmp1[5]
//
// tmp1[k] = make_val(k+1), sentinel = make_val(-999)
// Expected: tmp2[0]=make_val(1), tmp2[3]=make_val(2), tmp2[4]=make_val(3),
//           tmp2[6]=make_val(4), tmp2[7]=make_val(5), tmp2[8]=make_val(6)
//           tmp2[1],tmp2[2],tmp2[5] unchanged (j > i, not written)
// ============================================================
template <typename T>
static void run_copy_tmp1_tmp2_test(const char *type_name)
{
    int nb = 3, nblk = 3;
    const int tmp1_n = nb * (nb + 1) / 2;  // 6
    const int tmp2_n = nblk * nb;           // 9

    T h_tmp1[6], h_tmp2[9];
    for (int k = 0; k < tmp1_n; k++) h_tmp1[k] = make_val<T>(k + 1);
    for (int k = 0; k < tmp2_n; k++) h_tmp2[k] = make_val<T>(-999);

    T *d_tmp1, *d_tmp2;
    CUDA_CHECK(cudaMalloc(&d_tmp1, tmp1_n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmp2, tmp2_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_tmp1, h_tmp1, tmp1_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmp2, h_tmp2, tmp2_n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_tmp1_tmp2(d_tmp1, d_tmp2, &nblk, &nb, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[9];
    CUDA_CHECK(cudaMemcpy(h_res, d_tmp2, tmp2_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));

    // Written: tmp2[j-1 + (i-1)*nblk] = tmp1[i*(i-1)/2 + j-1]
    const int written_tmp2[] = {0, 3, 4, 6, 7, 8};
    const int tmp1_src[]     = {0, 1, 2, 3, 4, 5};
    const int sentinel_idx[] = {1, 2, 5};

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 6; k++) {
        T exp = make_val<T>(tmp1_src[k] + 1);
        if (!vals_eq(h_res[written_tmp2[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_tmp1_tmp2 <%s> tmp2[%d]", type_name, written_tmp2[k]);
            REPORT(buf, false);
        }
    }
    for (int k = 0; k < 3; k++) {
        if (!vals_eq(h_res[sentinel_idx[k]], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_tmp1_tmp2 <%s> tmp2[%d] sentinel clobbered", type_name, sentinel_idx[k]);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf),
                 "copy_tmp1_tmp2 <%s> 6 writes + 3 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: cuda_copy_a_tmp1
//
// nb=3, l_row1=1, l_col1=1, matrixRows=3
// blocks = nb = 3 (j_index), threads = nb = 3 (i_index)
//
// Kernel: if (j_index <= i_index):
//   tmp1[i*(i-1)/2 + j - 1] = a[l_row1-1 + j-1 + (l_col1-1 + i-1)*matrixRows]
//
//   (i=1,j=1): tmp1[0] = a[0]
//   (i=2,j=1): tmp1[1] = a[3]   (i=2,j=2): tmp1[2] = a[4]
//   (i=3,j=1): tmp1[3] = a[6]   (i=3,j=2): tmp1[4] = a[7]   (i=3,j=3): tmp1[5] = a[8]
//
// a[k] = make_val(k+1): a[0]=1, a[3]=4, a[4]=5, a[6]=7, a[7]=8, a[8]=9
// Expected: tmp1 = {make_val(1), make_val(4), make_val(5),
//                   make_val(7), make_val(8), make_val(9)}
// ============================================================
template <typename T>
static void run_copy_a_tmp1_test(const char *type_name)
{
    int nb = 3, l_row1 = 1, l_col1 = 1, matrixRows = 3;
    const int a_n    = matrixRows * nb;     // 9
    const int tmp1_n = nb * (nb + 1) / 2;  // 6

    T h_a[9], h_tmp1[6];
    for (int k = 0; k < a_n;    k++) h_a[k]    = make_val<T>(k + 1);
    for (int k = 0; k < tmp1_n; k++) h_tmp1[k] = make_val<T>(-999);

    T *d_a, *d_tmp1;
    CUDA_CHECK(cudaMalloc(&d_a,    a_n    * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmp1, tmp1_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,    h_a,    a_n    * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmp1, h_tmp1, tmp1_n * sizeof(T), cudaMemcpyHostToDevice));

    call_copy_a_tmp1(d_a, d_tmp1, &l_row1, &l_col1, &matrixRows, &nb, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[6];
    CUDA_CHECK(cudaMemcpy(h_res, d_tmp1, tmp1_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_tmp1));

    // a indices that map to tmp1[0..5]
    const int a_idx[] = {0, 3, 4, 6, 7, 8};

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 6; k++) {
        T exp = make_val<T>(a_idx[k] + 1);
        if (!vals_eq(h_res[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf),
                     "copy_a_tmp1 <%s> tmp1[%d]", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf),
                 "copy_a_tmp1 <%s> all 6 elements correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for elpa_invert_trm_cuda.cu kernels ===\n\n");

    printf("cuda_copy_a_tmat2:\n");
    run_copy_a_tmat2_test<double>          ("double");
    run_copy_a_tmat2_test<float>           ("float");
    run_copy_a_tmat2_test<cuDoubleComplex> ("cuDoubleComplex");
    run_copy_a_tmat2_test<cuFloatComplex>  ("cuFloatComplex");

    printf("\ncuda_copy_tmp2_tmat2:\n");
    run_copy_tmp2_tmat2_test<double>          ("double");
    run_copy_tmp2_tmat2_test<float>           ("float");
    run_copy_tmp2_tmat2_test<cuDoubleComplex> ("cuDoubleComplex");
    run_copy_tmp2_tmat2_test<cuFloatComplex>  ("cuFloatComplex");

    printf("\ncuda_copy_a_tmat1:\n");
    run_copy_a_tmat1_test<double>          ("double");
    run_copy_a_tmat1_test<float>           ("float");
    run_copy_a_tmat1_test<cuDoubleComplex> ("cuDoubleComplex");
    run_copy_a_tmat1_test<cuFloatComplex>  ("cuFloatComplex");

    printf("\ncuda_copy_tmp1_tmp2:\n");
    run_copy_tmp1_tmp2_test<double>          ("double");
    run_copy_tmp1_tmp2_test<float>           ("float");
    run_copy_tmp1_tmp2_test<cuDoubleComplex> ("cuDoubleComplex");
    run_copy_tmp1_tmp2_test<cuFloatComplex>  ("cuFloatComplex");

    printf("\ncuda_copy_a_tmp1:\n");
    run_copy_a_tmp1_test<double>          ("double");
    run_copy_a_tmp1_test<float>           ("float");
    run_copy_a_tmp1_test<cuDoubleComplex> ("cuDoubleComplex");
    run_copy_a_tmp1_test<cuFloatComplex>  ("cuFloatComplex");

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
    fprintf(stderr, "Error: this test requires WITH_NVIDIA_GPU_VERSION\n");
    abort();
}

#endif /* WITH_NVIDIA_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
