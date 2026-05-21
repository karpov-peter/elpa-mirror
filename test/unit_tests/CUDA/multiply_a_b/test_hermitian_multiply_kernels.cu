//    Copyright 2026, A. Marek, MPCDF
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

// Unit tests for the three kernels in elpa_hermitian_multiply_gpu.h (CUDA).
//
// Kernel 1 — gpu_copy_tmp2_c:
//   Copies a dense nstor×(lce-lcs+1) matrix from tmp2_dev (col-major, stride nstor)
//   into a subblock of c_dev (col-major, stride ldc) starting at row nr_done,
//   column lcs-1 (0-indexed).
//   Formula: c[nr_done+i + ldc*(lcs-1+j)] = tmp2[i + nstor*j]
//            for i in [0,nstor), j in [0,lce-lcs].
//
// Kernel 2 — gpu_copy_a_aux_bc_loop:
//   For each block n in [0,n_size):
//     aux_bc[n_aux_bc_save[n]+i] = a[(lrs_save[n]-1)+i + lda*(noff*nblk+n)]
//     for i in [0, lre_save[n]-lrs_save[n]].
//   Gathers contiguous row ranges from columns of a into the packed aux_bc array.
//
// Kernel 3 — gpu_copy_aux_bc_aux_mat_loop:
//   For each block n in [0,n_size):
//     aux_mat[(lrs_save[n]-1)+i + l_rows*(nstor0+n-1)] = aux_bc[n_aux_bc_save[n]+i]
//     for i in [0, lre_save[n]-lrs_save[n]].
//   Scatters packed aux_bc data into column nstor0+n-1 of aux_mat at specific rows.
//   Guard: if n_size <= 0, returns immediately without any writes.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cuComplex.h>
#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"
#include "../../../../src/multiply_a_b/GPU/elpa_hermitian_multiply_gpu.h"

// ============================================================
// Infrastructure
// ============================================================

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
              __FILE__, __LINE__, cudaGetErrorString(_err));                  \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-72s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool deq(double a, double b, double tol = 1e-10) { return fabs(a - b) < tol; }
static bool feq(float  a, float  b, float  tol = 1e-5f) { return fabsf(a - b) < tol; }

template <typename T>
static T *dev_alloc_upload(const T *h, int n) {
  T *d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
static T *dev_alloc_zero(int n) {
  T *d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T)));
  return d;
}

template <typename T>
static void dev_read_n(T *h, T *d, int n) {
  CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// ============================================================
// CPU references — exact C translations of each kernel's loop body.
// ============================================================

// gpu_copy_tmp2_c reference:
//   c[nr_done + i + ldc*(lcs-1+j)] = tmp2[i + nstor*j]
//   i in [0,nstor), j in [0,lce-lcs].
template <typename T>
static void cpu_copy_tmp2_c_ref(const T *tmp2, T *c,
                                 int nr_done, int nstor,
                                 int lcs, int lce, int ldc, int /*ldcCols*/)
{
  for (int j = 0; j < lce - lcs + 1; j++)
    for (int i = 0; i < nstor; i++)
      c[nr_done + i + ldc * (lcs - 1 + j)] = tmp2[i + nstor * j];
}

// gpu_copy_a_aux_bc_loop reference:
//   aux_bc[n_aux_bc_save[n]+i] = a[(lrs_save[n]-1)+i + lda*(noff*nblk+n)]
//   for each block n in [0,n_size), i in [0, lre_save[n]-lrs_save[n]].
template <typename T>
static void cpu_copy_a_aux_bc_ref(const T *a, T *aux_bc,
                                   const int *lrs_save, const int *lre_save,
                                   const int *n_aux_bc_save,
                                   int noff, int nblk, int lda, int n_size)
{
  for (int n = 0; n < n_size; n++) {
    int lrs      = lrs_save[n];
    int lre      = lre_save[n];
    int n_aux_bc = n_aux_bc_save[n];
    for (int i = 0; i < lre - lrs + 1; i++)
      aux_bc[n_aux_bc + i] = a[(lrs - 1) + i + lda * (noff * nblk + n)];
  }
}

// gpu_copy_aux_bc_aux_mat_loop reference:
//   aux_mat[(lrs_save[n]-1)+i + l_rows*(nstor0+n-1)] = aux_bc[n_aux_bc_save[n]+i]
//   for each block n in [0,n_size), i in [0, lre_save[n]-lrs_save[n]].
//   Early exit if n_size <= 0.
template <typename T>
static void cpu_copy_aux_bc_aux_mat_ref(const T *aux_bc, T *aux_mat,
                                         const int *lrs_save, const int *lre_save,
                                         const int *n_aux_bc_save,
                                         int nstor0, int l_rows, int n_size)
{
  if (n_size <= 0) return;
  for (int n = 0; n < n_size; n++) {
    int nstor    = nstor0 + n;
    int lrs      = lrs_save[n];
    int lre      = lre_save[n];
    int n_aux_bc = n_aux_bc_save[n];
    for (int i = 0; i < lre - lrs + 1; i++)
      aux_mat[(lrs - 1) + i + l_rows * (nstor - 1)] = aux_bc[n_aux_bc + i];
  }
}

template <typename T>
static int count_diffs(const T *a, const T *b, int n, double tol = 1e-10) {
  int cnt = 0;
  for (int i = 0; i < n; i++)
    if (fabs((double)a[i] - (double)b[i]) > tol) cnt++;
  return cnt;
}

// ============================================================
// ============================================================
// Tests for gpu_copy_tmp2_c
// ============================================================
// ============================================================

// ============================================================
// Test 1: basic copy — nr_done=0, lcs=1, no offsets
//
// Parameters:  nstor=3, lcs=1, lce=3, ldc=3, ldcCols=3, nr_done=0.
// Expected:    c == tmp2 (3×3 matrix copied verbatim).
// Verifies:    the identity path where both arrays start at position 0.
// ============================================================
static void test_copy_tmp2_c_basic()
{
  printf("\ngpu_copy_tmp2_c — basic copy (nr_done=0, lcs=1):\n");

  const int NSTOR=3, LCS=1, LCE=3, LDC=3, LDCCOLS=3, NR_DONE=0;
  const int NCOLS = LCE - LCS + 1;

  double htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS] = {0}, hc[LDC * LDCCOLS] = {0};
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = (j + 1) * 10.0 + (i + 1);  // 11,21,31 / 12,22,32 / 13,23,33

  cpu_copy_tmp2_c_ref<double>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  double *dtmp2 = dev_alloc_upload(htmp2,  NSTOR * NCOLS);
  double *dc    = dev_alloc_zero<double>(LDC * LDCCOLS);

  gpu_copy_tmp2_c<double>(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_basic: 0 diffs vs CPU reference", count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_basic: c[0,0]=tmp2[0,0]=11",     deq(hc[0 + LDC * 0], 11.0));
  REPORT("copy_tmp2_c_basic: c[2,2]=tmp2[2,2]=33",     deq(hc[2 + LDC * 2], 33.0));
  REPORT("copy_tmp2_c_basic: c[1,1]=tmp2[1,1]=22",     deq(hc[1 + LDC * 1], 22.0));

  CUDA_CHECK(cudaFree(dtmp2));
  CUDA_CHECK(cudaFree(dc));
}

// ============================================================
// Test 2: row offset — nr_done shifts destination rows
//
// Parameters:  nstor=2, lcs=1, lce=3, ldc=5, ldcCols=3, nr_done=2.
// Expected:    c[2..3, 0..2] = tmp2[0..1, 0..2].
//              c[0..1, *] and c[4, *] remain zero.
// Verifies:    the nr_done addend in the destination index.
// ============================================================
static void test_copy_tmp2_c_row_offset()
{
  printf("\ngpu_copy_tmp2_c — row offset (nr_done=2):\n");

  const int NSTOR=2, LCS=1, LCE=3, LDC=5, LDCCOLS=3, NR_DONE=2;
  const int NCOLS = LCE - LCS + 1;

  double htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS] = {0}, hc[LDC * LDCCOLS] = {0};
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = (j + 1) * 10.0 + (i + 1);

  cpu_copy_tmp2_c_ref<double>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  double *dtmp2 = dev_alloc_upload(htmp2,  NSTOR * NCOLS);
  double *dc    = dev_alloc_zero<double>(LDC * LDCCOLS);

  gpu_copy_tmp2_c<double>(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_row_off: 0 diffs vs CPU reference",      count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_row_off: c[2,0]=tmp2[0,0]=11",           deq(hc[2 + LDC * 0], 11.0));
  REPORT("copy_tmp2_c_row_off: c[3,2]=tmp2[1,2]=32",           deq(hc[3 + LDC * 2], 32.0));
  REPORT("copy_tmp2_c_row_off: c[0,0]=0 (before nr_done)",     deq(hc[0 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_row_off: c[1,0]=0 (before nr_done)",     deq(hc[1 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_row_off: c[4,0]=0 (past nr_done+nstor)", deq(hc[4 + LDC * 0],  0.0));

  CUDA_CHECK(cudaFree(dtmp2));
  CUDA_CHECK(cudaFree(dc));
}

// ============================================================
// Test 3: column offset — lcs shifts destination columns
//
// Parameters:  nstor=3, lcs=3, lce=5, ldc=3, ldcCols=6, nr_done=0.
// Expected:    c[0..2, 2..4] = tmp2[0..2, 0..2].
//              c[:,0], c[:,1], c[:,5] remain zero.
// Verifies:    the (lcs-1) addend selects the correct destination column base.
// ============================================================
static void test_copy_tmp2_c_col_offset()
{
  printf("\ngpu_copy_tmp2_c — column offset (lcs=3):\n");

  const int NSTOR=3, LCS=3, LCE=5, LDC=3, LDCCOLS=6, NR_DONE=0;
  const int NCOLS = LCE - LCS + 1;

  double htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS] = {0}, hc[LDC * LDCCOLS] = {0};
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = (j + 1) * 10.0 + (i + 1);

  cpu_copy_tmp2_c_ref<double>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  double *dtmp2 = dev_alloc_upload(htmp2,  NSTOR * NCOLS);
  double *dc    = dev_alloc_zero<double>(LDC * LDCCOLS);

  gpu_copy_tmp2_c<double>(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_col_off: 0 diffs vs CPU reference",    count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  // Destination starts at col lcs-1=2 (0-indexed)
  REPORT("copy_tmp2_c_col_off: c[0,2]=tmp2[0,0]=11",        deq(hc[0 + LDC * 2], 11.0));
  REPORT("copy_tmp2_c_col_off: c[2,4]=tmp2[2,2]=33",        deq(hc[2 + LDC * 4], 33.0));
  // Columns before lcs-1 and after lce-1 (0-indexed) must be zero
  REPORT("copy_tmp2_c_col_off: c[0,0]=0 (before lcs)",      deq(hc[0 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_col_off: c[0,1]=0 (before lcs)",      deq(hc[0 + LDC * 1],  0.0));
  REPORT("copy_tmp2_c_col_off: c[0,5]=0 (after lce)",       deq(hc[0 + LDC * 5],  0.0));

  CUDA_CHECK(cudaFree(dtmp2));
  CUDA_CHECK(cudaFree(dc));
}

// ============================================================
// Test 4: combined row+column offset
//
// Parameters:  nstor=2, lcs=2, lce=4, ldc=4, ldcCols=5, nr_done=1.
// Expected:    c[1..2, 1..3] = tmp2[0..1, 0..2].
//              c[0,*], c[3,*], c[*,0], c[*,4] remain zero.
// Verifies:    both offsets apply simultaneously without interfering.
// ============================================================
static void test_copy_tmp2_c_combined_offset()
{
  printf("\ngpu_copy_tmp2_c — combined row+col offset (nr_done=1, lcs=2):\n");

  const int NSTOR=2, LCS=2, LCE=4, LDC=4, LDCCOLS=5, NR_DONE=1;
  const int NCOLS = LCE - LCS + 1;

  double htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS] = {0}, hc[LDC * LDCCOLS] = {0};
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = (j + 1) * 10.0 + (i + 1);

  cpu_copy_tmp2_c_ref<double>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  double *dtmp2 = dev_alloc_upload(htmp2,  NSTOR * NCOLS);
  double *dc    = dev_alloc_zero<double>(LDC * LDCCOLS);

  gpu_copy_tmp2_c<double>(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_combined: 0 diffs vs CPU reference",  count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_combined: c[1,1]=tmp2[0,0]=11",       deq(hc[1 + LDC * 1], 11.0));
  REPORT("copy_tmp2_c_combined: c[2,3]=tmp2[1,2]=32",       deq(hc[2 + LDC * 3], 32.0));
  REPORT("copy_tmp2_c_combined: c[0,0]=0 (untouched)",      deq(hc[0 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_combined: c[3,1]=0 (past nr_done+nstor)", deq(hc[3 + LDC * 1], 0.0));
  REPORT("copy_tmp2_c_combined: c[1,0]=0 (before lcs)",     deq(hc[1 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_combined: c[1,4]=0 (after lce)",      deq(hc[1 + LDC * 4],  0.0));

  CUDA_CHECK(cudaFree(dtmp2));
  CUDA_CHECK(cudaFree(dc));
}

// ============================================================
// Test 5: float precision
//
// Same geometry as test 1 but using float arrays.
// Verifies the T=float template instantiation.
// ============================================================
static void test_copy_tmp2_c_float()
{
  printf("\ngpu_copy_tmp2_c<float> — basic copy:\n");

  const int NSTOR=3, LCS=1, LCE=3, LDC=3, LDCCOLS=3, NR_DONE=0;
  const int NCOLS = LCE - LCS + 1;

  float htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS] = {0.0f}, hc[LDC * LDCCOLS] = {0.0f};
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = (float)((j + 1) * 10 + (i + 1));

  cpu_copy_tmp2_c_ref<float>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  float *dtmp2 = dev_alloc_upload(htmp2,  NSTOR * NCOLS);
  float *dc    = dev_alloc_zero<float>(LDC * LDCCOLS);

  gpu_copy_tmp2_c<float>(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hc, dc, LDC * LDCCOLS);

  int ndiffs = 0;
  for (int k = 0; k < LDC * LDCCOLS; k++) if (!feq(hc[k], hc_ref[k])) ndiffs++;

  REPORT("copy_tmp2_c_float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("copy_tmp2_c_float: c[0,0]=11.0f",             feq(hc[0 + LDC * 0], 11.0f));
  REPORT("copy_tmp2_c_float: c[2,2]=33.0f",             feq(hc[2 + LDC * 2], 33.0f));

  CUDA_CHECK(cudaFree(dtmp2));
  CUDA_CHECK(cudaFree(dc));
}

// ============================================================
// ============================================================
// Tests for gpu_copy_a_aux_bc_loop
// ============================================================
// ============================================================

// ============================================================
// Test 6: single block, full column — n_size=1, all rows
//
// Parameters:  noff=0, nblk=4, lda=4, n_size=1.
//              lrs_save=[1], lre_save=[4], n_aux_bc_save=[0].
// Analysis:    n=0: column = noff*nblk+0 = 0 (0-indexed), rows 0..3.
//              aux_bc[0..3] = a[0..3, 0].
// Values:      a[r, c] = (c+1)*10 + (r+1), so a[:,0] = 11,12,13,14.
// ============================================================
static void test_copy_a_aux_bc_single_full()
{
  printf("\ngpu_copy_a_aux_bc_loop — single block, full column:\n");

  const int LDA=4, NBLK=4, N_SIZE=1;
  const int noff=0;
  const int AUX_BC_SIZE = 4;

  double ha[LDA * 4], haux_bc_ref[AUX_BC_SIZE] = {0}, haux_bc[AUX_BC_SIZE] = {0};
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < LDA; r++)
      ha[r + LDA * c] = (c + 1) * 10.0 + (r + 1);

  int hlrs[N_SIZE]       = {1};
  int hlre[N_SIZE]       = {4};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_a_aux_bc_ref<double>(ha, haux_bc_ref, hlrs, hlre, hn_aux_bc, noff, NBLK, LDA, N_SIZE);

  double *da    = dev_alloc_upload(ha,     LDA * 4);
  double *dabc  = dev_alloc_zero<double>(AUX_BC_SIZE);
  int    *dlrs  = dev_alloc_upload(hlrs,   N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,   N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_a_aux_bc_loop<double>(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_single_full: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_single_full: aux_bc[0]=a[0,0]=11",     deq(haux_bc[0], 11.0));
  REPORT("a_aux_bc_single_full: aux_bc[3]=a[3,0]=14",     deq(haux_bc[3], 14.0));

  CUDA_CHECK(cudaFree(da));  CUDA_CHECK(cudaFree(dabc));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 7: single block, partial rows (a_lower/a_upper scenario)
//
// Parameters:  noff=0, nblk=4, lda=4, n_size=1.
//              lrs_save=[2], lre_save=[3], n_aux_bc_save=[0].
// Analysis:    n=0: column 0, rows 1..2 (0-indexed) → aux_bc[0..1].
//              a[1,0]=12, a[2,0]=13 → aux_bc=[12,13].
// Verifies:    lrs and lre trim the gathered row range.
// ============================================================
static void test_copy_a_aux_bc_partial_rows()
{
  printf("\ngpu_copy_a_aux_bc_loop — partial rows (lrs=2, lre=3):\n");

  const int LDA=4, NBLK=4, N_SIZE=1;
  const int noff=0;
  const int AUX_BC_SIZE = 2;

  double ha[LDA * 4], haux_bc_ref[AUX_BC_SIZE] = {0}, haux_bc[AUX_BC_SIZE] = {0};
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < LDA; r++)
      ha[r + LDA * c] = (c + 1) * 10.0 + (r + 1);

  int hlrs[N_SIZE]       = {2};
  int hlre[N_SIZE]       = {3};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_a_aux_bc_ref<double>(ha, haux_bc_ref, hlrs, hlre, hn_aux_bc, noff, NBLK, LDA, N_SIZE);

  double *da    = dev_alloc_upload(ha,     LDA * 4);
  double *dabc  = dev_alloc_zero<double>(AUX_BC_SIZE);
  int    *dlrs  = dev_alloc_upload(hlrs,   N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,   N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_a_aux_bc_loop<double>(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_partial: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_partial: aux_bc[0]=a[1,0]=12",     deq(haux_bc[0], 12.0));
  REPORT("a_aux_bc_partial: aux_bc[1]=a[2,0]=13",     deq(haux_bc[1], 13.0));

  CUDA_CHECK(cudaFree(da));  CUDA_CHECK(cudaFree(dabc));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 8: multiple blocks — two different columns packed into aux_bc
//
// Parameters:  noff=0, nblk=4, lda=4, n_size=2.
//              n=0: lrs=1, lre=3, n_aux_bc=0 → column 0, rows 0..2 → aux_bc[0..2].
//              n=1: lrs=2, lre=4, n_aux_bc=3 → column 1, rows 1..3 → aux_bc[3..5].
// Values:      a[r,0]=r+11, a[r,1]=r+21 (0-indexed r).
//              aux_bc = [11,12,13, 22,23,24].
// ============================================================
static void test_copy_a_aux_bc_multi_block()
{
  printf("\ngpu_copy_a_aux_bc_loop — multi-block (n_size=2):\n");

  const int LDA=4, NBLK=4, N_SIZE=2;
  const int noff=0;
  const int AUX_BC_SIZE = 6;

  double ha[LDA * 2], haux_bc_ref[AUX_BC_SIZE] = {0}, haux_bc[AUX_BC_SIZE] = {0};
  for (int c = 0; c < 2; c++)
    for (int r = 0; r < LDA; r++)
      ha[r + LDA * c] = (c + 1) * 10.0 + (r + 1);

  int hlrs[N_SIZE]       = {1, 2};
  int hlre[N_SIZE]       = {3, 4};
  int hn_aux_bc[N_SIZE]  = {0, 3};

  cpu_copy_a_aux_bc_ref<double>(ha, haux_bc_ref, hlrs, hlre, hn_aux_bc, noff, NBLK, LDA, N_SIZE);

  double *da    = dev_alloc_upload(ha,     LDA * 2);
  double *dabc  = dev_alloc_zero<double>(AUX_BC_SIZE);
  int    *dlrs  = dev_alloc_upload(hlrs,   N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,   N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_a_aux_bc_loop<double>(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_multi: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  // Block n=0: column 0, rows 0..2 → aux_bc[0..2]
  REPORT("a_aux_bc_multi: aux_bc[0]=a[0,0]=11",     deq(haux_bc[0], 11.0));
  REPORT("a_aux_bc_multi: aux_bc[2]=a[2,0]=13",     deq(haux_bc[2], 13.0));
  // Block n=1: column 1, rows 1..3 → aux_bc[3..5]
  REPORT("a_aux_bc_multi: aux_bc[3]=a[1,1]=22",     deq(haux_bc[3], 22.0));
  REPORT("a_aux_bc_multi: aux_bc[5]=a[3,1]=24",     deq(haux_bc[5], 24.0));

  CUDA_CHECK(cudaFree(da));  CUDA_CHECK(cudaFree(dabc));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 9: noff > 0 — column index shifts by noff*nblk
//
// Parameters:  noff=1, nblk=4, lda=4, n_size=1.
//              lrs_save=[1], lre_save=[4], n_aux_bc_save=[0].
// Analysis:    n=0: column = noff*nblk + 0 = 4 (0-indexed).
//              a is 4×8; aux_bc[0..3] = a[:,4] = {51,52,53,54}.
// Verifies:    noff correctly selects a non-first block column.
// ============================================================
static void test_copy_a_aux_bc_noff()
{
  printf("\ngpu_copy_a_aux_bc_loop — noff=1 (column offset by noff*nblk):\n");

  const int LDA=4, NBLK=4, N_SIZE=1;
  const int noff=1;
  const int NCOLS=8;
  const int AUX_BC_SIZE = 4;

  double ha[LDA * NCOLS], haux_bc_ref[AUX_BC_SIZE] = {0}, haux_bc[AUX_BC_SIZE] = {0};
  for (int c = 0; c < NCOLS; c++)
    for (int r = 0; r < LDA; r++)
      ha[r + LDA * c] = (c + 1) * 10.0 + (r + 1);

  int hlrs[N_SIZE]       = {1};
  int hlre[N_SIZE]       = {4};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_a_aux_bc_ref<double>(ha, haux_bc_ref, hlrs, hlre, hn_aux_bc, noff, NBLK, LDA, N_SIZE);

  double *da    = dev_alloc_upload(ha,     LDA * NCOLS);
  double *dabc  = dev_alloc_zero<double>(AUX_BC_SIZE);
  int    *dlrs  = dev_alloc_upload(hlrs,   N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,   N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_a_aux_bc_loop<double>(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_noff: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  // noff*nblk+0 = 4 (0-indexed column), values 51..54
  REPORT("a_aux_bc_noff: aux_bc[0]=a[0,4]=51",     deq(haux_bc[0], 51.0));
  REPORT("a_aux_bc_noff: aux_bc[3]=a[3,4]=54",     deq(haux_bc[3], 54.0));

  CUDA_CHECK(cudaFree(da));  CUDA_CHECK(cudaFree(dabc));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 10: float precision
//
// Same geometry as test 6 but using float arrays.
// Verifies the T=float template instantiation.
// ============================================================
static void test_copy_a_aux_bc_float()
{
  printf("\ngpu_copy_a_aux_bc_loop<float> — basic:\n");

  const int LDA=4, NBLK=4, N_SIZE=1;
  const int noff=0;
  const int AUX_BC_SIZE = 4;

  float ha[LDA * 4], haux_bc_ref[AUX_BC_SIZE] = {0.0f}, haux_bc[AUX_BC_SIZE] = {0.0f};
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < LDA; r++)
      ha[r + LDA * c] = (float)((c + 1) * 10 + (r + 1));

  int hlrs[N_SIZE]       = {1};
  int hlre[N_SIZE]       = {4};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_a_aux_bc_ref<float>(ha, haux_bc_ref, hlrs, hlre, hn_aux_bc, noff, NBLK, LDA, N_SIZE);

  float *da    = dev_alloc_upload(ha,     LDA * 4);
  float *dabc  = dev_alloc_zero<float>(AUX_BC_SIZE);
  int   *dlrs  = dev_alloc_upload(hlrs,   N_SIZE);
  int   *dlre  = dev_alloc_upload(hlre,   N_SIZE);
  int   *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_a_aux_bc_loop<float>(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  int ndiffs = 0;
  for (int k = 0; k < AUX_BC_SIZE; k++) if (!feq(haux_bc[k], haux_bc_ref[k])) ndiffs++;

  REPORT("a_aux_bc_float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("a_aux_bc_float: aux_bc[0]=11.0f",          feq(haux_bc[0], 11.0f));
  REPORT("a_aux_bc_float: aux_bc[3]=14.0f",          feq(haux_bc[3], 14.0f));

  CUDA_CHECK(cudaFree(da));  CUDA_CHECK(cudaFree(dabc));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// ============================================================
// Tests for gpu_copy_aux_bc_aux_mat_loop
// ============================================================
// ============================================================

// ============================================================
// Test 11: basic scatter — n_size=1, full column, nstor0=1
//
// Parameters:  n_size=1, nstor0=1, l_rows=4.
//              lrs_save=[1], lre_save=[4], n_aux_bc_save=[0].
// Analysis:    n=0: nstor=1, rows 0..3, aux_bc[0..3] → aux_mat col 0.
// Values:      aux_bc = [1,2,3,4]; aux_mat col 0 = [1,2,3,4].
// ============================================================
static void test_copy_aux_bc_aux_mat_basic()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop — basic scatter (n_size=1, nstor0=1):\n");

  const int L_ROWS=4, N_SIZE=1, NCOLS_AUX=2;
  const int nstor0=1;
  const int AUX_BC_SIZE = 4;

  double haux_bc[AUX_BC_SIZE], haux_mat_ref[L_ROWS * NCOLS_AUX] = {0}, haux_mat[L_ROWS * NCOLS_AUX] = {0};
  for (int i = 0; i < AUX_BC_SIZE; i++) haux_bc[i] = (double)(i + 1);

  int hlrs[N_SIZE]       = {1};
  int hlre[N_SIZE]       = {4};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_aux_bc_aux_mat_ref<double>(haux_bc, haux_mat_ref, hlrs, hlre, hn_aux_bc, nstor0, L_ROWS, N_SIZE);

  double *dabc  = dev_alloc_upload(haux_bc,    AUX_BC_SIZE);
  double *damat = dev_alloc_zero<double>(L_ROWS * NCOLS_AUX);
  int    *dlrs  = dev_alloc_upload(hlrs,   N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,   N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_aux_bc_aux_mat_loop<double>(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_basic: 0 diffs vs CPU reference", count_diffs(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  // aux_mat col 0 (nstor=1): rows 0..3
  REPORT("aux_bc_aux_mat_basic: aux_mat[0,0]=aux_bc[0]=1", deq(haux_mat[0 + L_ROWS * 0], 1.0));
  REPORT("aux_bc_aux_mat_basic: aux_mat[3,0]=aux_bc[3]=4", deq(haux_mat[3 + L_ROWS * 0], 4.0));
  // Second column must remain zero
  REPORT("aux_bc_aux_mat_basic: aux_mat[0,1]=0",           deq(haux_mat[0 + L_ROWS * 1], 0.0));

  CUDA_CHECK(cudaFree(dabc)); CUDA_CHECK(cudaFree(damat));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 12: multiple blocks scatter into different columns
//
// Parameters:  n_size=2, nstor0=1, l_rows=4.
//              n=0: nstor=1, lrs=1, lre=2, n_aux_bc=0 → col 0, rows 0..1.
//              n=1: nstor=2, lrs=3, lre=4, n_aux_bc=2 → col 1, rows 2..3.
// Values:      aux_bc=[10,20,30,40].
//              aux_mat[:,0] has [10,20,0,0]; aux_mat[:,1] has [0,0,30,40].
// ============================================================
static void test_copy_aux_bc_aux_mat_multi_block()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop — multi-block (n_size=2, nstor0=1):\n");

  const int L_ROWS=4, N_SIZE=2, NCOLS_AUX=2;
  const int nstor0=1;
  const int AUX_BC_SIZE = 4;

  double haux_bc[AUX_BC_SIZE] = {10.0, 20.0, 30.0, 40.0};
  double haux_mat_ref[L_ROWS * NCOLS_AUX] = {0}, haux_mat[L_ROWS * NCOLS_AUX] = {0};

  int hlrs[N_SIZE]      = {1, 3};
  int hlre[N_SIZE]      = {2, 4};
  int hn_aux_bc[N_SIZE] = {0, 2};

  cpu_copy_aux_bc_aux_mat_ref<double>(haux_bc, haux_mat_ref, hlrs, hlre, hn_aux_bc, nstor0, L_ROWS, N_SIZE);

  double *dabc  = dev_alloc_upload(haux_bc,  AUX_BC_SIZE);
  double *damat = dev_alloc_zero<double>(L_ROWS * NCOLS_AUX);
  int    *dlrs  = dev_alloc_upload(hlrs,     N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,     N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_aux_bc_aux_mat_loop<double>(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_multi: 0 diffs vs CPU reference",    count_diffs(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  // n=0: nstor=1, rows 0..1 of col 0
  REPORT("aux_bc_aux_mat_multi: aux_mat[0,0]=aux_bc[0]=10",  deq(haux_mat[0 + L_ROWS * 0], 10.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[1,0]=aux_bc[1]=20",  deq(haux_mat[1 + L_ROWS * 0], 20.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[2,0]=0 (not written)", deq(haux_mat[2 + L_ROWS * 0], 0.0));
  // n=1: nstor=2, rows 2..3 of col 1
  REPORT("aux_bc_aux_mat_multi: aux_mat[2,1]=aux_bc[2]=30",  deq(haux_mat[2 + L_ROWS * 1], 30.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[3,1]=aux_bc[3]=40",  deq(haux_mat[3 + L_ROWS * 1], 40.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[0,1]=0 (not written)", deq(haux_mat[0 + L_ROWS * 1], 0.0));

  CUDA_CHECK(cudaFree(dabc)); CUDA_CHECK(cudaFree(damat));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 13: nstor0 offset — scatter into non-first columns
//
// Parameters:  n_size=1, nstor0=3, l_rows=6.
//              lrs_save=[2], lre_save=[5], n_aux_bc_save=[0].
// Analysis:    n=0: nstor=3, rows 1..4 (0-indexed), col 2 (0-indexed).
//              aux_bc[0..3] → aux_mat[1..4, 2].
// Verifies:    the nstor0 addend places data in the correct output column.
// ============================================================
static void test_copy_aux_bc_aux_mat_nstor_offset()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop — nstor0 offset (nstor0=3, col index 2):\n");

  const int L_ROWS=6, N_SIZE=1, NCOLS_AUX=4;
  const int nstor0=3;
  const int AUX_BC_SIZE = 4;

  double haux_bc[AUX_BC_SIZE] = {100.0, 200.0, 300.0, 400.0};
  double haux_mat_ref[L_ROWS * NCOLS_AUX] = {0}, haux_mat[L_ROWS * NCOLS_AUX] = {0};

  int hlrs[N_SIZE]       = {2};
  int hlre[N_SIZE]       = {5};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_aux_bc_aux_mat_ref<double>(haux_bc, haux_mat_ref, hlrs, hlre, hn_aux_bc, nstor0, L_ROWS, N_SIZE);

  double *dabc  = dev_alloc_upload(haux_bc,  AUX_BC_SIZE);
  double *damat = dev_alloc_zero<double>(L_ROWS * NCOLS_AUX);
  int    *dlrs  = dev_alloc_upload(hlrs,     N_SIZE);
  int    *dlre  = dev_alloc_upload(hlre,     N_SIZE);
  int    *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_aux_bc_aux_mat_loop<double>(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_nstor: 0 diffs vs CPU reference",     count_diffs(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  // nstor=3 → col index 2 (0-indexed); lrs=2 → row index 1 (0-indexed)
  REPORT("aux_bc_aux_mat_nstor: aux_mat[1,2]=aux_bc[0]=100",   deq(haux_mat[1 + L_ROWS * 2], 100.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[4,2]=aux_bc[3]=400",   deq(haux_mat[4 + L_ROWS * 2], 400.0));
  // Other columns remain zero
  REPORT("aux_bc_aux_mat_nstor: aux_mat[1,0]=0 (wrong col)",   deq(haux_mat[1 + L_ROWS * 0], 0.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[1,1]=0 (wrong col)",   deq(haux_mat[1 + L_ROWS * 1], 0.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[0,2]=0 (before lrs)",  deq(haux_mat[0 + L_ROWS * 2], 0.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[5,2]=0 (after lre)",   deq(haux_mat[5 + L_ROWS * 2], 0.0));

  CUDA_CHECK(cudaFree(dabc)); CUDA_CHECK(cudaFree(damat));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 14: n_size=0 early-return guard
//
// The launcher has `if (n_size <= 0) return;` before calling the kernel.
// With n_size=0, aux_mat must remain all-zero.
// ============================================================
static void test_copy_aux_bc_aux_mat_empty()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop — n_size=0 early return:\n");

  const int L_ROWS=4, N_SIZE=0, NCOLS_AUX=2;
  const int nstor0=1;
  const int AUX_BC_SIZE = 4;

  double haux_bc[AUX_BC_SIZE] = {1.0, 2.0, 3.0, 4.0};
  double haux_mat[L_ROWS * NCOLS_AUX] = {0};

  // No lrs/lre/n_aux_bc arrays needed for n_size=0, but launchers must
  // still receive valid device pointers (they are not dereferenced).
  int dummy_hlrs[1] = {1}, dummy_hlre[1] = {1}, dummy_hn[1] = {0};

  double *dabc  = dev_alloc_upload(haux_bc,   AUX_BC_SIZE);
  double *damat = dev_alloc_zero<double>(L_ROWS * NCOLS_AUX);
  int    *dlrs  = dev_alloc_upload(dummy_hlrs, 1);
  int    *dlre  = dev_alloc_upload(dummy_hlre, 1);
  int    *dnabc = dev_alloc_upload(dummy_hn,   1);

  gpu_copy_aux_bc_aux_mat_loop<double>(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  bool all_zero = true;
  for (int k = 0; k < L_ROWS * NCOLS_AUX; k++) if (!deq(haux_mat[k], 0.0)) { all_zero = false; break; }

  REPORT("aux_bc_aux_mat_empty: aux_mat all zero when n_size=0", all_zero);

  CUDA_CHECK(cudaFree(dabc)); CUDA_CHECK(cudaFree(damat));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Test 15: float precision
//
// Same geometry as test 11 but using float arrays.
// Verifies the T=float template instantiation.
// ============================================================
static void test_copy_aux_bc_aux_mat_float()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop<float> — basic:\n");

  const int L_ROWS=4, N_SIZE=1, NCOLS_AUX=2;
  const int nstor0=1;
  const int AUX_BC_SIZE = 4;

  float haux_bc[AUX_BC_SIZE], haux_mat_ref[L_ROWS * NCOLS_AUX] = {0.0f}, haux_mat[L_ROWS * NCOLS_AUX] = {0.0f};
  for (int i = 0; i < AUX_BC_SIZE; i++) haux_bc[i] = (float)(i + 1);

  int hlrs[N_SIZE]       = {1};
  int hlre[N_SIZE]       = {4};
  int hn_aux_bc[N_SIZE]  = {0};

  cpu_copy_aux_bc_aux_mat_ref<float>(haux_bc, haux_mat_ref, hlrs, hlre, hn_aux_bc, nstor0, L_ROWS, N_SIZE);

  float *dabc  = dev_alloc_upload(haux_bc,   AUX_BC_SIZE);
  float *damat = dev_alloc_zero<float>(L_ROWS * NCOLS_AUX);
  int   *dlrs  = dev_alloc_upload(hlrs,  N_SIZE);
  int   *dlre  = dev_alloc_upload(hlre,  N_SIZE);
  int   *dnabc = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_aux_bc_aux_mat_loop<float>(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  int ndiffs = 0;
  for (int k = 0; k < L_ROWS * NCOLS_AUX; k++) if (!feq(haux_mat[k], haux_mat_ref[k])) ndiffs++;

  REPORT("aux_bc_aux_mat_float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("aux_bc_aux_mat_float: aux_mat[0,0]=1.0f",        feq(haux_mat[0 + L_ROWS * 0], 1.0f));
  REPORT("aux_bc_aux_mat_float: aux_mat[3,0]=4.0f",        feq(haux_mat[3 + L_ROWS * 0], 4.0f));

  CUDA_CHECK(cudaFree(dabc)); CUDA_CHECK(cudaFree(damat));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

// ============================================================
// Half-precision tests
// ============================================================

#ifdef WANT_HALF_PRECISION_REAL

static bool heq(__half a, __half b, float tol = 0.02f) {
    float fa = __half2float(a), fb = __half2float(b);
    return fabsf(fa - fb) < tol * (fabsf(fb) + 1.0f);
}

// Count differences in a __half array
static int count_diffs_h(const __half *a, const __half *b, int n) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (!heq(a[i], b[i])) cnt++;
    return cnt;
}

static void test_copy_tmp2_c_half()
{
  printf("\ngpu_copy_tmp2_c<__half> — basic copy:\n");

  const int NSTOR=3, LCS=1, LCE=3, LDC=3, LDCCOLS=3, NR_DONE=0;
  const int NCOLS = LCE - LCS + 1;

  __half htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS], hc[LDC * LDCCOLS];
  for (int k = 0; k < LDC * LDCCOLS; k++) hc_ref[k] = hc[k] = __float2half(0.0f);
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = __float2half((float)((j + 1) * 10 + (i + 1)));

  cpu_copy_tmp2_c_ref<__half>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  __half *dtmp2 = dev_alloc_upload(htmp2, NSTOR * NCOLS);
  __half *dc    = dev_alloc_zero<__half>(LDC * LDCCOLS);

  gpu_copy_tmp2_c<__half>(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_half: 0 diffs vs CPU reference", count_diffs_h(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_half: c[0,0]=11", heq(hc[0 + LDC * 0], __float2half(11.0f)));
  REPORT("copy_tmp2_c_half: c[2,2]=33", heq(hc[2 + LDC * 2], __float2half(33.0f)));

  CUDA_CHECK(cudaFree(dtmp2));
  CUDA_CHECK(cudaFree(dc));
}

static void test_copy_a_aux_bc_half()
{
  printf("\ngpu_copy_a_aux_bc_loop<__half> — basic:\n");

  const int LDA=4, NBLK=4, N_SIZE=1;
  const int noff=0;
  const int AUX_BC_SIZE = 4;

  __half ha[LDA * 4], haux_bc_ref[AUX_BC_SIZE], haux_bc[AUX_BC_SIZE];
  for (int k = 0; k < AUX_BC_SIZE; k++) haux_bc_ref[k] = haux_bc[k] = __float2half(0.0f);
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < LDA; r++)
      ha[r + LDA * c] = __float2half((float)((c + 1) * 10 + (r + 1)));

  int hlrs[N_SIZE] = {1}, hlre[N_SIZE] = {4}, hn_aux_bc[N_SIZE] = {0};

  cpu_copy_a_aux_bc_ref<__half>(ha, haux_bc_ref, hlrs, hlre, hn_aux_bc, noff, NBLK, LDA, N_SIZE);

  __half *da   = dev_alloc_upload(ha, LDA * 4);
  __half *dabc = dev_alloc_zero<__half>(AUX_BC_SIZE);
  int *dlrs    = dev_alloc_upload(hlrs,  N_SIZE);
  int *dlre    = dev_alloc_upload(hlre,  N_SIZE);
  int *dnabc   = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_a_aux_bc_loop<__half>(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_half: 0 diffs vs CPU reference", count_diffs_h(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_half: aux_bc[0]=11", heq(haux_bc[0], __float2half(11.0f)));
  REPORT("a_aux_bc_half: aux_bc[3]=14", heq(haux_bc[3], __float2half(14.0f)));

  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(dabc));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

static void test_copy_aux_bc_aux_mat_half()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop<__half> — basic:\n");

  const int L_ROWS=4, N_SIZE=1, NCOLS_AUX=2;
  const int nstor0=1;
  const int AUX_BC_SIZE = 4;

  __half haux_bc[AUX_BC_SIZE], haux_mat_ref[L_ROWS * NCOLS_AUX], haux_mat[L_ROWS * NCOLS_AUX];
  for (int k = 0; k < L_ROWS * NCOLS_AUX; k++) haux_mat_ref[k] = haux_mat[k] = __float2half(0.0f);
  for (int i = 0; i < AUX_BC_SIZE; i++) haux_bc[i] = __float2half((float)(i + 1));

  int hlrs[N_SIZE] = {1}, hlre[N_SIZE] = {4}, hn_aux_bc[N_SIZE] = {0};

  cpu_copy_aux_bc_aux_mat_ref<__half>(haux_bc, haux_mat_ref, hlrs, hlre, hn_aux_bc, nstor0, L_ROWS, N_SIZE);

  __half *dabc  = dev_alloc_upload(haux_bc,  AUX_BC_SIZE);
  __half *damat = dev_alloc_zero<__half>(L_ROWS * NCOLS_AUX);
  int *dlrs     = dev_alloc_upload(hlrs,  N_SIZE);
  int *dlre     = dev_alloc_upload(hlre,  N_SIZE);
  int *dnabc    = dev_alloc_upload(hn_aux_bc, N_SIZE);

  gpu_copy_aux_bc_aux_mat_loop<__half>(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_half: 0 diffs vs CPU reference", count_diffs_h(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  REPORT("aux_bc_aux_mat_half: aux_mat[0,0]=1", heq(haux_mat[0 + L_ROWS * 0], __float2half(1.0f)));
  REPORT("aux_bc_aux_mat_half: aux_mat[3,0]=4", heq(haux_mat[3 + L_ROWS * 0], __float2half(4.0f)));
  REPORT("aux_bc_aux_mat_half: aux_mat[0,1]=0", heq(haux_mat[0 + L_ROWS * 1], __float2half(0.0f)));

  CUDA_CHECK(cudaFree(dabc)); CUDA_CHECK(cudaFree(damat));
  CUDA_CHECK(cudaFree(dlrs)); CUDA_CHECK(cudaFree(dlre)); CUDA_CHECK(cudaFree(dnabc));
}

#endif /* WANT_HALF_PRECISION_REAL */

// ============================================================
int main(void)
{
  printf("=== Unit tests for elpa_hermitian_multiply_gpu.h (CUDA) ===\n");

  printf("\n--- gpu_copy_tmp2_c ---\n");
  test_copy_tmp2_c_basic();
  test_copy_tmp2_c_row_offset();
  test_copy_tmp2_c_col_offset();
  test_copy_tmp2_c_combined_offset();
  test_copy_tmp2_c_float();

  printf("\n--- gpu_copy_a_aux_bc_loop ---\n");
  test_copy_a_aux_bc_single_full();
  test_copy_a_aux_bc_partial_rows();
  test_copy_a_aux_bc_multi_block();
  test_copy_a_aux_bc_noff();
  test_copy_a_aux_bc_float();

  printf("\n--- gpu_copy_aux_bc_aux_mat_loop ---\n");
  test_copy_aux_bc_aux_mat_basic();
  test_copy_aux_bc_aux_mat_multi_block();
  test_copy_aux_bc_aux_mat_nstor_offset();
  test_copy_aux_bc_aux_mat_empty();
  test_copy_aux_bc_aux_mat_float();

#ifdef WANT_HALF_PRECISION_REAL
  printf("\n--- gpu_copy_tmp2_c<__half> ---\n");
  test_copy_tmp2_c_half();
  printf("\n--- gpu_copy_a_aux_bc_loop<__half> ---\n");
  test_copy_a_aux_bc_half();
  printf("\n--- gpu_copy_aux_bc_aux_mat_loop<__half> ---\n");
  test_copy_aux_bc_aux_mat_half();
#endif /* WANT_HALF_PRECISION_REAL */

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
