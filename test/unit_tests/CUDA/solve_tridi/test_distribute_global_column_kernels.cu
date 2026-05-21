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

// Unit tests for gpu_distribute_global_column (CUDA backend).
//
// The kernel implements the Fortran subroutine distribute_global_column_4:
// given a global column g_col(1:g_col_dim1, 1:g_col_dim2), it scatters
// rows belonging to this MPI process rank (my_prow, np_rows, nblk) into
// the distributed local matrix l_col(1:ldq, 1:matrixCols).
//
// Both arrays are col-major (Fortran order); the kernel uses the local_to_global
// mapping to convert a linear index (starting from a given (row,col) offset) to
// a 2-D (row,col) position.
//
// Key algorithmic details:
//   nbs = noff / (nblk * np_rows)          -- first block touched
//   nbe = (noff + nlen - 1) / (nblk * np_rows) -- last block touched
//   For jb in [nbs, nbe]:
//     g_off = jb * nblk * np_rows + nblk * my_prow  -- global row base
//     l_off = jb * nblk                             -- local  row base
//     js = max(noff + 1 - g_off, 1)
//     je = min(noff + nlen - g_off, nblk)
//     For index in [js, je]:
//       local_to_global maps (g_off - noff + index) → (g_row, g_col) in g_col
//       local_to_global maps (l_off + index)         → (l_row, l_col) in l_col
//       l_col(l_row, l_col) = g_col(g_row, g_col)
//
// GPU launcher block dimensions: threadsPerBlock(32, 16, 2).
// Requirement: matrixCols >= nblk (ensures the z-dimension covers [js..je]).

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
#include "../../../../src/solve_tridi/GPU/gpu_distribute_global_column.h"

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
// CPU reference: exact translation of Fortran distribute_global_column_4
// and the local_to_global helper it uses.
//
// local_to_global(dim1, dim2, n, m, local_index) → (row, col):
//   Starting from position (n, m) in a col-major dim1×dim2 array,
//   count local_index elements and return the resulting 1-indexed (row, col).
// ============================================================

static void local_to_global_ref(int dim1, int /*dim2*/, int n, int m,
                                 int local_index, int &row, int &col) {
  int esc = dim1 - n + 1;  // entries in starting column from row n onward
  if (local_index > esc) {
    int idx_sub  = local_index - esc;
    int col_sub  = (idx_sub - 1) / dim1 + 1;
    int row_sub  = idx_sub % dim1;
    if (row_sub == 0) row_sub = dim1;
    col = col_sub + m;
    row = row_sub;
  } else {
    col = m;
    row = n + local_index - 1;
  }
}

template <typename T>
static void cpu_distribute_ref(
    const T *g_col, T *l_col,
    int g_col_dim1, int g_col_dim2,
    int ldq,        int matrixCols,
    int noff_in, int noff, int nlen,
    int my_prow, int np_rows, int nblk)
{
  int nbs = noff / (nblk * np_rows);
  int nbe = (noff + nlen - 1) / (nblk * np_rows);

  for (int i = 1; i <= nlen; i++) {
    // 1-indexed starting positions in each 2-D array
    int g_off2 = i;           // g_col_offset2 = i,    g_col_offset1 = 1
    int l_off2 = noff_in + i; // l_col_offset2 = noff_in+i, l_col_offset1 = 1

    for (int jb = nbs; jb <= nbe; jb++) {
      int g_off = jb * nblk * np_rows + nblk * my_prow;
      int l_off = jb * nblk;

      int js = std::max(noff + 1 - g_off, 1);
      int je = std::min(noff + nlen - g_off, nblk);
      if (je < js) continue;

      for (int index = js; index <= je; index++) {
        int g_row, g_c, l_row, l_c;
        local_to_global_ref(g_col_dim1, g_col_dim2, 1, g_off2,
                            g_off - noff + index, g_row, g_c);
        local_to_global_ref(ldq,        matrixCols, 1, l_off2,
                            l_off + index,        l_row, l_c);
        l_col[(l_row - 1) + ldq * (l_c - 1)] =
            g_col[(g_row - 1) + g_col_dim1 * (g_c - 1)];
      }
    }
  }
}

// Count element-wise differences between two arrays.
template <typename T>
static int count_diffs(const T *a, const T *b, int n, double tol = 1e-10) {
  int cnt = 0;
  for (int i = 0; i < n; i++)
    if (fabs((double)a[i] - (double)b[i]) > tol) cnt++;
  return cnt;
}

// ============================================================
// Test 1: basic copy — noff=0, single process, nblk≥nlen
//
// Parameters:  g_col_dim1=4, ldq=4, nlen=4, nblk=4, np_rows=1, my_prow=0,
//              noff_in=0, noff=0, matrixCols=4.
// Expected:    l_col is a verbatim copy of g_col (both 4×4 matrices).
// Verifies:    the straightforward no-wrap path where g_off=l_off=0.
// ============================================================
static void test_basic_copy()
{
  printf("\ngpu_distribute_global_column — basic copy (noff=0, np_rows=1):\n");

  const int G1=4, G2=4, LDQ=4, MC=4, NLEN=4, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);  // values 11..44

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("basic_copy: 0 diffs vs CPU reference", count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("basic_copy: l_col[0,0]=g_col[0,0]=11", deq(hl[0+LDQ*0], 11.0));
  REPORT("basic_copy: l_col[3,3]=g_col[3,3]=44", deq(hl[3+LDQ*3], 44.0));
  REPORT("basic_copy: l_col[1,2]=g_col[1,2]=32", deq(hl[1+LDQ*2], 32.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 2: noff_in offset — destination columns shifted
//
// Parameters:  same g_col; noff_in=2, matrixCols=8.
// Expected:    l_col columns 2..5 (0-indexed) receive g_col columns 0..3.
//              l_col columns 0..1 and 6..7 remain zero.
// Verifies:    the l_col_offset2 = noff_in + i path.
// ============================================================
static void test_noff_in_offset()
{
  printf("\ngpu_distribute_global_column — noff_in=2 (destination column offset):\n");

  const int G1=4, G2=4, LDQ=4, MC=8, NLEN=4, NBLK=4;
  const int noff_in=2, noff=0, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("noff_in: 0 diffs vs CPU reference",     count_diffs(hl, hl_ref, LDQ*MC) == 0);
  // l_col[:,2] (0-indexed col 2) should equal g_col[:,0]
  REPORT("noff_in: l_col[0,2]=g_col[0,0]=11",     deq(hl[0+LDQ*2], 11.0));
  REPORT("noff_in: l_col[3,5]=g_col[3,3]=44",     deq(hl[3+LDQ*5], 44.0));
  // Untouched columns
  REPORT("noff_in: l_col[0,0]=0 (before offset)", deq(hl[0+LDQ*0],  0.0));
  REPORT("noff_in: l_col[0,1]=0 (before offset)", deq(hl[0+LDQ*1],  0.0));
  REPORT("noff_in: l_col[0,6]=0 (after range)",   deq(hl[0+LDQ*6],  0.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 3: noff row skip — rows land at bottom of first block
//
// Parameters:  g_col_dim1=4, ldq=4, nlen=2, nblk=4, np_rows=1, my_prow=0,
//              noff_in=0, noff=2, matrixCols=4.
// Analysis:    nbs=nbe=0; jb=0: g_off=0, l_off=0, js=3, je=4.
//              For i=1..2, index=3,4:
//                g_local = 0-2+index = index-2 ∈ {1,2} → rows 0,1 of g_col col i-1.
//                l_local = index ∈ {3,4} → rows 2,3 of l_col col i-1.
// Expected:    l_col[2,c] = g_col[0,c],  l_col[3,c] = g_col[1,c]  for c=0,1.
//              l_col[0..1, 0..1] = 0.
// ============================================================
static void test_noff_row_skip()
{
  printf("\ngpu_distribute_global_column — noff=2 (rows land deep in block):\n");

  const int G1=4, G2=2, LDQ=4, MC=4, NLEN=2, NBLK=4;
  const int noff_in=0, noff=2, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);
  // col 0: 11,12,13,14  col 1: 21,22,23,24

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("noff_skip: 0 diffs vs CPU reference",  count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("noff_skip: l_col[2,0]=g_col[0,0]=11",  deq(hl[2+LDQ*0], 11.0));
  REPORT("noff_skip: l_col[3,0]=g_col[1,0]=12",  deq(hl[3+LDQ*0], 12.0));
  REPORT("noff_skip: l_col[2,1]=g_col[0,1]=21",  deq(hl[2+LDQ*1], 21.0));
  REPORT("noff_skip: l_col[3,1]=g_col[1,1]=22",  deq(hl[3+LDQ*1], 22.0));
  REPORT("noff_skip: l_col[0,0]=0 (untouched)",  deq(hl[0+LDQ*0],  0.0));
  REPORT("noff_skip: l_col[1,0]=0 (untouched)",  deq(hl[1+LDQ*0],  0.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 4: two-block span — nblk=4, nlen=8
//
// Parameters:  g_col_dim1=8, ldq=8, nlen=8, nblk=4, np_rows=1, my_prow=0,
//              noff_in=0, noff=0, matrixCols=8.
// Analysis:    nbs=0, nbe=1.  Both jb=0 and jb=1 are fully covered with je=4.
//              All g_local and l_local values are within their leading dimensions,
//              so no wrapping occurs.  Result is a full 8×8 copy.
// Expected:    l_col[r,c] = g_col[r,c] for all r=0..7, c=0..7.
// ============================================================
static void test_two_blocks()
{
  printf("\ngpu_distribute_global_column — two-block span (nlen=8, nblk=4):\n");

  const int G1=8, G2=8, LDQ=8, MC=8, NLEN=8, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("two_blocks: 0 diffs vs CPU reference",  count_diffs(hl, hl_ref, LDQ*MC) == 0);
  // Spot-check corner elements and the block boundary at row 4
  REPORT("two_blocks: l_col[0,0]=11 (jb=0 first row)", deq(hl[0+LDQ*0], 11.0));
  REPORT("two_blocks: l_col[3,0]=14 (jb=0 last row)",  deq(hl[3+LDQ*0], 14.0));
  REPORT("two_blocks: l_col[4,0]=15 (jb=1 first row)", deq(hl[4+LDQ*0], 15.0));
  REPORT("two_blocks: l_col[7,7]=88 (jb=1 last row)",  deq(hl[7+LDQ*7], 88.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 5: two MPI process rows, prow=0
//
// Parameters:  g_col_dim1=8, ldq=8, nlen=8, nblk=4, np_rows=2, my_prow=0,
//              noff_in=0, noff=0, matrixCols=8.
// Analysis:    nbs=nbe=0 (only one block jb=0).
//              g_off = 0, l_off = 0, js=1, je=min(8,4)=4.
//              Only the first nblk=4 rows of each g_col column are copied
//              into the first 4 rows of the corresponding l_col column.
// Expected:    l_col[0..3, c] = g_col[0..3, c];  l_col[4..7, c] = 0.
// ============================================================
static void test_np_rows2_prow0()
{
  printf("\ngpu_distribute_global_column — np_rows=2, my_prow=0:\n");

  const int G1=8, G2=8, LDQ=8, MC=8, NLEN=8, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=2;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("prow0: 0 diffs vs CPU reference", count_diffs(hl, hl_ref, LDQ*MC) == 0);
  // First four rows populated
  REPORT("prow0: l_col[0,0]=g_col[0,0]=11", deq(hl[0+LDQ*0], 11.0));
  REPORT("prow0: l_col[3,7]=g_col[3,7]=84", deq(hl[3+LDQ*7], 84.0));
  // Rows 4..7 must remain zero (belong to prow=1's block)
  REPORT("prow0: l_col[4,0]=0 (prow=1 territory)", deq(hl[4+LDQ*0], 0.0));
  REPORT("prow0: l_col[7,7]=0 (prow=1 territory)", deq(hl[7+LDQ*7], 0.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 6: two MPI process rows, prow=1
//
// Parameters:  same as test 5 but my_prow=1.
// Analysis:    jb=0: g_off = 0*4*2 + 4*1 = 4, l_off=0, js=1, je=4.
//              g_local = g_off + index = 4 + index  (5..8) → g_col rows 4..7.
//              l_local = index (1..4) → l_col rows 0..3.
// Expected:    l_col[0..3, c] = g_col[4..7, c];  l_col[4..7, c] = 0.
// ============================================================
static void test_np_rows2_prow1()
{
  printf("\ngpu_distribute_global_column — np_rows=2, my_prow=1:\n");

  const int G1=8, G2=8, LDQ=8, MC=8, NLEN=8, NBLK=4;
  const int noff_in=0, noff=0, my_prow=1, np_rows=2;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("prow1: 0 diffs vs CPU reference", count_diffs(hl, hl_ref, LDQ*MC) == 0);
  // l_col row 0 should hold g_col row 4 (value = col*10 + 5)
  REPORT("prow1: l_col[0,0]=g_col[4,0]=15", deq(hl[0+LDQ*0], 15.0));
  REPORT("prow1: l_col[3,0]=g_col[7,0]=18", deq(hl[3+LDQ*0], 18.0));
  REPORT("prow1: l_col[0,7]=g_col[4,7]=85", deq(hl[0+LDQ*7], 85.0));
  REPORT("prow1: l_col[3,7]=g_col[7,7]=88", deq(hl[3+LDQ*7], 88.0));
  // Rows 4..7 of l_col not touched
  REPORT("prow1: l_col[4,0]=0 (untouched)", deq(hl[4+LDQ*0], 0.0));
  REPORT("prow1: l_col[7,7]=0 (untouched)", deq(hl[7+LDQ*7], 0.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 7: float precision
//
// Same configuration as test 1 (basic copy) but using float arrays.
// Verifies that the T=float template instantiation is correct.
// ============================================================
static void test_float()
{
  printf("\ngpu_distribute_global_column<float> — basic copy:\n");

  const int G1=4, G2=4, LDQ=4, MC=4, NLEN=4, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  float hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (float)((c+1)*10 + (r+1));

  cpu_distribute_ref<float>(hg, hl_ref, G1, G2, LDQ, MC,
                            noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  float *dg  = dev_alloc_upload(hg,    G1*G2);
  float *dlc = dev_alloc_zero<float>(LDQ*MC);

  gpu_distribute_global_column<float>(dg, dlc, G1, G2, LDQ, MC,
                                      noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                      0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  int ndiffs = 0;
  for (int k=0; k<LDQ*MC; k++) if (!feq(hl[k], hl_ref[k])) ndiffs++;

  REPORT("float: 0 diffs vs CPU reference",  ndiffs == 0);
  REPORT("float: l_col[0,0]=11.0f",          feq(hl[0+LDQ*0], 11.0f));
  REPORT("float: l_col[3,3]=44.0f",          feq(hl[3+LDQ*3], 44.0f));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 8: g_col column wrapping
//
// Parameters:  g_col_dim1=2, g_col_dim2=5, ldq=4, nlen=4, nblk=4,
//              np_rows=1, my_prow=0, noff_in=0, noff=0, matrixCols=4.
// Analysis:    Only jb=0; g_off=0, l_off=0, js=1, je=4.
//              entries_in_started_col = g_col_dim1 = 2.
//              For g_local ≤ 2: straightforward (row=g_local, col=i).
//              For g_local > 2: wraps into the NEXT column of g_col:
//                g_local=3 → (row=1, col=i+1);  g_local=4 → (row=2, col=i+1).
//              l_col has ldq=4, so l_local=1..4 ≤ 4, no wrapping in l_col.
// Expected per outer index i:
//   l_col[0,i-1] = g_col[0,i-1]          (g_local=1, no wrap)
//   l_col[1,i-1] = g_col[1,i-1]          (g_local=2, no wrap)
//   l_col[2,i-1] = g_col[0,i]   (=g_col[0,(i-1)+1])  (g_local=3, wrap)
//   l_col[3,i-1] = g_col[1,i]   (=g_col[1,(i-1)+1])  (g_local=4, wrap)
// ============================================================
static void test_gcol_column_wrap()
{
  printf("\ngpu_distribute_global_column — g_col column wrap (g_col_dim1=2, nblk=4):\n");

  const int G1=2, G2=5, LDQ=4, MC=4, NLEN=4, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  // g_col[r + 2*c] = (c+1)*10 + (r+1)
  // col 0: 11,12  col 1: 21,22  col 2: 31,32  col 3: 41,42  col 4: 51,52
  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("gcol_wrap: 0 diffs vs CPU reference", count_diffs(hl, hl_ref, LDQ*MC) == 0);
  // i=1 (col 0 of l_col): rows 0,1 from g_col[:,0]; rows 2,3 from g_col[:,1]
  REPORT("gcol_wrap: l_col[0,0]=g_col[0,0]=11", deq(hl[0+LDQ*0], 11.0));
  REPORT("gcol_wrap: l_col[1,0]=g_col[1,0]=12", deq(hl[1+LDQ*0], 12.0));
  REPORT("gcol_wrap: l_col[2,0]=g_col[0,1]=21 (wrapped)", deq(hl[2+LDQ*0], 21.0));
  REPORT("gcol_wrap: l_col[3,0]=g_col[1,1]=22 (wrapped)", deq(hl[3+LDQ*0], 22.0));
  // i=4 (col 3 of l_col): rows 0,1 from g_col[:,3]; rows 2,3 from g_col[:,4]
  REPORT("gcol_wrap: l_col[0,3]=g_col[0,3]=41", deq(hl[0+LDQ*3], 41.0));
  REPORT("gcol_wrap: l_col[2,3]=g_col[0,4]=51 (wrapped)", deq(hl[2+LDQ*3], 51.0));
  REPORT("gcol_wrap: l_col[3,3]=g_col[1,4]=52 (wrapped)", deq(hl[3+LDQ*3], 52.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
// Test 9: noff + nblk = multi-block with partial blocks at both ends
//
// Parameters:  g_col_dim1=8, ldq=8, nlen=6, nblk=4, np_rows=1, my_prow=0,
//              noff_in=0, noff=2, matrixCols=8.
// Analysis:    nbs = 2/4 = 0;  nbe = (2+6-1)/4 = 7/4 = 1.
//              jb=0: g_off=0, l_off=0, js=max(3,1)=3, je=min(8,4)=4 → index 3,4.
//                g_local = 0-2+index = index-2 ∈ {1,2} → g_col rows 0,1 (0-indexed).
//                l_local = index ∈ {3,4}             → l_col rows 2,3 (0-indexed).
//              jb=1: g_off=4, l_off=4, js=max(-1,1)=1, je=min(4,4)=4 → index 1..4.
//                g_local = 4-2+index = 2+index ∈ {3,4,5,6} → g_col rows 2..5 (0-indexed).
//                l_local = 4+index ∈ {5,6,7,8}              → l_col rows 4..7 (0-indexed).
//              Total rows distributed: 2 (from jb=0) + 4 (from jb=1) = 6 = nlen ✓.
// Verifies:    partial-first-block + full-second-block correctly handled.
// ============================================================
static void test_noff_partial_blocks()
{
  printf("\ngpu_distribute_global_column — noff=2, partial blocks at both ends:\n");

  const int G1=8, G2=6, LDQ=8, MC=8, NLEN=6, NBLK=4;
  const int noff_in=0, noff=2, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  gpu_distribute_global_column<double>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("partial_blk: 0 diffs vs CPU reference",  count_diffs(hl, hl_ref, LDQ*MC) == 0);
  // jb=0: g_local = index-2 ∈ {1,2} → g_col rows 0,1; l_col rows 2,3
  REPORT("partial_blk: l_col[2,0]=g_col[0,0]=11 (jb=0,index=3)", deq(hl[2+LDQ*0], 11.0));
  REPORT("partial_blk: l_col[3,0]=g_col[1,0]=12 (jb=0,index=4)", deq(hl[3+LDQ*0], 12.0));
  // jb=1: g_local = g_off-noff+index = 4-2+index = 2+index ∈ {3,4,5,6}
  //        → g_col rows 3..6 (1-indexed) = rows 2..5 (0-indexed); l_col rows 4..7
  REPORT("partial_blk: l_col[4,0]=g_col[2,0]=13 (jb=1,index=1)", deq(hl[4+LDQ*0], 13.0));
  REPORT("partial_blk: l_col[7,0]=g_col[5,0]=16 (jb=1,index=4)", deq(hl[7+LDQ*0], 16.0));
  // Rows 0,1 of l_col must remain zero (no jb touches them)
  REPORT("partial_blk: l_col[0,0]=0 (untouched)",  deq(hl[0+LDQ*0],  0.0));
  REPORT("partial_blk: l_col[1,0]=0 (untouched)",  deq(hl[1+LDQ*0],  0.0));

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}

// ============================================================
#ifdef WANT_HALF_PRECISION_REAL
// ============================================================
// Test: gpu_distribute_global_column<__half> — basic copy
//
// Same parameters as test_basic_copy but with __half values.
// Values 11..44 are exactly representable in __half (integers <= 2048).
// Comparison uses __half2float() since host code has no __half→double cast.
// ============================================================
static void test_half()
{
  printf("\ngpu_distribute_global_column<__half> — basic copy:\n");

  const int G1=4, G2=4, LDQ=4, MC=4, NLEN=4, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  __half hg[G1*G2], hl_ref[LDQ*MC], hl[LDQ*MC];
  for (int k=0; k<LDQ*MC; k++) { hl_ref[k] = __float2half(0.0f); hl[k] = __float2half(0.0f); }
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = __float2half((float)((c+1)*10 + (r+1)));

  cpu_distribute_ref<__half>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  __half *dg  = dev_alloc_upload(hg,    G1*G2);
  __half *dlc = dev_alloc_zero<__half>(LDQ*MC);

  gpu_distribute_global_column<__half>(dg, dlc, G1, G2, LDQ, MC,
                                       noff_in, noff, NLEN, my_prow, np_rows, NBLK,
                                       0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hl, dlc, LDQ*MC);

  int ndiffs = 0;
  for (int k=0; k<LDQ*MC; k++)
    if (__half2float(hl[k]) != __half2float(hl_ref[k])) ndiffs++;

  REPORT("__half: 0 diffs vs CPU reference",    ndiffs == 0);
  REPORT("__half: l_col[0,0]=11.0f (half)",     __half2float(hl[0+LDQ*0]) == 11.0f);
  REPORT("__half: l_col[3,3]=44.0f (half)",     __half2float(hl[3+LDQ*3]) == 44.0f);

  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(dlc));
}
#endif /* WANT_HALF_PRECISION_REAL */

// ============================================================
int main(void)
{
  printf("=== Unit tests for gpu_distribute_global_column.h (CUDA) ===\n");

  test_basic_copy();
  test_noff_in_offset();
  test_noff_row_skip();
  test_two_blocks();
  test_np_rows2_prow0();
  test_np_rows2_prow1();
  test_float();
  test_gcol_column_wrap();
  test_noff_partial_blocks();

#ifdef WANT_HALF_PRECISION_REAL
  test_half();
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
