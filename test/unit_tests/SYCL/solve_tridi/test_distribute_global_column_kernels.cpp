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

// Unit tests for gpu_distribute_global_column (Intel SYCL backend).
// SYCL port of test/unit_tests/CUDA/test_distribute_global_column_kernels.cu.
//
// The SYCL kernel is a 3-D nd_range kernel with dimension mapping:
//   dim 2 → i    (outer loop 1..nlen, same as CUDA x-dim)
//   dim 1 → ind  (jb block offset, same as CUDA y-dim)
//   dim 0 → ind2 (index within block, same as CUDA z-dim)
// Requirement: matrixCols >= nblk (ensures dim-0 covers all index values).
//
// The kernel body is inlined here because sycl_distribute_global_column.cpp
// includes syclCommon.hpp (getQueueOrDefault, QueueData, etc.) which is not
// available in self-contained unit tests.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_SYCL_GPU_VERSION

#include <stdint.h>
#include <cstring>
#include <sycl/sycl.hpp>
#include <complex>

// ============================================================
// Kernel body (copied from sycl_distribute_global_column.cpp)
// ============================================================

template <typename T>
void gpu_distribute_global_column_kernel(T *g_col, T *l_col,
                                         const int g_col_dim1, const int g_col_dim2,
                                         const int ldq, const int matrixCols,
                                         const int noff_in, const int noff, const int nlen,
                                         const int my_prow, const int np_rows, const int nblk,
                                         const sycl::nd_item<3> &it)
{
  int i    = it.get_group(2) * it.get_local_range(2) + it.get_local_id(2) + 1;
  int ind  = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);
  int ind2 = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

  int nbs = noff / (nblk * np_rows);
  int nbe = (noff + nlen - 1) / (nblk * np_rows);

  int number_of_entries;
  int entries_in_started_col;
  int entries_in_sub_matrix;
  int columns_in_sub_matrix;
  int index_sub_matrix;
  int col_sub_matrix;
  int row_sub_matrix;
  int l_col_global_row2, l_col_global_col2;
  int g_col_global_row2, g_col_global_col2;

  if (i >= 1 && i < nlen + 1) {
    int g_col_offset1 = 1;
    int g_col_offset2 = i;
    int l_col_offset1 = 1;
    int l_col_offset2 = noff_in + i;

    int jb = ind + nbs;

    if (jb >= nbs && jb <= nbe) {
      int g_off2 = jb * nblk * np_rows + nblk * my_prow;
      int l_off2 = jb * nblk;

      int js2 = sycl::max(noff + 1 - g_off2, 1);
      int je2 = sycl::min(noff + nlen - g_off2, nblk);

      if (je2 >= js2) {
        int index = ind2 + js2;

        if (index >= js2 && index <= je2) {
          number_of_entries     = (g_col_dim2 - g_col_offset2) * g_col_dim1 + g_col_dim1 - g_col_offset1 + 1;
          entries_in_started_col = g_col_dim1 - g_col_offset1 + 1;
          entries_in_sub_matrix  = number_of_entries - entries_in_started_col;
          columns_in_sub_matrix  = entries_in_sub_matrix / g_col_dim1;

          if (g_off2 - noff + index > entries_in_started_col) {
            index_sub_matrix = g_off2 - noff + index - entries_in_started_col;
            col_sub_matrix   = (index_sub_matrix - 1) / g_col_dim1 + 1;
            row_sub_matrix   = index_sub_matrix % g_col_dim1;
            if (row_sub_matrix == 0) row_sub_matrix = g_col_dim1;
            g_col_global_col2 = col_sub_matrix + g_col_offset2;
            g_col_global_row2 = row_sub_matrix;
          } else {
            g_col_global_col2 = g_col_offset2;
            g_col_global_row2 = g_col_offset1 + g_off2 - noff + index - 1;
          }

          number_of_entries     = (matrixCols - l_col_offset2) * ldq + ldq - l_col_offset1 + 1;
          entries_in_started_col = ldq - l_col_offset1 + 1;
          entries_in_sub_matrix  = number_of_entries - entries_in_started_col;
          columns_in_sub_matrix  = entries_in_sub_matrix / ldq;

          if (l_off2 + index > entries_in_started_col) {
            index_sub_matrix = l_off2 + index - entries_in_started_col;
            col_sub_matrix   = (index_sub_matrix - 1) / ldq + 1;
            row_sub_matrix   = index_sub_matrix % ldq;
            if (row_sub_matrix == 0) row_sub_matrix = ldq;
            l_col_global_col2 = col_sub_matrix + l_col_offset2;
            l_col_global_row2 = row_sub_matrix;
          } else {
            l_col_global_col2 = l_col_offset2;
            l_col_global_row2 = l_col_offset1 + l_off2 + index - 1;
          }

          l_col[(l_col_global_row2 - 1) + ldq * (l_col_global_col2 - 1)] =
              g_col[(g_col_global_row2 - 1) + g_col_dim1 * (g_col_global_col2 - 1)];
        }
      }
    }
  }
}

// ============================================================
// SYCL queue (default selector, created once)
// ============================================================

static sycl::queue &get_queue() {
  static sycl::queue q(sycl::default_selector_v);
  return q;
}

// ============================================================
// Self-contained launcher (mirrors sycl_distribute_global_column.cpp)
// ============================================================

template <typename T>
static void launch_distribute_global_column(
    T *g_col_dev, T *l_col_dev,
    int g_col_dim1, int g_col_dim2,
    int ldq,        int matrixCols,
    int noff_in, int noff, int nlen,
    int my_prow, int np_rows, int nblk)
{
  sycl::queue &q = get_queue();
  int nbs = noff / (nblk * np_rows);
  int nbe = (noff + nlen - 1) / (nblk * np_rows);

  // Dimensions match sycl_distribute_global_column.cpp: (dim0=2, dim1=16, dim2=16)
  sycl::range<3> tpb(2, 16, 16);
  sycl::range<3> blocks(
      (matrixCols + tpb.get(0) - 1) / tpb.get(0),
      (nbe - nbs + 1 + tpb.get(1) - 1) / tpb.get(1),
      (nlen + tpb.get(2) - 1) / tpb.get(2));

  q.parallel_for(sycl::nd_range<3>(blocks * tpb, tpb),
                 [=](sycl::nd_item<3> it) {
                   gpu_distribute_global_column_kernel(
                       g_col_dev, l_col_dev,
                       g_col_dim1, g_col_dim2, ldq, matrixCols,
                       noff_in, noff, nlen, my_prow, np_rows, nblk, it);
                 }).wait();
}

// ============================================================
// Device memory helpers
// ============================================================

template <typename T>
static T *dev_alloc_upload(const T *h, int n) {
  sycl::queue &q = get_queue();
  T *d = sycl::malloc_device<T>(n, q);
  q.memcpy(d, h, n * sizeof(T)).wait();
  return d;
}

template <typename T>
static T *dev_alloc_zero(int n) {
  sycl::queue &q = get_queue();
  T *d = sycl::malloc_device<T>(n, q);
  q.memset(d, 0, n * sizeof(T)).wait();
  return d;
}

template <typename T>
static void dev_read_n(T *h, T *d, int n) {
  get_queue().memcpy(h, d, n * sizeof(T)).wait();
}

template <typename T>
static void dev_free(T *d) {
  sycl::free(d, get_queue());
}

// ============================================================
// CPU reference: exact translation of Fortran distribute_global_column_4
// ============================================================

static void local_to_global_ref(int dim1, int /*dim2*/, int n, int m,
                                 int local_index, int &row, int &col) {
  int esc = dim1 - n + 1;
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
    int g_off2 = i;
    int l_off2 = noff_in + i;

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

template <typename T>
static int count_diffs(const T *a, const T *b, int n, double tol = 1e-10) {
  int cnt = 0;
  for (int i = 0; i < n; i++)
    if (fabs((double)a[i] - (double)b[i]) > tol) cnt++;
  return cnt;
}

// ============================================================
// Test infrastructure
// ============================================================

static int g_failures = 0;

#define REPORT(label, ok)                                                      \
  do {                                                                         \
    printf("  %-72s [%s]\n", (label), (ok) ? "PASS" : "FAIL");                \
    if (!(ok)) g_failures++;                                                   \
  } while (0)

static bool deq(double a, double b, double tol = 1e-10) { return fabs(a - b) < tol; }
static bool feq(float  a, float  b, float  tol = 1e-5f) { return fabsf(a - b) < tol; }

// ============================================================
// Test 1: basic copy — noff=0, single process, nblk≥nlen
// ============================================================
static void test_basic_copy()
{
  printf("\ngpu_distribute_global_column — basic copy (noff=0, np_rows=1):\n");

  const int G1=4, G2=4, LDQ=4, MC=4, NLEN=4, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("basic_copy: 0 diffs vs CPU reference", count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("basic_copy: l_col[0,0]=g_col[0,0]=11", deq(hl[0+LDQ*0], 11.0));
  REPORT("basic_copy: l_col[3,3]=g_col[3,3]=44", deq(hl[3+LDQ*3], 44.0));
  REPORT("basic_copy: l_col[1,2]=g_col[1,2]=32", deq(hl[1+LDQ*2], 32.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 2: noff_in offset — destination columns shifted
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

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("noff_in: 0 diffs vs CPU reference",     count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("noff_in: l_col[0,2]=g_col[0,0]=11",     deq(hl[0+LDQ*2], 11.0));
  REPORT("noff_in: l_col[3,5]=g_col[3,3]=44",     deq(hl[3+LDQ*5], 44.0));
  REPORT("noff_in: l_col[0,0]=0 (before offset)", deq(hl[0+LDQ*0],  0.0));
  REPORT("noff_in: l_col[0,1]=0 (before offset)", deq(hl[0+LDQ*1],  0.0));
  REPORT("noff_in: l_col[0,6]=0 (after range)",   deq(hl[0+LDQ*6],  0.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 3: noff row skip — rows land at bottom of first block
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

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("noff_skip: 0 diffs vs CPU reference",  count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("noff_skip: l_col[2,0]=g_col[0,0]=11",  deq(hl[2+LDQ*0], 11.0));
  REPORT("noff_skip: l_col[3,0]=g_col[1,0]=12",  deq(hl[3+LDQ*0], 12.0));
  REPORT("noff_skip: l_col[2,1]=g_col[0,1]=21",  deq(hl[2+LDQ*1], 21.0));
  REPORT("noff_skip: l_col[3,1]=g_col[1,1]=22",  deq(hl[3+LDQ*1], 22.0));
  REPORT("noff_skip: l_col[0,0]=0 (untouched)",  deq(hl[0+LDQ*0],  0.0));
  REPORT("noff_skip: l_col[1,0]=0 (untouched)",  deq(hl[1+LDQ*0],  0.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 4: two-block span — nblk=4, nlen=8
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

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("two_blocks: 0 diffs vs CPU reference",       count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("two_blocks: l_col[0,0]=11 (jb=0 first row)", deq(hl[0+LDQ*0], 11.0));
  REPORT("two_blocks: l_col[3,0]=14 (jb=0 last row)",  deq(hl[3+LDQ*0], 14.0));
  REPORT("two_blocks: l_col[4,0]=15 (jb=1 first row)", deq(hl[4+LDQ*0], 15.0));
  REPORT("two_blocks: l_col[7,7]=88 (jb=1 last row)",  deq(hl[7+LDQ*7], 88.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 5: two MPI process rows, prow=0
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

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("prow0: 0 diffs vs CPU reference",         count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("prow0: l_col[0,0]=g_col[0,0]=11",         deq(hl[0+LDQ*0], 11.0));
  REPORT("prow0: l_col[3,7]=g_col[3,7]=84",         deq(hl[3+LDQ*7], 84.0));
  REPORT("prow0: l_col[4,0]=0 (prow=1 territory)",  deq(hl[4+LDQ*0],  0.0));
  REPORT("prow0: l_col[7,7]=0 (prow=1 territory)",  deq(hl[7+LDQ*7],  0.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 6: two MPI process rows, prow=1
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

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("prow1: 0 diffs vs CPU reference",  count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("prow1: l_col[0,0]=g_col[4,0]=15", deq(hl[0+LDQ*0], 15.0));
  REPORT("prow1: l_col[3,0]=g_col[7,0]=18", deq(hl[3+LDQ*0], 18.0));
  REPORT("prow1: l_col[0,7]=g_col[4,7]=85", deq(hl[0+LDQ*7], 85.0));
  REPORT("prow1: l_col[3,7]=g_col[7,7]=88", deq(hl[3+LDQ*7], 88.0));
  REPORT("prow1: l_col[4,0]=0 (untouched)", deq(hl[4+LDQ*0],  0.0));
  REPORT("prow1: l_col[7,7]=0 (untouched)", deq(hl[7+LDQ*7],  0.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 7: float precision
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

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  int ndiffs = 0;
  for (int k=0; k<LDQ*MC; k++) if (!feq(hl[k], hl_ref[k])) ndiffs++;

  REPORT("float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("float: l_col[0,0]=11.0f",         feq(hl[0+LDQ*0], 11.0f));
  REPORT("float: l_col[3,3]=44.0f",         feq(hl[3+LDQ*3], 44.0f));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 8: g_col column wrapping (g_col_dim1=2, nblk=4)
// ============================================================
static void test_gcol_column_wrap()
{
  printf("\ngpu_distribute_global_column — g_col column wrap (g_col_dim1=2, nblk=4):\n");

  const int G1=2, G2=5, LDQ=4, MC=4, NLEN=4, NBLK=4;
  const int noff_in=0, noff=0, my_prow=0, np_rows=1;

  double hg[G1*G2], hl_ref[LDQ*MC]={0}, hl[LDQ*MC]={0};
  for (int c=0; c<G2; c++)
    for (int r=0; r<G1; r++)
      hg[r + G1*c] = (c+1)*10.0 + (r+1);

  cpu_distribute_ref<double>(hg, hl_ref, G1, G2, LDQ, MC,
                             noff_in, noff, NLEN, my_prow, np_rows, NBLK);

  double *dg  = dev_alloc_upload(hg,     G1*G2);
  double *dlc = dev_alloc_zero<double>(LDQ*MC);

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("gcol_wrap: 0 diffs vs CPU reference",                deq(count_diffs(hl, hl_ref, LDQ*MC), 0.0));
  REPORT("gcol_wrap: l_col[0,0]=g_col[0,0]=11",               deq(hl[0+LDQ*0], 11.0));
  REPORT("gcol_wrap: l_col[1,0]=g_col[1,0]=12",               deq(hl[1+LDQ*0], 12.0));
  REPORT("gcol_wrap: l_col[2,0]=g_col[0,1]=21 (wrapped)",     deq(hl[2+LDQ*0], 21.0));
  REPORT("gcol_wrap: l_col[3,0]=g_col[1,1]=22 (wrapped)",     deq(hl[3+LDQ*0], 22.0));
  REPORT("gcol_wrap: l_col[0,3]=g_col[0,3]=41",               deq(hl[0+LDQ*3], 41.0));
  REPORT("gcol_wrap: l_col[2,3]=g_col[0,4]=51 (wrapped)",     deq(hl[2+LDQ*3], 51.0));
  REPORT("gcol_wrap: l_col[3,3]=g_col[1,4]=52 (wrapped)",     deq(hl[3+LDQ*3], 52.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
// Test 9: noff + nblk = multi-block with partial blocks at both ends
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

  launch_distribute_global_column(dg, dlc, G1, G2, LDQ, MC,
                                  noff_in, noff, NLEN, my_prow, np_rows, NBLK);
  dev_read_n(hl, dlc, LDQ*MC);

  REPORT("partial_blk: 0 diffs vs CPU reference",                count_diffs(hl, hl_ref, LDQ*MC) == 0);
  REPORT("partial_blk: l_col[2,0]=g_col[0,0]=11 (jb=0,index=3)", deq(hl[2+LDQ*0], 11.0));
  REPORT("partial_blk: l_col[3,0]=g_col[1,0]=12 (jb=0,index=4)", deq(hl[3+LDQ*0], 12.0));
  REPORT("partial_blk: l_col[4,0]=g_col[2,0]=13 (jb=1,index=1)", deq(hl[4+LDQ*0], 13.0));
  REPORT("partial_blk: l_col[7,0]=g_col[5,0]=16 (jb=1,index=4)", deq(hl[7+LDQ*0], 16.0));
  REPORT("partial_blk: l_col[0,0]=0 (untouched)",                 deq(hl[0+LDQ*0],  0.0));
  REPORT("partial_blk: l_col[1,0]=0 (untouched)",                 deq(hl[1+LDQ*0],  0.0));

  dev_free(dg); dev_free(dlc);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for gpu_distribute_global_column (Intel SYCL) ===\n");

  test_basic_copy();
  test_noff_in_offset();
  test_noff_row_skip();
  test_two_blocks();
  test_np_rows2_prow0();
  test_np_rows2_prow1();
  test_float();
  test_gcol_column_wrap();
  test_noff_partial_blocks();

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
