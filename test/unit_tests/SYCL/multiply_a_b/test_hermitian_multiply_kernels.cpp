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

// ============================================================
// Kernel bodies (copied from elpa_multiply_a_b_sycl.cpp)
// ============================================================

template <typename T>
static void gpu_copy_tmp2_c_kernel(T *tmp2_dev, T *c_dev,
                                    const int nr_done, const int nstor,
                                    const int lcs, const int lce,
                                    const int ldc, const int /*ldcCols*/,
                                    const sycl::nd_item<1> &it)
{
  int idex = (int)it.get_local_id(0) + 1;
  int jdex = (int)it.get_group(0)    + 1;
  c_dev[nr_done + (idex - 1) + ldc * (lcs - 1 + jdex - 1)] =
      tmp2_dev[(idex - 1) + nstor * (jdex - 1)];
}

template <typename T>
static void gpu_copy_a_aux_bc_loop_kernel(T *a_dev, T *aux_bc_dev,
                                           int *lrs_save_dev, int *lre_save_dev,
                                           int *n_aux_bc_save_dev,
                                           const int noff, const int nblk, const int lda,
                                           const sycl::nd_item<1> &it)
{
  int n        = (int)it.get_group(0);
  int lrs      = lrs_save_dev[n];
  int lre      = lre_save_dev[n];
  int n_aux_bc = n_aux_bc_save_dev[n];
  for (int i = (int)it.get_local_id(0); i < (lre - lrs + 1); i += (int)it.get_local_range(0))
    aux_bc_dev[n_aux_bc + i] = a_dev[(lrs - 1) + i + lda * (noff * nblk + n)];
}

template <typename T>
static void gpu_copy_aux_bc_aux_mat_loop_kernel(const T *aux_bc_dev, T *aux_mat_dev,
                                                 int *lrs_save_dev, int *lre_save_dev,
                                                 int *n_aux_bc_save_dev,
                                                 const int nstor0, const int l_rows,
                                                 const sycl::nd_item<1> &it)
{
  int n        = (int)it.get_group(0);
  int nstor    = nstor0 + n;
  int lrs      = lrs_save_dev[n];
  int lre      = lre_save_dev[n];
  int n_aux_bc = n_aux_bc_save_dev[n];
  for (int i = (int)it.get_local_id(0); i < (lre - lrs + 1); i += (int)it.get_local_range(0))
    aux_mat_dev[(lrs - 1) + i + l_rows * (nstor - 1)] = aux_bc_dev[n_aux_bc + i];
}

// ============================================================
// Self-contained queue singleton and launchers
// ============================================================

static sycl::queue &get_queue() {
  static sycl::queue q(sycl::default_selector_v);
  return q;
}

template <typename T>
static void launch_copy_tmp2_c(T *tmp2_dev, T *c_dev,
                                int nr_done, int nstor, int lcs, int lce,
                                int ldc, int ldcCols)
{
  sycl::queue &q = get_queue();
  sycl::range<1> tpb(nstor);
  sycl::range<1> blocks(lce - lcs + 1);
  q.parallel_for(sycl::nd_range<1>(blocks * tpb, tpb), [=](sycl::nd_item<1> it) {
    gpu_copy_tmp2_c_kernel(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, it);
  }).wait();
}

// Work-group size for the two loop kernels (matches MIN_THREADS_PER_BLOCK for SYCL).
static constexpr int TPB = 32;

template <typename T>
static void launch_copy_a_aux_bc_loop(T *a_dev, T *aux_bc_dev,
                                       int *lrs_save_dev, int *lre_save_dev,
                                       int *n_aux_bc_save_dev,
                                       int noff, int nblk, int lda, int n_size)
{
  sycl::queue &q = get_queue();
  sycl::range<1> tpb(TPB);
  sycl::range<1> blocks(n_size);
  q.parallel_for(sycl::nd_range<1>(blocks * tpb, tpb), [=](sycl::nd_item<1> it) {
    gpu_copy_a_aux_bc_loop_kernel(a_dev, aux_bc_dev, lrs_save_dev, lre_save_dev,
                                  n_aux_bc_save_dev, noff, nblk, lda, it);
  }).wait();
}

template <typename T>
static void launch_copy_aux_bc_aux_mat_loop(T *aux_bc_dev, T *aux_mat_dev,
                                             int *lrs_save_dev, int *lre_save_dev,
                                             int *n_aux_bc_save_dev,
                                             int nstor0, int l_rows, int n_size)
{
  if (n_size <= 0) return;
  sycl::queue &q = get_queue();
  sycl::range<1> tpb(TPB);
  sycl::range<1> blocks(n_size);
  q.parallel_for(sycl::nd_range<1>(blocks * tpb, tpb), [=](sycl::nd_item<1> it) {
    gpu_copy_aux_bc_aux_mat_loop_kernel(aux_bc_dev, aux_mat_dev, lrs_save_dev, lre_save_dev,
                                        n_aux_bc_save_dev, nstor0, l_rows, it);
  }).wait();
}

// ============================================================
// Infrastructure
// ============================================================

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
static void dev_free(T *d) { sycl::free(d, get_queue()); }

// ============================================================
// CPU references
// ============================================================

template <typename T>
static void cpu_copy_tmp2_c_ref(const T *tmp2, T *c,
                                 int nr_done, int nstor,
                                 int lcs, int lce, int ldc, int /*ldcCols*/)
{
  for (int j = 0; j < lce - lcs + 1; j++)
    for (int i = 0; i < nstor; i++)
      c[nr_done + i + ldc * (lcs - 1 + j)] = tmp2[i + nstor * j];
}

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
// Tests for gpu_copy_tmp2_c
// ============================================================

static void test_copy_tmp2_c_basic()
{
  printf("\ngpu_copy_tmp2_c — basic copy (nr_done=0, lcs=1):\n");

  const int NSTOR=3, LCS=1, LCE=3, LDC=3, LDCCOLS=3, NR_DONE=0;
  const int NCOLS = LCE - LCS + 1;

  double htmp2[NSTOR * NCOLS], hc_ref[LDC * LDCCOLS] = {0}, hc[LDC * LDCCOLS] = {0};
  for (int j = 0; j < NCOLS; j++)
    for (int i = 0; i < NSTOR; i++)
      htmp2[i + NSTOR * j] = (j + 1) * 10.0 + (i + 1);

  cpu_copy_tmp2_c_ref<double>(htmp2, hc_ref, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);

  double *dtmp2 = dev_alloc_upload(htmp2,  NSTOR * NCOLS);
  double *dc    = dev_alloc_zero<double>(LDC * LDCCOLS);

  launch_copy_tmp2_c(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_basic: 0 diffs vs CPU reference", count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_basic: c[0,0]=tmp2[0,0]=11",     deq(hc[0 + LDC * 0], 11.0));
  REPORT("copy_tmp2_c_basic: c[2,2]=tmp2[2,2]=33",     deq(hc[2 + LDC * 2], 33.0));
  REPORT("copy_tmp2_c_basic: c[1,1]=tmp2[1,1]=22",     deq(hc[1 + LDC * 1], 22.0));

  dev_free(dtmp2); dev_free(dc);
}

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

  launch_copy_tmp2_c(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_row_off: 0 diffs vs CPU reference",      count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_row_off: c[2,0]=tmp2[0,0]=11",           deq(hc[2 + LDC * 0], 11.0));
  REPORT("copy_tmp2_c_row_off: c[3,2]=tmp2[1,2]=32",           deq(hc[3 + LDC * 2], 32.0));
  REPORT("copy_tmp2_c_row_off: c[0,0]=0 (before nr_done)",     deq(hc[0 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_row_off: c[1,0]=0 (before nr_done)",     deq(hc[1 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_row_off: c[4,0]=0 (past nr_done+nstor)", deq(hc[4 + LDC * 0],  0.0));

  dev_free(dtmp2); dev_free(dc);
}

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

  launch_copy_tmp2_c(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_col_off: 0 diffs vs CPU reference",    count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_col_off: c[0,2]=tmp2[0,0]=11",        deq(hc[0 + LDC * 2], 11.0));
  REPORT("copy_tmp2_c_col_off: c[2,4]=tmp2[2,2]=33",        deq(hc[2 + LDC * 4], 33.0));
  REPORT("copy_tmp2_c_col_off: c[0,0]=0 (before lcs)",      deq(hc[0 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_col_off: c[0,1]=0 (before lcs)",      deq(hc[0 + LDC * 1],  0.0));
  REPORT("copy_tmp2_c_col_off: c[0,5]=0 (after lce)",       deq(hc[0 + LDC * 5],  0.0));

  dev_free(dtmp2); dev_free(dc);
}

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

  launch_copy_tmp2_c(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);
  dev_read_n(hc, dc, LDC * LDCCOLS);

  REPORT("copy_tmp2_c_combined: 0 diffs vs CPU reference",  count_diffs(hc, hc_ref, LDC * LDCCOLS) == 0);
  REPORT("copy_tmp2_c_combined: c[1,1]=tmp2[0,0]=11",       deq(hc[1 + LDC * 1], 11.0));
  REPORT("copy_tmp2_c_combined: c[2,3]=tmp2[1,2]=32",       deq(hc[2 + LDC * 3], 32.0));
  REPORT("copy_tmp2_c_combined: c[0,0]=0 (untouched)",      deq(hc[0 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_combined: c[3,1]=0 (past nr_done+nstor)", deq(hc[3 + LDC * 1], 0.0));
  REPORT("copy_tmp2_c_combined: c[1,0]=0 (before lcs)",     deq(hc[1 + LDC * 0],  0.0));
  REPORT("copy_tmp2_c_combined: c[1,4]=0 (after lce)",      deq(hc[1 + LDC * 4],  0.0));

  dev_free(dtmp2); dev_free(dc);
}

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

  launch_copy_tmp2_c(dtmp2, dc, NR_DONE, NSTOR, LCS, LCE, LDC, LDCCOLS);
  dev_read_n(hc, dc, LDC * LDCCOLS);

  int ndiffs = 0;
  for (int k = 0; k < LDC * LDCCOLS; k++) if (!feq(hc[k], hc_ref[k])) ndiffs++;

  REPORT("copy_tmp2_c_float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("copy_tmp2_c_float: c[0,0]=11.0f",             feq(hc[0 + LDC * 0], 11.0f));
  REPORT("copy_tmp2_c_float: c[2,2]=33.0f",             feq(hc[2 + LDC * 2], 33.0f));

  dev_free(dtmp2); dev_free(dc);
}

// ============================================================
// Tests for gpu_copy_a_aux_bc_loop
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

  launch_copy_a_aux_bc_loop(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE);
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_single_full: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_single_full: aux_bc[0]=a[0,0]=11",     deq(haux_bc[0], 11.0));
  REPORT("a_aux_bc_single_full: aux_bc[3]=a[3,0]=14",     deq(haux_bc[3], 14.0));

  dev_free(da); dev_free(dabc);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_a_aux_bc_loop(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE);
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_partial: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_partial: aux_bc[0]=a[1,0]=12",     deq(haux_bc[0], 12.0));
  REPORT("a_aux_bc_partial: aux_bc[1]=a[2,0]=13",     deq(haux_bc[1], 13.0));

  dev_free(da); dev_free(dabc);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_a_aux_bc_loop(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE);
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_multi: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_multi: aux_bc[0]=a[0,0]=11",     deq(haux_bc[0], 11.0));
  REPORT("a_aux_bc_multi: aux_bc[2]=a[2,0]=13",     deq(haux_bc[2], 13.0));
  REPORT("a_aux_bc_multi: aux_bc[3]=a[1,1]=22",     deq(haux_bc[3], 22.0));
  REPORT("a_aux_bc_multi: aux_bc[5]=a[3,1]=24",     deq(haux_bc[5], 24.0));

  dev_free(da); dev_free(dabc);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_a_aux_bc_loop(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE);
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  REPORT("a_aux_bc_noff: 0 diffs vs CPU reference", count_diffs(haux_bc, haux_bc_ref, AUX_BC_SIZE) == 0);
  REPORT("a_aux_bc_noff: aux_bc[0]=a[0,4]=51",     deq(haux_bc[0], 51.0));
  REPORT("a_aux_bc_noff: aux_bc[3]=a[3,4]=54",     deq(haux_bc[3], 54.0));

  dev_free(da); dev_free(dabc);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_a_aux_bc_loop(da, dabc, dlrs, dlre, dnabc, noff, NBLK, LDA, N_SIZE);
  dev_read_n(haux_bc, dabc, AUX_BC_SIZE);

  int ndiffs = 0;
  for (int k = 0; k < AUX_BC_SIZE; k++) if (!feq(haux_bc[k], haux_bc_ref[k])) ndiffs++;

  REPORT("a_aux_bc_float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("a_aux_bc_float: aux_bc[0]=11.0f",          feq(haux_bc[0], 11.0f));
  REPORT("a_aux_bc_float: aux_bc[3]=14.0f",          feq(haux_bc[3], 14.0f));

  dev_free(da); dev_free(dabc);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

// ============================================================
// Tests for gpu_copy_aux_bc_aux_mat_loop
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

  launch_copy_aux_bc_aux_mat_loop(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE);
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_basic: 0 diffs vs CPU reference", count_diffs(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  REPORT("aux_bc_aux_mat_basic: aux_mat[0,0]=aux_bc[0]=1", deq(haux_mat[0 + L_ROWS * 0], 1.0));
  REPORT("aux_bc_aux_mat_basic: aux_mat[3,0]=aux_bc[3]=4", deq(haux_mat[3 + L_ROWS * 0], 4.0));
  REPORT("aux_bc_aux_mat_basic: aux_mat[0,1]=0",           deq(haux_mat[0 + L_ROWS * 1], 0.0));

  dev_free(dabc); dev_free(damat);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_aux_bc_aux_mat_loop(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE);
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_multi: 0 diffs vs CPU reference",    count_diffs(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  REPORT("aux_bc_aux_mat_multi: aux_mat[0,0]=aux_bc[0]=10",  deq(haux_mat[0 + L_ROWS * 0], 10.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[1,0]=aux_bc[1]=20",  deq(haux_mat[1 + L_ROWS * 0], 20.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[2,0]=0 (not written)", deq(haux_mat[2 + L_ROWS * 0], 0.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[2,1]=aux_bc[2]=30",  deq(haux_mat[2 + L_ROWS * 1], 30.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[3,1]=aux_bc[3]=40",  deq(haux_mat[3 + L_ROWS * 1], 40.0));
  REPORT("aux_bc_aux_mat_multi: aux_mat[0,1]=0 (not written)", deq(haux_mat[0 + L_ROWS * 1], 0.0));

  dev_free(dabc); dev_free(damat);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_aux_bc_aux_mat_loop(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE);
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  REPORT("aux_bc_aux_mat_nstor: 0 diffs vs CPU reference",     count_diffs(haux_mat, haux_mat_ref, L_ROWS * NCOLS_AUX) == 0);
  REPORT("aux_bc_aux_mat_nstor: aux_mat[1,2]=aux_bc[0]=100",   deq(haux_mat[1 + L_ROWS * 2], 100.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[4,2]=aux_bc[3]=400",   deq(haux_mat[4 + L_ROWS * 2], 400.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[1,0]=0 (wrong col)",   deq(haux_mat[1 + L_ROWS * 0], 0.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[1,1]=0 (wrong col)",   deq(haux_mat[1 + L_ROWS * 1], 0.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[0,2]=0 (before lrs)",  deq(haux_mat[0 + L_ROWS * 2], 0.0));
  REPORT("aux_bc_aux_mat_nstor: aux_mat[5,2]=0 (after lre)",   deq(haux_mat[5 + L_ROWS * 2], 0.0));

  dev_free(dabc); dev_free(damat);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

static void test_copy_aux_bc_aux_mat_empty()
{
  printf("\ngpu_copy_aux_bc_aux_mat_loop — n_size=0 early return:\n");

  const int L_ROWS=4, N_SIZE=0, NCOLS_AUX=2;
  const int nstor0=1;
  const int AUX_BC_SIZE = 4;

  double haux_bc[AUX_BC_SIZE] = {1.0, 2.0, 3.0, 4.0};
  double haux_mat[L_ROWS * NCOLS_AUX] = {0};

  int dummy_hlrs[1] = {1}, dummy_hlre[1] = {1}, dummy_hn[1] = {0};

  double *dabc  = dev_alloc_upload(haux_bc,   AUX_BC_SIZE);
  double *damat = dev_alloc_zero<double>(L_ROWS * NCOLS_AUX);
  int    *dlrs  = dev_alloc_upload(dummy_hlrs, 1);
  int    *dlre  = dev_alloc_upload(dummy_hlre, 1);
  int    *dnabc = dev_alloc_upload(dummy_hn,   1);

  launch_copy_aux_bc_aux_mat_loop(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE);
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  bool all_zero = true;
  for (int k = 0; k < L_ROWS * NCOLS_AUX; k++) if (!deq(haux_mat[k], 0.0)) { all_zero = false; break; }

  REPORT("aux_bc_aux_mat_empty: aux_mat all zero when n_size=0", all_zero);

  dev_free(dabc); dev_free(damat);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

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

  launch_copy_aux_bc_aux_mat_loop(dabc, damat, dlrs, dlre, dnabc, nstor0, L_ROWS, N_SIZE);
  dev_read_n(haux_mat, damat, L_ROWS * NCOLS_AUX);

  int ndiffs = 0;
  for (int k = 0; k < L_ROWS * NCOLS_AUX; k++) if (!feq(haux_mat[k], haux_mat_ref[k])) ndiffs++;

  REPORT("aux_bc_aux_mat_float: 0 diffs vs CPU reference", ndiffs == 0);
  REPORT("aux_bc_aux_mat_float: aux_mat[0,0]=1.0f",        feq(haux_mat[0 + L_ROWS * 0], 1.0f));
  REPORT("aux_bc_aux_mat_float: aux_mat[3,0]=4.0f",        feq(haux_mat[3 + L_ROWS * 0], 4.0f));

  dev_free(dabc); dev_free(damat);
  dev_free(dlrs); dev_free(dlre); dev_free(dnabc);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for elpa_hermitian_multiply_gpu.h (Intel SYCL) ===\n");

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
