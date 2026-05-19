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
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_SYCL_GPU_VERSION

#include <stdint.h>
#include <complex>
#include <type_traits>
#include <sycl/sycl.hpp>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-70s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool neq (double a, double b, double tol=1e-10) { return fabs (a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }

template <typename T>
static void dev_read_n(sycl::queue &q, T *h, T *d, int n) {
  q.memcpy(h, d, n*sizeof(T)).wait();
}

// ============================================================
// Kernel bodies (self-contained, no getQueueOrDefault)
// ============================================================

template <typename T>
static void gpu_copy_aux_full_kernel(T *lhs_dev, T *rhs_dev, int l_rows, int l_cols,
                                      int lld_lhs, int lld_rhs, const sycl::nd_item<1> &it) {
  int i_loc = it.get_local_id(0);
  int j_loc = it.get_group(0);
  for (; j_loc < l_cols; j_loc += it.get_group_range(0))
    for (; i_loc < l_rows; i_loc += it.get_local_range(0))
      lhs_dev[i_loc + j_loc*lld_lhs] = rhs_dev[i_loc + j_loc*lld_rhs];
}

template <typename T>
static void gpu_copy_and_set_zeros_aux_full_kernel(T *mat_dev, T *aux_mat_full_dev,
                                                    int l_rows, int l_cols, int nblk_mult,
                                                    const sycl::nd_item<1> &it) {
  int i_loc = it.get_local_id(0);
  int j_loc = it.get_group(0);
  T Zero = elpaDeviceNumber<T>(0.0);
  for (; j_loc < nblk_mult; j_loc += it.get_group_range(0))
    for (; i_loc < nblk_mult; i_loc += it.get_local_range(0)) {
      if (i_loc < l_rows && j_loc < l_cols)
        aux_mat_full_dev[i_loc + j_loc * nblk_mult] = mat_dev[i_loc + j_loc * l_rows];
      else
        aux_mat_full_dev[i_loc + j_loc * nblk_mult] = Zero;
    }
}

// ============================================================
// gpu_copy_aux_full: lhs(i,j) = rhs(i,j) for i<l_rows, j<l_cols
// ============================================================

template <typename T>
static void test_copy_aux_full(sycl::queue &q, const char *tname,
                                bool is_float = false) {
  printf("\ngpu_copy_aux_full<%s>:\n", tname);

  // l_rows=2, l_cols=3, lld_lhs=2, lld_rhs=2
  // rhs = {1,2, 3,4, 5,6} (column-major: col0={1,2}, col1={3,4}, col2={5,6})
  const int l_rows=2, l_cols=3, lld_lhs=2, lld_rhs=2;
  const int n = lld_rhs * l_cols; // 6 elements

  T hrhs[6], hlhs_out[6];
  for (int k=0; k<6; k++) hrhs[k] = elpaHostNumberFromInt<T>(k+1);

  T *drhs = sycl::malloc_device<T>(6, q);
  T *dlhs = sycl::malloc_device<T>(6, q);
  q.memcpy(drhs, hrhs, 6*sizeof(T)).wait();
  q.memset(dlhs, 0, 6*sizeof(T)).wait();

  sycl::range<1> tpb(32);
  sycl::range<1> blocks(l_cols); // one block per column
  q.submit([&](sycl::handler &cgh){
    cgh.parallel_for(sycl::nd_range<1>(blocks*tpb, tpb), [=](sycl::nd_item<1> it){
      gpu_copy_aux_full_kernel(dlhs, drhs, l_rows, l_cols, lld_lhs, lld_rhs, it);
    });
  }).wait();

  dev_read_n(q, hlhs_out, dlhs, 6);

  char lbl[128];
  if (!is_float) {
    snprintf(lbl,sizeof(lbl),"%s copy_aux_full: lhs[0+0*2]==1", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(hlhs_out[0]), 1.0));
    snprintf(lbl,sizeof(lbl),"%s copy_aux_full: lhs[1+0*2]==2", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(hlhs_out[1]), 2.0));
    snprintf(lbl,sizeof(lbl),"%s copy_aux_full: lhs[0+1*2]==3", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(hlhs_out[2]), 3.0));
    snprintf(lbl,sizeof(lbl),"%s copy_aux_full: lhs[0+2*2]==5", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(hlhs_out[4]), 5.0));
  } else {
    snprintf(lbl,sizeof(lbl),"%s copy_aux_full: lhs[0+0*2]==1", tname);
    REPORT(lbl, neqf((float)elpaDeviceRealPart(hlhs_out[0]), 1.0f));
    snprintf(lbl,sizeof(lbl),"%s copy_aux_full: lhs[1+2*2]==6", tname);
    REPORT(lbl, neqf((float)elpaDeviceRealPart(hlhs_out[5]), 6.0f));
  }

  sycl::free(drhs, q); sycl::free(dlhs, q);
}

// ============================================================
// gpu_copy_and_set_zeros_aux_full: pads to nblk_mult x nblk_mult with zeros
// ============================================================

template <typename T>
static void test_copy_and_set_zeros_aux_full(sycl::queue &q, const char *tname,
                                              bool is_float = false) {
  printf("\ngpu_copy_and_set_zeros_aux_full<%s>:\n", tname);

  // l_rows=2, l_cols=2, nblk_mult=4
  // mat[i+j*l_rows] for i<2,j<2: mat = {1,2,3,4} (col-major)
  // aux_full[i+j*nblk_mult]: for i<2,j<2 copy from mat; else 0
  // Expected: aux[0]=1,aux[1]=2,aux[2]=0,aux[3]=0 (col0)
  //           aux[4]=3,aux[5]=4,aux[6]=0,aux[7]=0 (col1)
  //           aux[8..15]=0 (col2,col3)
  const int l_rows=2, l_cols=2, nblk_mult=4;

  T hmat[4];
  T haux_out[16];
  for (int k=0; k<4; k++) hmat[k] = elpaHostNumberFromInt<T>(k+1);

  T *dmat = sycl::malloc_device<T>(4, q);
  T *daux = sycl::malloc_device<T>(16, q);
  q.memcpy(dmat, hmat, 4*sizeof(T)).wait();
  q.memset(daux, 0, 16*sizeof(T)).wait();

  sycl::range<1> tpb(32);
  sycl::range<1> blocks(nblk_mult);
  q.submit([&](sycl::handler &cgh){
    cgh.parallel_for(sycl::nd_range<1>(blocks*tpb, tpb), [=](sycl::nd_item<1> it){
      gpu_copy_and_set_zeros_aux_full_kernel(dmat, daux, l_rows, l_cols, nblk_mult, it);
    });
  }).wait();

  dev_read_n(q, haux_out, daux, 16);

  char lbl[128];
  if (!is_float) {
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[0]==1 (mat[0,0])", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[0]), 1.0));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[1]==2 (mat[1,0])", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[1]), 2.0));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[2]==0 (padded)", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[2]), 0.0));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[4]==3 (mat[0,1])", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[4]), 3.0));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[5]==4 (mat[1,1])", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[5]), 4.0));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[6]==0 (padded)", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[6]), 0.0));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[8]==0 (col2 padded)", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(haux_out[8]), 0.0));
  } else {
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[0]==1", tname);
    REPORT(lbl, neqf((float)elpaDeviceRealPart(haux_out[0]), 1.0f));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[2]==0 (padded)", tname);
    REPORT(lbl, neqf((float)elpaDeviceRealPart(haux_out[2]), 0.0f));
    snprintf(lbl,sizeof(lbl),"%s set_zeros_aux_full: aux[5]==4", tname);
    REPORT(lbl, neqf((float)elpaDeviceRealPart(haux_out[5]), 4.0f));
  }

  sycl::free(dmat, q); sycl::free(daux, q);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for pxgemm multiply kernels (SYCL) ===\n");

  sycl::queue q(sycl::default_selector_v);

  printf("\n--- gpu_copy_aux_full ---\n");
  test_copy_aux_full<double>          (q, "double");
  test_copy_aux_full<float>           (q, "float",            true);
  test_copy_aux_full<gpuDoubleComplex>(q, "double_complex");
  test_copy_aux_full<gpuFloatComplex> (q, "float_complex",    true);

  printf("\n--- gpu_copy_and_set_zeros_aux_full ---\n");
  test_copy_and_set_zeros_aux_full<double>          (q, "double");
  test_copy_and_set_zeros_aux_full<float>           (q, "float",            true);
  test_copy_and_set_zeros_aux_full<gpuDoubleComplex>(q, "double_complex");
  test_copy_and_set_zeros_aux_full<gpuFloatComplex> (q, "float_complex",    true);

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
