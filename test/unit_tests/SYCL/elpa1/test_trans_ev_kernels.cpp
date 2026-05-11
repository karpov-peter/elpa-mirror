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

// Unit tests for trans_ev kernels (SYCL backend).
// Covers: gpu_set_tmat_diag_from_tau, gpu_copy_hvb_a, gpu_copy_hvm_hvb.
// Self-contained: kernel bodies copied here to avoid syclCommon.hpp dependency.
//
// SYCL-specific behavioural difference from CUDA:
//   gpu_copy_hvm_hvb: when tau[ic-1]==0, the SYCL kernel ZERO-FILLS the entire
//   hvm column (up to ld_hvm elements), whereas the CUDA kernel skips it entirely.
//   The tests below verify SYCL's actual behaviour.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_SYCL_GPU_VERSION

#include <stdint.h>
#include <complex>
#include <sycl/sycl.hpp>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-72s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool neq(double a, double b, double tol=1e-10) { return fabs(a-b) < tol; }

template <typename T>
static void dev_read_n(sycl::queue &q, T *h, T *d, int n) {
  q.memcpy(h, d, n*sizeof(T)).wait();
}

// ============================================================
// Kernel bodies (self-contained)
// ============================================================

template <typename T>
static void gpu_set_tmat_diag_from_tau_kernel(T *tmat_dev, T *tau_dev, int max_stored_rows, int nstor, int tau_offset,
                                               const sycl::nd_item<1> &it) {
  int i = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  if (i < nstor) {
    T One  = elpaDeviceNumber<T>(1.0);
    T Zero = elpaDeviceNumber<T>(0.0);
    T tau  = tau_dev[i + tau_offset];
    tmat_dev[i + i * max_stored_rows] = elpaDeviceEqualBool(tau, Zero) ? One : elpaDeviceDivide(One, tau);
  }
}

template <typename T>
static void gpu_copy_hvb_a_kernel(T *hvb_dev, T *a_dev, int ld_hvb, int lda,
                                   int my_prow, int np_rows, int my_pcol, int np_cols, int nblk,
                                   int ics, int ice, const sycl::nd_item<1> &it) {
  int i0    = it.get_local_id();
  int ic_0  = it.get_group(0);
  T constexpr One = static_cast<T>(1.0);

  for (int ic = ic_0 + ics; ic <= ice; ic += it.get_group_range(0)) {
    int l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1);
    int l_rows  = local_index(ic-1, my_prow, np_rows, nblk, -1);
    for (int i = i0; i < l_rows; i += it.get_local_range(0))
      hvb_dev[i + ld_hvb*(ic-ics)] = a_dev[i + (l_colh-1)*lda];
    it.barrier();
    if (my_prow == prow(ic - 1, nblk, np_rows) && i0 == 0)
      hvb_dev[(l_rows-1) + ld_hvb*(ic-ics)] = One;
  }
}

template <typename T>
static void gpu_copy_hvm_hvb_kernel(T *hvm_dev, const T *hvb_dev, const T *tau_dev,
                                     int ld_hvm, int ld_hvb, int my_prow, int np_rows,
                                     int nstor, int nblk, int ics, int ice,
                                     const sycl::nd_item<1> &it) {
  int i0   = it.get_local_id();
  int ic_0 = it.get_group(0);
  T constexpr Zero = static_cast<T>(0.0);

  for (int ic = ic_0 + ics; ic <= ice; ic += it.get_group_range(0)) {
    int l_rows    = local_index(ic-1, my_prow, np_rows, nblk, -1);
    int shift_hvm = ld_hvm*(ic-ics+nstor);
    // SYCL behaviour: when tau[ic-1]==0, zero-fill the entire column
    if (elpaDeviceEqualBool(tau_dev[ic-1], Zero)) {
      for (int i = i0; i < ld_hvm; i += it.get_local_range(0))
        hvm_dev[i + shift_hvm] = Zero;
      continue;
    }
    for (int i = i0; i < l_rows; i += it.get_local_range(0))
      hvm_dev[i + shift_hvm] = hvb_dev[i + ld_hvb*(ic-ics)];
    for (int i = l_rows + i0; i < ld_hvm; i += it.get_local_range(0))
      hvm_dev[i + shift_hvm] = Zero;
  }
}

// ============================================================
// gpu_set_tmat_diag_from_tau
// ============================================================
static void test_set_tmat_diag_from_tau(sycl::queue &q)
{
  printf("\ngpu_set_tmat_diag_from_tau:\n");

  // double: nstor=3, mrs=4, tau=[2.0,0.0,4.0], tau_offset=0
  // Expected diagonal: [0.5, 1.0, 0.25]
  {
    int nstor=3, mrs=4, tau_offset=0;
    double h_tau[]={2.0, 0.0, 4.0};
    double *dtmat = sycl::malloc_device<double>(mrs*mrs, q);
    double *dtau  = sycl::malloc_device<double>(nstor, q);
    q.memset(dtmat, 0, mrs*mrs*sizeof(double)).wait();
    q.memcpy(dtau, h_tau, nstor*sizeof(double)).wait();
    int threads = 32, blocks = (nstor + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    q.parallel_for(sycl::nd_range<1>(blocks*threads, threads), [=](sycl::nd_item<1> it){
      gpu_set_tmat_diag_from_tau_kernel(dtmat, dtau, mrs, nstor, tau_offset, it);
    }).wait();
    double htmat[4*4];
    dev_read_n(q, htmat, dtmat, mrs*mrs);
    REPORT("double diag[0]=1/tau[0]=0.5",  neq(htmat[0+0*mrs], 0.5));
    REPORT("double diag[1]=1 (tau[1]==0)", neq(htmat[1+1*mrs], 1.0));
    REPORT("double diag[2]=1/tau[2]=0.25", neq(htmat[2+2*mrs], 0.25));
    REPORT("double off-diag [1+0*4] unchanged (0)", neq(htmat[1+0*mrs], 0.0));
    sycl::free(dtmat, q); sycl::free(dtau, q);
  }

  // double: tau_offset=1, nstor=2, tau=[99,2.0,0.0]
  {
    int nstor=2, mrs=3, tau_offset=1;
    double h_tau[]={99.0, 2.0, 0.0};
    double *dtmat = sycl::malloc_device<double>(mrs*mrs, q);
    double *dtau  = sycl::malloc_device<double>(3, q);
    q.memset(dtmat, 0, mrs*mrs*sizeof(double)).wait();
    q.memcpy(dtau, h_tau, 3*sizeof(double)).wait();
    int threads=32, blocks=(nstor+threads-1)/threads;
    if (blocks<1) blocks=1;
    q.parallel_for(sycl::nd_range<1>(blocks*threads, threads), [=](sycl::nd_item<1> it){
      gpu_set_tmat_diag_from_tau_kernel(dtmat, dtau, mrs, nstor, tau_offset, it);
    }).wait();
    double htmat[3*3];
    dev_read_n(q, htmat, dtmat, mrs*mrs);
    REPORT("double tau_offset=1: diag[0]=0.5", neq(htmat[0+0*mrs], 0.5));
    REPORT("double tau_offset=1: diag[1]=1.0", neq(htmat[1+1*mrs], 1.0));
    sycl::free(dtmat, q); sycl::free(dtau, q);
  }

  // float: nstor=2, mrs=2, tau=[3.0,0.0]
  {
    int nstor=2, mrs=2, tau_offset=0;
    float h_tau[]={3.0f, 0.0f};
    float *dtmat = sycl::malloc_device<float>(mrs*mrs, q);
    float *dtau  = sycl::malloc_device<float>(nstor, q);
    q.memset(dtmat, 0, mrs*mrs*sizeof(float)).wait();
    q.memcpy(dtau, h_tau, nstor*sizeof(float)).wait();
    int threads=32, blocks=(nstor+threads-1)/threads;
    if (blocks<1) blocks=1;
    q.parallel_for(sycl::nd_range<1>(blocks*threads, threads), [=](sycl::nd_item<1> it){
      gpu_set_tmat_diag_from_tau_kernel(dtmat, dtau, mrs, nstor, tau_offset, it);
    }).wait();
    float htmat[2*2];
    dev_read_n(q, htmat, dtmat, mrs*mrs);
    REPORT("float diag[0]=1/3.0", fabs(htmat[0+0*mrs] - 1.0f/3.0f) < 1e-6f);
    REPORT("float diag[1]=1.0 (tau==0)", fabs(htmat[1+1*mrs] - 1.0f) < 1e-6f);
    sycl::free(dtmat, q); sycl::free(dtau, q);
  }
}

// ============================================================
// gpu_copy_hvb_a
// np_rows=np_cols=1, nblk=1, my_prow=my_pcol=0, ics=2, ice=3
// With np=1 and nblk=1: local_index(k,0,1,1,-1)==k for k>=1.
// ic=2: l_colh=2, l_rows=1  -> hvb[0+ld*0]=a[0+lda], then 1.0 overwrite
// ic=3: l_colh=3, l_rows=2  -> hvb[0+ld*1]=a[0+2*lda]=7.0, hvb[1+ld*1]=1.0
// ============================================================
static void test_copy_hvb_a(sycl::queue &q)
{
  printf("\ngpu_copy_hvb_a:\n");

  int np_rows=1, np_cols=1, nblk=1, my_prow=0, my_pcol=0;
  int ics=2, ice=3, lda=4, ld_hvb=4;

  double h_a[12]={0};
  h_a[0 + 1*lda] = 5.0;
  h_a[0 + 2*lda] = 7.0;
  h_a[1 + 2*lda] = 8.0;
  double h_hvb[12]={0};

  double *da   = sycl::malloc_device<double>(12, q);
  double *dhvb = sycl::malloc_device<double>(12, q);
  q.memcpy(da,   h_a,   12*sizeof(double)).wait();
  q.memcpy(dhvb, h_hvb, 12*sizeof(double)).wait();

  const int SM=1, TPB=32;
  q.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
    gpu_copy_hvb_a_kernel(dhvb, da, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, it);
  }).wait();

  dev_read_n(q, h_hvb, dhvb, 12);
  REPORT("double copy_hvb_a: ic=2 last-row set to 1.0", neq(h_hvb[0 + ld_hvb*0], 1.0));
  REPORT("double copy_hvb_a: ic=3 row 0 = 7.0",         neq(h_hvb[0 + ld_hvb*1], 7.0));
  REPORT("double copy_hvb_a: ic=3 last-row set to 1.0", neq(h_hvb[1 + ld_hvb*1], 1.0));

  sycl::free(da, q); sycl::free(dhvb, q);
}

// ============================================================
// gpu_copy_hvm_hvb (SYCL behaviour)
// ics=2, ice=3, nstor=0, np_rows=1, nblk=1, my_prow=0, ld_hvm=4, ld_hvb=4
// tau[1]=2.0 (ic=2 → copy), tau[2]=0.0 (ic=3 → ZERO-FILL)
// hvb col 0 = [10,0,0,0], col 1 = [50,60,0,0]
// hvm initialised to -1.0 (sentinel)
//
// ic=2: l_rows=1 -> hvm[0+4*0]=10; hvm[1..3+4*0]=0 (zero-fill tail)
// ic=3: tau==0   -> entire hvm col 1 zero-filled (SYCL-specific!)
// ============================================================
static void test_copy_hvm_hvb(sycl::queue &q)
{
  printf("\ngpu_copy_hvm_hvb:\n");

  int np_rows=1, nblk=1, my_prow=0;
  int ics=2, ice=3, nstor=0, ld_hvm=4, ld_hvb=4;

  double h_tau[3]={99.0, 2.0, 0.0};
  double h_hvb[8]={10.0, 0, 0, 0,   50.0, 60.0, 0, 0};
  double h_hvm[8]; for (int k=0; k<8; k++) h_hvm[k]=-1.0;

  double *dtau = sycl::malloc_device<double>(3, q);
  double *dhvb = sycl::malloc_device<double>(8, q);
  double *dhvm = sycl::malloc_device<double>(8, q);
  q.memcpy(dtau, h_tau, 3*sizeof(double)).wait();
  q.memcpy(dhvb, h_hvb, 8*sizeof(double)).wait();
  q.memcpy(dhvm, h_hvm, 8*sizeof(double)).wait();

  const int SM=1, TPB=32;
  q.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
    gpu_copy_hvm_hvb_kernel(dhvm, dhvb, dtau, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, it);
  }).wait();

  dev_read_n(q, h_hvm, dhvm, 8);

  // ic=2 was copied (tau[1]=2.0 != 0)
  REPORT("double copy_hvm_hvb: ic=2 hvm[0]=10.0",          neq(h_hvm[0], 10.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[1]=0 (zero-fill)", neq(h_hvm[1],  0.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[2]=0 (zero-fill)", neq(h_hvm[2],  0.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[3]=0 (zero-fill)", neq(h_hvm[3],  0.0));
  // ic=3: tau==0 → SYCL zero-fills entire column (unlike CUDA which skips)
  REPORT("double copy_hvm_hvb: ic=3 tau==0, hvm[4]=0 (SYCL zero-fill)", neq(h_hvm[4], 0.0));
  REPORT("double copy_hvm_hvb: ic=3 tau==0, hvm[5]=0 (SYCL zero-fill)", neq(h_hvm[5], 0.0));
  REPORT("double copy_hvm_hvb: ic=3 tau==0, hvm[6]=0 (SYCL zero-fill)", neq(h_hvm[6], 0.0));
  REPORT("double copy_hvm_hvb: ic=3 tau==0, hvm[7]=0 (SYCL zero-fill)", neq(h_hvm[7], 0.0));

  sycl::free(dtau, q); sycl::free(dhvb, q); sycl::free(dhvm, q);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for trans_ev kernels (SYCL) ===\n");

  sycl::queue q(sycl::default_selector_v);

  test_set_tmat_diag_from_tau(q);
  test_copy_hvb_a(q);
  test_copy_hvm_hvb(q);

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
