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

// Unit tests for kernels from src/elpa1/GPU/trans_ev_gpu.h (CUDA backend).
// Covers: gpu_set_tmat_diag_from_tau, gpu_copy_hvb_a, gpu_copy_hvm_hvb.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <type_traits>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"
#include "../../../../src/elpa1/GPU/trans_ev_gpu.h"

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

static bool neq(double a, double b, double tol=1e-10) { return fabs(a-b) < tol; }

template <typename T>
static T dev_read(T *d) {
  T h;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

template <typename T>
static void dev_read_n(T *h, T *d, int n) {
  CUDA_CHECK(cudaMemcpy(h, d, n*sizeof(T), cudaMemcpyDeviceToHost));
}

// ============================================================
// gpu_set_tmat_diag_from_tau
//
// tmat_dev[i + i*max_stored_rows] = 1/tau[i+tau_offset]  if tau != 0
//                                 = 1                     if tau == 0
// ============================================================
static void test_set_tmat_diag_from_tau()
{
  printf("\ngpu_set_tmat_diag_from_tau:\n");

  // double: nstor=3, max_stored_rows=4, tau=[2.0, 0.0, 4.0], tau_offset=0
  // Expected diagonal: [0.5, 1.0, 0.25]
  {
    int nstor = 3, mrs = 4, tau_offset = 0;
    double h_tau[]  = {2.0, 0.0, 4.0};
    double *dtmat, *dtau;
    CUDA_CHECK(cudaMalloc(&dtmat, mrs*mrs*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dtau,  nstor*sizeof(double)));
    CUDA_CHECK(cudaMemset(dtmat, 0, mrs*mrs*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dtau, h_tau, nstor*sizeof(double), cudaMemcpyHostToDevice));
    gpu_set_tmat_diag_from_tau<double>(dtmat, dtau, mrs, nstor, tau_offset, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double htmat[4*4];
    dev_read_n(htmat, dtmat, mrs*mrs);
    REPORT("double diag[0]=1/tau[0]=0.5",  neq(htmat[0+0*mrs], 0.5));
    REPORT("double diag[1]=1  (tau[1]==0)", neq(htmat[1+1*mrs], 1.0));
    REPORT("double diag[2]=1/tau[2]=0.25", neq(htmat[2+2*mrs], 0.25));
    REPORT("double off-diag [1+0*4] unchanged (0)", neq(htmat[1+0*mrs], 0.0));
    CUDA_CHECK(cudaFree(dtmat)); CUDA_CHECK(cudaFree(dtau));
  }

  // double: tau_offset=1, nstor=2, tau=[99, 2.0, 0.0]
  // Expected diagonal: [0.5, 1.0]
  {
    int nstor = 2, mrs = 3, tau_offset = 1;
    double h_tau[] = {99.0, 2.0, 0.0};
    double *dtmat, *dtau;
    CUDA_CHECK(cudaMalloc(&dtmat, mrs*mrs*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dtau,  3*sizeof(double)));
    CUDA_CHECK(cudaMemset(dtmat, 0, mrs*mrs*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dtau, h_tau, 3*sizeof(double), cudaMemcpyHostToDevice));
    gpu_set_tmat_diag_from_tau<double>(dtmat, dtau, mrs, nstor, tau_offset, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double htmat[3*3];
    dev_read_n(htmat, dtmat, mrs*mrs);
    REPORT("double tau_offset=1: diag[0]=0.5",  neq(htmat[0+0*mrs], 0.5));
    REPORT("double tau_offset=1: diag[1]=1.0",  neq(htmat[1+1*mrs], 1.0));
    CUDA_CHECK(cudaFree(dtmat)); CUDA_CHECK(cudaFree(dtau));
  }

  // float: nstor=2, max_stored_rows=2, tau=[3.0, 0.0], tau_offset=0
  {
    int nstor = 2, mrs = 2, tau_offset = 0;
    float h_tau[] = {3.0f, 0.0f};
    float *dtmat, *dtau;
    CUDA_CHECK(cudaMalloc(&dtmat, mrs*mrs*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dtau,  nstor*sizeof(float)));
    CUDA_CHECK(cudaMemset(dtmat, 0, mrs*mrs*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dtau, h_tau, nstor*sizeof(float), cudaMemcpyHostToDevice));
    gpu_set_tmat_diag_from_tau<float>(dtmat, dtau, mrs, nstor, tau_offset, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float htmat[2*2];
    dev_read_n(htmat, dtmat, mrs*mrs);
    REPORT("float diag[0]=1/3.0", fabs(htmat[0+0*mrs] - 1.0f/3.0f) < 1e-6f);
    REPORT("float diag[1]=1.0 (tau==0)", fabs(htmat[1+1*mrs] - 1.0f) < 1e-6f);
    CUDA_CHECK(cudaFree(dtmat)); CUDA_CHECK(cudaFree(dtau));
  }
}

// ============================================================
// gpu_copy_hvb_a
//
// Setup: np_rows=np_cols=1, nblk=1, my_prow=my_pcol=0, ics=2, ice=3
// With np=1 and nblk=1, local_index(k,0,1,1,-1) == k for k >= 1.
//
// ic=2: l_colh=2, l_rows=1
//   hvb[0 + ld_hvb*0] = a[0 + (2-1)*lda] = a[lda]
//   prow condition: hvb[0] = 1.0  (overwrites)
// ic=3: l_colh=3, l_rows=2
//   hvb[0 + ld_hvb*1] = a[0 + 2*lda]       -> 7.0
//   hvb[1 + ld_hvb*1] = a[1 + 2*lda]       -> 8.0
//   prow condition (thread 1): hvb[1 + ld_hvb*1] = 1.0 (overwrites 8.0)
// ============================================================
static void test_copy_hvb_a()
{
  printf("\ngpu_copy_hvb_a:\n");

  int np_rows=1, np_cols=1, nblk=1, my_prow=0, my_pcol=0;
  int ics=2, ice=3, lda=4, ld_hvb=4;

  // a: 4 rows x 3 cols (col-major), only relevant entries set
  double h_a[12] = {0};
  h_a[0 + 1*lda] = 5.0;   // col 1 (l_colh for ic=2), row 0
  h_a[0 + 2*lda] = 7.0;   // col 2 (l_colh for ic=3), row 0
  h_a[1 + 2*lda] = 8.0;   // col 2, row 1

  double h_hvb[12] = {0};

  double *da, *dhvb;
  CUDA_CHECK(cudaMalloc(&da,   12*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dhvb, 12*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(da,   h_a,   12*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dhvb, h_hvb, 12*sizeof(double), cudaMemcpyHostToDevice));

  gpu_copy_hvb_a<double>(dhvb, da, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  dev_read_n(h_hvb, dhvb, 12);

  // ic=2: hvb[0 + 4*0] = 1.0 (last-row overwrites copy)
  REPORT("double copy_hvb_a: ic=2 last-row set to 1.0", neq(h_hvb[0 + ld_hvb*0], 1.0));
  // ic=3: hvb[0 + 4*1] = 7.0 (first row copied from a)
  REPORT("double copy_hvb_a: ic=3 row 0 = 7.0",         neq(h_hvb[0 + ld_hvb*1], 7.0));
  // ic=3: hvb[1 + 4*1] = 1.0 (last-row overwrites 8.0)
  REPORT("double copy_hvb_a: ic=3 last-row set to 1.0", neq(h_hvb[1 + ld_hvb*1], 1.0));

  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(dhvb));
}

// ============================================================
// gpu_copy_hvm_hvb
//
// Setup: ics=2, ice=3, nstor=0, np_rows=1, nblk=1, my_prow=0
// tau[ic-1]: tau[1]=2.0 (ic=2 → copy), tau[2]=0.0 (ic=3 → skip)
// hvb col 0 = [10, 0, 0, 0], col 1 = [50, 60, 0, 0]
//
// ic=2: l_rows=1, shift_hvm=0 → hvm[0]=hvb[0]=10; hvm[1..3]=0
// ic=3: tau==0 → skip, hvm col 1 unchanged (-1.0 sentinel)
// ============================================================
static void test_copy_hvm_hvb()
{
  printf("\ngpu_copy_hvm_hvb:\n");

  int np_rows=1, nblk=1, my_prow=0;
  int ics=2, ice=3, nstor=0, ld_hvm=4, ld_hvb=4;

  double h_tau[3]  = {99.0, 2.0, 0.0};   // tau[ic-1]: ic=2 → tau[1], ic=3 → tau[2]
  double h_hvb[8]  = {10.0, 0, 0, 0,     // col 0 of hvb (ic=2 − ics=0)
                       50.0, 60.0, 0, 0}; // col 1 of hvb (ic=3 − ics=1)
  // Initialise hvm to -1 so skipped columns are detectable.
  double h_hvm[8];
  for (int k=0; k<8; k++) h_hvm[k] = -1.0;

  double *dtau, *dhvb, *dhvm;
  CUDA_CHECK(cudaMalloc(&dtau, 3*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dhvb, 8*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dhvm, 8*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dtau, h_tau, 3*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dhvb, h_hvb, 8*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dhvm, h_hvm, 8*sizeof(double), cudaMemcpyHostToDevice));

  gpu_copy_hvm_hvb<double>(dhvm, dhvb, dtau, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  dev_read_n(h_hvm, dhvm, 8);

  // ic=2 was copied (tau[1]=2.0 != 0)
  REPORT("double copy_hvm_hvb: ic=2 hvm[0]=10.0",          neq(h_hvm[0], 10.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[1]=0 (zero-fill)", neq(h_hvm[1],  0.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[2]=0 (zero-fill)", neq(h_hvm[2],  0.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[3]=0 (zero-fill)", neq(h_hvm[3],  0.0));
  // ic=3 was skipped (tau[2]=0.0)
  REPORT("double copy_hvm_hvb: ic=3 skipped, hvm[4]=-1",   neq(h_hvm[4], -1.0));
  REPORT("double copy_hvm_hvb: ic=3 skipped, hvm[5]=-1",   neq(h_hvm[5], -1.0));

  CUDA_CHECK(cudaFree(dtau)); CUDA_CHECK(cudaFree(dhvb)); CUDA_CHECK(cudaFree(dhvm));
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for trans_ev_gpu.h kernels (CUDA) ===\n");

  test_set_tmat_diag_from_tau();
  test_copy_hvb_a();
  test_copy_hvm_hvb();

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
