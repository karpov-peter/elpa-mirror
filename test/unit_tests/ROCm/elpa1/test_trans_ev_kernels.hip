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

// Unit tests for kernels from src/elpa1/GPU/trans_ev_gpu.h.
// AMD ROCm/HIP port of test/unit_tests/CUDA/test_trans_ev_kernels.cu.
// Covers: gpu_set_tmat_diag_from_tau, gpu_copy_hvb_a, gpu_copy_hvm_hvb.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_AMD_GPU_VERSION

#include <stdint.h>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <algorithm>
#include <type_traits>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"
#include "../../../../src/elpa1/GPU/trans_ev_gpu.h"

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t _err = (call);                                                 \
    if (_err != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, hipGetErrorString(_err));                   \
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
static void dev_read_n(T *h, T *d, int n) {
  HIP_CHECK(hipMemcpy(h, d, n*sizeof(T), hipMemcpyDeviceToHost));
}

// ============================================================
// gpu_set_tmat_diag_from_tau
// ============================================================
static void test_set_tmat_diag_from_tau()
{
  printf("\ngpu_set_tmat_diag_from_tau:\n");

  // double: nstor=3, max_stored_rows=4, tau=[2.0, 0.0, 4.0], tau_offset=0
  {
    int nstor = 3, mrs = 4, tau_offset = 0;
    double h_tau[]  = {2.0, 0.0, 4.0};
    double *dtmat, *dtau;
    HIP_CHECK(hipMalloc(&dtmat, mrs*mrs*sizeof(double)));
    HIP_CHECK(hipMalloc(&dtau,  nstor*sizeof(double)));
    HIP_CHECK(hipMemset(dtmat, 0, mrs*mrs*sizeof(double)));
    HIP_CHECK(hipMemcpy(dtau, h_tau, nstor*sizeof(double), hipMemcpyHostToDevice));
    gpu_set_tmat_diag_from_tau<double>(dtmat, dtau, mrs, nstor, tau_offset, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    double htmat[4*4];
    dev_read_n(htmat, dtmat, mrs*mrs);
    REPORT("double diag[0]=1/tau[0]=0.5",  neq(htmat[0+0*mrs], 0.5));
    REPORT("double diag[1]=1  (tau[1]==0)", neq(htmat[1+1*mrs], 1.0));
    REPORT("double diag[2]=1/tau[2]=0.25", neq(htmat[2+2*mrs], 0.25));
    REPORT("double off-diag [1+0*4] unchanged (0)", neq(htmat[1+0*mrs], 0.0));
    HIP_CHECK(hipFree(dtmat)); HIP_CHECK(hipFree(dtau));
  }

  // double: tau_offset=1, nstor=2, tau=[99, 2.0, 0.0]
  {
    int nstor = 2, mrs = 3, tau_offset = 1;
    double h_tau[] = {99.0, 2.0, 0.0};
    double *dtmat, *dtau;
    HIP_CHECK(hipMalloc(&dtmat, mrs*mrs*sizeof(double)));
    HIP_CHECK(hipMalloc(&dtau,  3*sizeof(double)));
    HIP_CHECK(hipMemset(dtmat, 0, mrs*mrs*sizeof(double)));
    HIP_CHECK(hipMemcpy(dtau, h_tau, 3*sizeof(double), hipMemcpyHostToDevice));
    gpu_set_tmat_diag_from_tau<double>(dtmat, dtau, mrs, nstor, tau_offset, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    double htmat[3*3];
    dev_read_n(htmat, dtmat, mrs*mrs);
    REPORT("double tau_offset=1: diag[0]=0.5",  neq(htmat[0+0*mrs], 0.5));
    REPORT("double tau_offset=1: diag[1]=1.0",  neq(htmat[1+1*mrs], 1.0));
    HIP_CHECK(hipFree(dtmat)); HIP_CHECK(hipFree(dtau));
  }

  // float: nstor=2, max_stored_rows=2, tau=[3.0, 0.0]
  {
    int nstor = 2, mrs = 2, tau_offset = 0;
    float h_tau[] = {3.0f, 0.0f};
    float *dtmat, *dtau;
    HIP_CHECK(hipMalloc(&dtmat, mrs*mrs*sizeof(float)));
    HIP_CHECK(hipMalloc(&dtau,  nstor*sizeof(float)));
    HIP_CHECK(hipMemset(dtmat, 0, mrs*mrs*sizeof(float)));
    HIP_CHECK(hipMemcpy(dtau, h_tau, nstor*sizeof(float), hipMemcpyHostToDevice));
    gpu_set_tmat_diag_from_tau<float>(dtmat, dtau, mrs, nstor, tau_offset, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    float htmat[2*2];
    dev_read_n(htmat, dtmat, mrs*mrs);
    REPORT("float diag[0]=1/3.0", fabsf(htmat[0+0*mrs] - 1.0f/3.0f) < 1e-6f);
    REPORT("float diag[1]=1.0 (tau==0)", fabsf(htmat[1+1*mrs] - 1.0f) < 1e-6f);
    HIP_CHECK(hipFree(dtmat)); HIP_CHECK(hipFree(dtau));
  }
}

// ============================================================
// gpu_copy_hvb_a
// ============================================================
static void test_copy_hvb_a()
{
  printf("\ngpu_copy_hvb_a:\n");

  int np_rows=1, np_cols=1, nblk=1, my_prow=0, my_pcol=0;
  int ics=2, ice=3, lda=4, ld_hvb=4;

  double h_a[12] = {0};
  h_a[0 + 1*lda] = 5.0;
  h_a[0 + 2*lda] = 7.0;
  h_a[1 + 2*lda] = 8.0;

  double h_hvb[12] = {0};

  double *da, *dhvb;
  HIP_CHECK(hipMalloc(&da,   12*sizeof(double)));
  HIP_CHECK(hipMalloc(&dhvb, 12*sizeof(double)));
  HIP_CHECK(hipMemcpy(da,   h_a,   12*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dhvb, h_hvb, 12*sizeof(double), hipMemcpyHostToDevice));

  gpu_copy_hvb_a<double>(dhvb, da, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, 1, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());

  dev_read_n(h_hvb, dhvb, 12);

  REPORT("double copy_hvb_a: ic=2 last-row set to 1.0", neq(h_hvb[0 + ld_hvb*0], 1.0));
  REPORT("double copy_hvb_a: ic=3 row 0 = 7.0",         neq(h_hvb[0 + ld_hvb*1], 7.0));
  REPORT("double copy_hvb_a: ic=3 last-row set to 1.0", neq(h_hvb[1 + ld_hvb*1], 1.0));

  HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(dhvb));
}

// ============================================================
// gpu_copy_hvm_hvb
// ============================================================
static void test_copy_hvm_hvb()
{
  printf("\ngpu_copy_hvm_hvb:\n");

  int np_rows=1, nblk=1, my_prow=0;
  int ics=2, ice=3, nstor=0, ld_hvm=4, ld_hvb=4;

  double h_tau[3]  = {99.0, 2.0, 0.0};
  double h_hvb[8]  = {10.0, 0, 0, 0, 50.0, 60.0, 0, 0};
  double h_hvm[8];
  for (int k=0; k<8; k++) h_hvm[k] = -1.0;

  double *dtau, *dhvb, *dhvm;
  HIP_CHECK(hipMalloc(&dtau, 3*sizeof(double)));
  HIP_CHECK(hipMalloc(&dhvb, 8*sizeof(double)));
  HIP_CHECK(hipMalloc(&dhvm, 8*sizeof(double)));
  HIP_CHECK(hipMemcpy(dtau, h_tau, 3*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dhvb, h_hvb, 8*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dhvm, h_hvm, 8*sizeof(double), hipMemcpyHostToDevice));

  gpu_copy_hvm_hvb<double>(dhvm, dhvb, dtau, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, 1, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());

  dev_read_n(h_hvm, dhvm, 8);

  REPORT("double copy_hvm_hvb: ic=2 hvm[0]=10.0",          neq(h_hvm[0], 10.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[1]=0 (zero-fill)", neq(h_hvm[1],  0.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[2]=0 (zero-fill)", neq(h_hvm[2],  0.0));
  REPORT("double copy_hvm_hvb: ic=2 hvm[3]=0 (zero-fill)", neq(h_hvm[3],  0.0));
  REPORT("double copy_hvm_hvb: ic=3 skipped, hvm[4]=-1",   neq(h_hvm[4], -1.0));
  REPORT("double copy_hvm_hvb: ic=3 skipped, hvm[5]=-1",   neq(h_hvm[5], -1.0));

  HIP_CHECK(hipFree(dtau)); HIP_CHECK(hipFree(dhvb)); HIP_CHECK(hipFree(dhvm));
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for trans_ev_gpu.h kernels (ROCm/HIP) ===\n");

  test_set_tmat_diag_from_tau();
  test_copy_hvb_a();
  test_copy_hvm_hvb();

  printf("\n=== Summary: %d failure(s) ===\n", g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
  fprintf(stderr, "Error: this test requires WITH_AMD_GPU_VERSION\n");
  abort();
}

#endif /* WITH_AMD_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
