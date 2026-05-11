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

// Unit tests for solve_tridi GPU kernels.
// AMD ROCm/HIP port of test/unit_tests/CUDA/test_solve_tridi_kernels.cu.
// Covers: gpu_transform_one_column, gpu_transform_two_columns,
//         gpu_update_d, gpu_copy_qmat1_to_qmat2,
//         gpu_fill_array, gpu_copy_qtmp1_to_qtmp1_tmp,
//         gpu_compute_nnzl_nnzu_val_part1/2.

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
#include "../../../../src/solve_tridi/GPU/gpu_transform_columns.h"
#include "../../../../src/solve_tridi/GPU/gpu_solve_tridi_col.h"
#include "../../../../src/solve_tridi/GPU/gpu_merge_systems.h"

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

static bool neq (double a, double b, double tol=1e-10) { return fabs(a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }

template <typename T>
static void dev_read_n(T *h, T *d, int n) {
  HIP_CHECK(hipMemcpy(h, d, n*sizeof(T), hipMemcpyDeviceToHost));
}

// ============================================================
// gpu_transform_one_column: c = alpha*a + beta*b
// ============================================================
static void test_transform_one_column()
{
  printf("\ngpu_transform_one_column:\n");

  // double: a=[1,2,3,4], b=[5,6,7,8], alpha=2, beta=3 → c=[17,22,27,32]
  {
    double ha[]={1,2,3,4}, hb[]={5,6,7,8};
    double halpha=2.0, hbeta=3.0;
    double *da,*db,*dc,*dalpha,*dbeta;
    HIP_CHECK(hipMalloc(&da,    4*sizeof(double)));
    HIP_CHECK(hipMalloc(&db,    4*sizeof(double)));
    HIP_CHECK(hipMalloc(&dc,    4*sizeof(double)));
    HIP_CHECK(hipMalloc(&dalpha,  sizeof(double)));
    HIP_CHECK(hipMalloc(&dbeta,   sizeof(double)));
    HIP_CHECK(hipMemcpy(da,     ha,     4*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db,     hb,     4*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dalpha, &halpha, sizeof(double),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbeta,  &hbeta,  sizeof(double),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dc, 0, 4*sizeof(double)));
    gpu_transform_one_column<double>(da, db, dc, dalpha, dbeta, 4, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    double hc[4];
    dev_read_n(hc, dc, 4);
    REPORT("double transform_one_col: c[0]==17", neq(hc[0], 17.0));
    REPORT("double transform_one_col: c[1]==22", neq(hc[1], 22.0));
    REPORT("double transform_one_col: c[2]==27", neq(hc[2], 27.0));
    REPORT("double transform_one_col: c[3]==32", neq(hc[3], 32.0));
    HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dc));
    HIP_CHECK(hipFree(dalpha)); HIP_CHECK(hipFree(dbeta));
  }

  // float: same values
  {
    float ha[]={1,2,3,4}, hb[]={5,6,7,8};
    float halpha=2.0f, hbeta=3.0f;
    float *da,*db,*dc,*dalpha,*dbeta;
    HIP_CHECK(hipMalloc(&da,    4*sizeof(float)));
    HIP_CHECK(hipMalloc(&db,    4*sizeof(float)));
    HIP_CHECK(hipMalloc(&dc,    4*sizeof(float)));
    HIP_CHECK(hipMalloc(&dalpha,  sizeof(float)));
    HIP_CHECK(hipMalloc(&dbeta,   sizeof(float)));
    HIP_CHECK(hipMemcpy(da,     ha,     4*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db,     hb,     4*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dalpha, &halpha, sizeof(float),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbeta,  &hbeta,  sizeof(float),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dc, 0, 4*sizeof(float)));
    gpu_transform_one_column<float>(da, db, dc, dalpha, dbeta, 4, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    float hc[4];
    dev_read_n(hc, dc, 4);
    REPORT("float transform_one_col: c[0]==17", neqf(hc[0], 17.0f));
    REPORT("float transform_one_col: c[3]==32", neqf(hc[3], 32.0f));
    HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dc));
    HIP_CHECK(hipFree(dalpha)); HIP_CHECK(hipFree(dbeta));
  }
}

// ============================================================
// gpu_transform_two_columns
// ============================================================
static void test_transform_two_columns()
{
  printf("\ngpu_transform_two_columns:\n");

  int ldq=4, l_rows=2, l_rqs=1, l_rqe=2, lc1=1, lc2=2;

  double hq[4*2] = {1,2,0,0, 3,4,0,0};
  double hqtrans[4] = {2,4,3,5};

  double *dq, *dqtrans, *dtmp;
  HIP_CHECK(hipMalloc(&dq,      4*2*sizeof(double)));
  HIP_CHECK(hipMalloc(&dqtrans, 4*sizeof(double)));
  HIP_CHECK(hipMalloc(&dtmp,    4*sizeof(double)));
  HIP_CHECK(hipMemcpy(dq,      hq,      4*2*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dqtrans, hqtrans, 4*sizeof(double),   hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dtmp, 0, 4*sizeof(double)));

  gpu_transform_two_columns<double>(dq, dqtrans, dtmp, ldq, l_rows, l_rqs, l_rqe, lc1, lc2, 1, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());

  dev_read_n(hq, dq, 4*2);
  REPORT("double transform_two_cols: q_col1[0]==14", neq(hq[0 + (lc1-1)*ldq], 14.0));
  REPORT("double transform_two_cols: q_col1[1]==20", neq(hq[1 + (lc1-1)*ldq], 20.0));
  REPORT("double transform_two_cols: q_col2[0]==18", neq(hq[0 + (lc2-1)*ldq], 18.0));
  REPORT("double transform_two_cols: q_col2[1]==26", neq(hq[1 + (lc2-1)*ldq], 26.0));

  HIP_CHECK(hipFree(dq)); HIP_CHECK(hipFree(dqtrans)); HIP_CHECK(hipFree(dtmp));
}

// ============================================================
// gpu_update_d
// ============================================================
static void test_update_d()
{
  printf("\ngpu_update_d:\n");

  double hd[] = {10, 20, 30, 40};
  double he[] = {1,  2,  3,  4};
  int    hl[] = {2};
  double *dd, *de; int *dl;
  HIP_CHECK(hipMalloc(&dd, 4*sizeof(double)));
  HIP_CHECK(hipMalloc(&de, 4*sizeof(double)));
  HIP_CHECK(hipMalloc(&dl, 1*sizeof(int)));
  HIP_CHECK(hipMemcpy(dd, hd, 4*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(de, he, 4*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dl, hl, 1*sizeof(int),    hipMemcpyHostToDevice));
  gpu_update_d<double>(dd, de, dl, 2, 4, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(hd, dd, 4);
  REPORT("double update_d: d[0] unchanged=10", neq(hd[0], 10.0));
  REPORT("double update_d: d[1]-=|e[1]|=18",  neq(hd[1], 18.0));
  REPORT("double update_d: d[2]-=|e[1]|=28",  neq(hd[2], 28.0));
  REPORT("double update_d: d[3] unchanged=40", neq(hd[3], 40.0));
  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(de)); HIP_CHECK(hipFree(dl));

  float hdf[] = {10,20,30,40}, hef[] = {1,2,3,4};
  int   hlf[] = {2};
  float *ddf, *def; int *dlf;
  HIP_CHECK(hipMalloc(&ddf, 4*sizeof(float)));
  HIP_CHECK(hipMalloc(&def, 4*sizeof(float)));
  HIP_CHECK(hipMalloc(&dlf, 1*sizeof(int)));
  HIP_CHECK(hipMemcpy(ddf, hdf, 4*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(def, hef, 4*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dlf, hlf, 1*sizeof(int),   hipMemcpyHostToDevice));
  gpu_update_d<float>(ddf, def, dlf, 2, 4, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(hdf, ddf, 4);
  REPORT("float  update_d: d[1]-=|e[1]|=18",  neqf(hdf[1], 18.0f));
  REPORT("float  update_d: d[2]-=|e[1]|=28",  neqf(hdf[2], 28.0f));
  HIP_CHECK(hipFree(ddf)); HIP_CHECK(hipFree(def)); HIP_CHECK(hipFree(dlf));
}

// ============================================================
// gpu_copy_qmat1_to_qmat2  [KNOWN BUG: only copies diagonal]
// ============================================================
static void test_copy_qmat1_to_qmat2()
{
  printf("\ngpu_copy_qmat1_to_qmat2 (verifies diagonal-only copy bug):\n");

  int max_size = 4;
  double h1[16], h2[16];
  for (int k=0; k<16; k++) h1[k] = (double)(k+1);
  for (int k=0; k<16; k++) h2[k] = 0.0;

  double *d1, *d2;
  HIP_CHECK(hipMalloc(&d1, 16*sizeof(double)));
  HIP_CHECK(hipMalloc(&d2, 16*sizeof(double)));
  HIP_CHECK(hipMemcpy(d1, h1, 16*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2, 16*sizeof(double), hipMemcpyHostToDevice));

  gpu_copy_qmat1_to_qmat2<double>(d1, d2, max_size, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(h2, d2, 16);

  REPORT("double copy_qmat (diag): [0,0]=1",  neq(h2[0+4*0],  1.0));
  REPORT("double copy_qmat (diag): [1,1]=6",  neq(h2[1+4*1],  6.0));
  REPORT("double copy_qmat (diag): [2,2]=11", neq(h2[2+4*2], 11.0));
  REPORT("double copy_qmat (diag): [3,3]=16", neq(h2[3+4*3], 16.0));
  REPORT("double copy_qmat (off-diag [1,0]=0)", neq(h2[1+4*0], 0.0));
  REPORT("double copy_qmat (off-diag [0,1]=0)", neq(h2[0+4*1], 0.0));
  REPORT("double copy_qmat (off-diag [2,1]=0)", neq(h2[2+4*1], 0.0));

  HIP_CHECK(hipFree(d1)); HIP_CHECK(hipFree(d2));
}

// ============================================================
// gpu_fill_array
// ============================================================
static void test_fill_array()
{
  printf("\ngpu_fill_array:\n");

  // double: 6 elements set to 3.14
  {
    double hval = 3.14, hout[6];
    double *dval, *dout;
    HIP_CHECK(hipMalloc(&dval, sizeof(double)));
    HIP_CHECK(hipMalloc(&dout, 6*sizeof(double)));
    HIP_CHECK(hipMemcpy(dval, &hval, sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dout, 0, 6*sizeof(double)));
    gpu_fill_array<double>(dout, dval, 6, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    dev_read_n(hout, dout, 6);
    bool ok = true;
    for (int k=0; k<6; k++) ok = ok && neq(hout[k], 3.14);
    REPORT("double fill_array: all 6 elements == 3.14", ok);
    HIP_CHECK(hipFree(dval)); HIP_CHECK(hipFree(dout));
  }

  // float: 4 elements set to 2.5f
  {
    float hval = 2.5f, hout[4];
    float *dval, *dout;
    HIP_CHECK(hipMalloc(&dval, sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, 4*sizeof(float)));
    HIP_CHECK(hipMemcpy(dval, &hval, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dout, 0, 4*sizeof(float)));
    gpu_fill_array<float>(dout, dval, 4, 1, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    dev_read_n(hout, dout, 4);
    bool ok = true;
    for (int k=0; k<4; k++) ok = ok && (fabsf(hout[k]-2.5f) < 1e-6f);
    REPORT("float  fill_array: all 4 elements == 2.5", ok);
    HIP_CHECK(hipFree(dval)); HIP_CHECK(hipFree(dout));
  }
}

// ============================================================
// gpu_copy_qtmp1_to_qtmp1_tmp
// ============================================================
static void test_copy_qtmp1_to_qtmp1_tmp()
{
  printf("\ngpu_copy_qtmp1_to_qtmp1_tmp:\n");

  int k=3, l=2;
  double hsrc[6]={1,2,3,4,5,6}, hdst[6]={0};
  double *dsrc, *ddst;
  HIP_CHECK(hipMalloc(&dsrc, 6*sizeof(double)));
  HIP_CHECK(hipMalloc(&ddst, 6*sizeof(double)));
  HIP_CHECK(hipMemcpy(dsrc, hsrc, 6*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(ddst, 0, 6*sizeof(double)));
  gpu_copy_qtmp1_to_qtmp1_tmp<double>(dsrc, ddst, k, l, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(hdst, ddst, 6);
  bool ok = true;
  for (int n=0; n<6; n++) ok = ok && neq(hdst[n], (double)(n+1));
  REPORT("double copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
  HIP_CHECK(hipFree(dsrc)); HIP_CHECK(hipFree(ddst));

  float hsrcf[6]={1,2,3,4,5,6}, hdstf[6]={0};
  float *dsrcf, *ddstf;
  HIP_CHECK(hipMalloc(&dsrcf, 6*sizeof(float)));
  HIP_CHECK(hipMalloc(&ddstf, 6*sizeof(float)));
  HIP_CHECK(hipMemcpy(dsrcf, hsrcf, 6*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(ddstf, 0, 6*sizeof(float)));
  gpu_copy_qtmp1_to_qtmp1_tmp<float>(dsrcf, ddstf, k, l, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(hdstf, ddstf, 6);
  ok = true;
  for (int n=0; n<6; n++) ok = ok && (fabsf(hdstf[n] - (float)(n+1)) < 1e-6f);
  REPORT("float  copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
  HIP_CHECK(hipFree(dsrcf)); HIP_CHECK(hipFree(ddstf));
}

// ============================================================
// gpu_compute_nnzl_nnzu_val_part1 and part2
// ============================================================
static void test_compute_nnzl_nnzu()
{
  printf("\ngpu_compute_nnzl_nnzu_val_part1+2:\n");

  int na1=4, na=4, npc_n=1, np_rem=0;
  int h_pcol[4]   = {0,0,0,0};
  int h_idx1[4]   = {1,2,3,4};
  int h_coltyp[4] = {1,3,2,1};
  int h_nnzu[4]   = {0}, h_nnzl[4] = {0};

  int *d_pcol, *d_idx1, *d_coltyp, *d_nnzu, *d_nnzl;
  HIP_CHECK(hipMalloc(&d_pcol,   4*sizeof(int)));
  HIP_CHECK(hipMalloc(&d_idx1,   4*sizeof(int)));
  HIP_CHECK(hipMalloc(&d_coltyp, 4*sizeof(int)));
  HIP_CHECK(hipMalloc(&d_nnzu,   4*sizeof(int)));
  HIP_CHECK(hipMalloc(&d_nnzl,   4*sizeof(int)));
  HIP_CHECK(hipMemcpy(d_pcol,   h_pcol,   4*sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_idx1,   h_idx1,   4*sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_coltyp, h_coltyp, 4*sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_nnzu, 0, 4*sizeof(int)));
  HIP_CHECK(hipMemset(d_nnzl, 0, 4*sizeof(int)));

  gpu_compute_nnzl_nnzu_val_part1(d_pcol, d_idx1, d_coltyp, d_nnzu, d_nnzl,
                                  na, na1, np_rem, npc_n, 0, 0, 1, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(h_nnzu, d_nnzu, 4);
  dev_read_n(h_nnzl, d_nnzl, 4);
  REPORT("part1: nnzu_val[0]=1 (coltyp=1)", h_nnzu[0] == 1);
  REPORT("part1: nnzu_val[1]=0 (coltyp=3)", h_nnzu[1] == 0);
  REPORT("part1: nnzu_val[2]=1 (coltyp=2)", h_nnzu[2] == 1);
  REPORT("part1: nnzu_val[3]=1 (coltyp=1)", h_nnzu[3] == 1);
  REPORT("part1: nnzl_val[0]=0 (coltyp=1)", h_nnzl[0] == 0);
  REPORT("part1: nnzl_val[1]=1 (coltyp=3)", h_nnzl[1] == 1);
  REPORT("part1: nnzl_val[2]=1 (coltyp=2)", h_nnzl[2] == 1);
  REPORT("part1: nnzl_val[3]=0 (coltyp=1)", h_nnzl[3] == 0);

  gpu_compute_nnzl_nnzu_val_part2(d_nnzu, d_nnzl, na, na1, 0, 0, npc_n, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  dev_read_n(h_nnzu, d_nnzu, 4);
  dev_read_n(h_nnzl, d_nnzl, 4);
  REPORT("part2: nnzu_val[0]=1",  h_nnzu[0] == 1);
  REPORT("part2: nnzu_val[1]=0",  h_nnzu[1] == 0);
  REPORT("part2: nnzu_val[2]=2",  h_nnzu[2] == 2);
  REPORT("part2: nnzu_val[3]=3",  h_nnzu[3] == 3);
  REPORT("part2: nnzl_val[0]=0",  h_nnzl[0] == 0);
  REPORT("part2: nnzl_val[1]=1",  h_nnzl[1] == 1);
  REPORT("part2: nnzl_val[2]=2",  h_nnzl[2] == 2);
  REPORT("part2: nnzl_val[3]=0",  h_nnzl[3] == 0);

  HIP_CHECK(hipFree(d_pcol));   HIP_CHECK(hipFree(d_idx1));
  HIP_CHECK(hipFree(d_coltyp)); HIP_CHECK(hipFree(d_nnzu));
  HIP_CHECK(hipFree(d_nnzl));
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for solve_tridi GPU kernels (ROCm/HIP) ===\n");

  test_transform_one_column();
  test_transform_two_columns();
  test_update_d();
  test_copy_qmat1_to_qmat2();
  test_fill_array();
  test_copy_qtmp1_to_qtmp1_tmp();
  test_compute_nnzl_nnzu();

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
