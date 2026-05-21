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

// Unit tests for solve_tridi GPU kernels (CUDA backend).
// Covers: gpu_transform_one_column, gpu_transform_two_columns (gpu_transform_columns.h),
//         gpu_update_d, gpu_copy_qmat1_to_qmat2 (gpu_solve_tridi_col.h),
//         gpu_fill_array, gpu_copy_qtmp1_to_qtmp1_tmp,
//         gpu_compute_nnzl_nnzu_val_part1/2 (gpu_merge_systems.h).

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
#ifdef WANT_HALF_PRECISION_REAL
#include <cuda_fp16.h>
#endif

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"
#include "../../../../src/solve_tridi/GPU/gpu_transform_columns.h"
#include "../../../../src/solve_tridi/GPU/gpu_solve_tridi_col.h"
#include "../../../../src/solve_tridi/GPU/gpu_merge_systems.h"

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

static bool neq (double a, double b, double tol=1e-10) { return fabs(a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }
#ifdef WANT_HALF_PRECISION_REAL
static bool neqh(__half a, __half b, float tol=0.02f) {
  return fabsf(__half2float(a) - __half2float(b)) < tol;
}
#endif

template <typename T>
static void dev_read_n(T *h, T *d, int n) {
  CUDA_CHECK(cudaMemcpy(h, d, n*sizeof(T), cudaMemcpyDeviceToHost));
}

// ============================================================
// gpu_transform_one_column: c = alpha*a + beta*b
// ============================================================
static void test_transform_one_column()
{
  printf("\ngpu_transform_one_column:\n");

  // double: a=[1,2,3,4], b=[5,6,7,8], alpha=2, beta=3
  // expected c=[17,22,27,32]
  {
    double ha[]={1,2,3,4}, hb[]={5,6,7,8};
    double halpha=2.0, hbeta=3.0;
    double *da,*db,*dc,*dalpha,*dbeta;
    CUDA_CHECK(cudaMalloc(&da,    4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&db,    4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dc,    4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dalpha,  sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dbeta,   sizeof(double)));
    CUDA_CHECK(cudaMemcpy(da,     ha,     4*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,     hb,     4*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dalpha, &halpha, sizeof(double),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta,  &hbeta,  sizeof(double),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc, 0, 4*sizeof(double)));
    gpu_transform_one_column<double>(da, db, dc, dalpha, dbeta, 4, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double hc[4];
    dev_read_n(hc, dc, 4);
    REPORT("double transform_one_col: c[0]==17", neq(hc[0], 17.0));
    REPORT("double transform_one_col: c[1]==22", neq(hc[1], 22.0));
    REPORT("double transform_one_col: c[2]==27", neq(hc[2], 27.0));
    REPORT("double transform_one_col: c[3]==32", neq(hc[3], 32.0));
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dalpha)); CUDA_CHECK(cudaFree(dbeta));
  }

  // float: same values
  {
    float ha[]={1,2,3,4}, hb[]={5,6,7,8};
    float halpha=2.0f, hbeta=3.0f;
    float *da,*db,*dc,*dalpha,*dbeta;
    CUDA_CHECK(cudaMalloc(&da,    4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db,    4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc,    4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dalpha,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbeta,   sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,     ha,     4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,     hb,     4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dalpha, &halpha, sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta,  &hbeta,  sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc, 0, 4*sizeof(float)));
    gpu_transform_one_column<float>(da, db, dc, dalpha, dbeta, 4, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float hc[4];
    dev_read_n(hc, dc, 4);
    REPORT("float transform_one_col: c[0]==17", neqf(hc[0], 17.0f));
    REPORT("float transform_one_col: c[3]==32", neqf(hc[3], 32.0f));
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dalpha)); CUDA_CHECK(cudaFree(dbeta));
  }
}

// ============================================================
// gpu_transform_two_columns
//
// q[i + lrqs-1 + (lc-1)*ldq], l_rows=2, l_rqs=1, lc1=1, lc2=2, ldq=4
// qtrans = {qtrans11, qtrans21, qtrans12, qtrans22} = {2,4,3,5}
//
// Before: col1=[1,2], col2=[3,4]
// tmp[i] = q_col1[i]*2 + q_col2[i]*4
// q_col2_new[i] = q_col1[i]*3 + q_col2[i]*5
// q_col1_new[i] = tmp[i]
//
// tmp[0]=1*2+3*4=14;  q_col2_new[0]=1*3+3*5=18;  q_col1_new[0]=14
// tmp[1]=2*2+4*4=20;  q_col2_new[1]=2*3+4*5=26;  q_col1_new[1]=20
// ============================================================
static void test_transform_two_columns()
{
  printf("\ngpu_transform_two_columns:\n");

  int ldq=4, l_rows=2, l_rqs=1, l_rqe=2, lc1=1, lc2=2;

  // q: 4 rows x 2 cols (col-major), col 1 and col 2 set; other entries 0
  double hq[4*2] = {1,2,0,0, 3,4,0,0};  // col1=[1,2,0,0], col2=[3,4,0,0]
  double hqtrans[4] = {2,4,3,5};         // qtrans11=2,qtrans21=4,qtrans12=3,qtrans22=5

  double *dq, *dqtrans, *dtmp;
  CUDA_CHECK(cudaMalloc(&dq,      4*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dqtrans, 4*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dtmp,    4*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dq,      hq,      4*2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dqtrans, hqtrans, 4*sizeof(double),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dtmp, 0, 4*sizeof(double)));

  gpu_transform_two_columns<double>(dq, dqtrans, dtmp, ldq, l_rows, l_rqs, l_rqe, lc1, lc2, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  dev_read_n(hq, dq, 4*2);
  REPORT("double transform_two_cols: q_col1[0]==14", neq(hq[0 + (lc1-1)*ldq], 14.0));
  REPORT("double transform_two_cols: q_col1[1]==20", neq(hq[1 + (lc1-1)*ldq], 20.0));
  REPORT("double transform_two_cols: q_col2[0]==18", neq(hq[0 + (lc2-1)*ldq], 18.0));
  REPORT("double transform_two_cols: q_col2[1]==26", neq(hq[1 + (lc2-1)*ldq], 26.0));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtrans)); CUDA_CHECK(cudaFree(dtmp));
}

// ============================================================
// gpu_update_d
//
// d=[10,20,30,40], e=[1,2,3,4], limits=[2], ndiv=2
// Loop: ii=0 (ndiv-1=1 iterations): n=limits[0]-1=1
//   d[1] -= |e[1]| = 2 → d[1]=18
//   d[2] -= |e[1]| = 2 → d[2]=28
// d[0] and d[3] unchanged.
// ============================================================
static void test_update_d()
{
  printf("\ngpu_update_d:\n");

  double hd[] = {10, 20, 30, 40};
  double he[] = {1,  2,  3,  4};
  int    hl[] = {2};
  double *dd, *de; int *dl;
  CUDA_CHECK(cudaMalloc(&dd, 4*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&de, 4*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dl, 1*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dd, hd, 4*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(de, he, 4*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dl, hl, 1*sizeof(int),    cudaMemcpyHostToDevice));
  gpu_update_d<double>(dd, de, dl, 2, 4, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hd, dd, 4);
  REPORT("double update_d: d[0] unchanged=10", neq(hd[0], 10.0));
  REPORT("double update_d: d[1]-=|e[1]|=18",  neq(hd[1], 18.0));
  REPORT("double update_d: d[2]-=|e[1]|=28",  neq(hd[2], 28.0));
  REPORT("double update_d: d[3] unchanged=40", neq(hd[3], 40.0));
  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de)); CUDA_CHECK(cudaFree(dl));

  // float version
  float hdf[] = {10,20,30,40}, hef[] = {1,2,3,4};
  int   hlf[] = {2};
  float *ddf, *def; int *dlf;
  CUDA_CHECK(cudaMalloc(&ddf, 4*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&def, 4*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dlf, 1*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(ddf, hdf, 4*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(def, hef, 4*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dlf, hlf, 1*sizeof(int),   cudaMemcpyHostToDevice));
  gpu_update_d<float>(ddf, def, dlf, 2, 4, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hdf, ddf, 4);
  REPORT("float  update_d: d[1]-=|e[1]|=18",  neqf(hdf[1], 18.0f));
  REPORT("float  update_d: d[2]-=|e[1]|=28",  neqf(hdf[2], 28.0f));
  CUDA_CHECK(cudaFree(ddf)); CUDA_CHECK(cudaFree(def)); CUDA_CHECK(cudaFree(dlf));
}

// ============================================================
// gpu_copy_qmat1_to_qmat2  [KNOWN BUG: only copies diagonal]
//
// The kernel computes both i and j from blockIdx.x*blockDim.x+threadIdx.x,
// so only qmat2[i + max_size*i] = qmat1[i + max_size*i] (diagonal) is set.
// All off-diagonal elements remain unchanged.
//
// max_size=4, qmat1=[1..16] (col-major), qmat2=zeros
// Expected: qmat2 diagonal = {1,6,11,16}; off-diagonal = 0.
// ============================================================
static void test_copy_qmat1_to_qmat2()
{
  printf("\ngpu_copy_qmat1_to_qmat2 (verifies diagonal-only copy bug):\n");

  int max_size = 4;
  double h1[16], h2[16];
  for (int k=0; k<16; k++) h1[k] = (double)(k+1);
  for (int k=0; k<16; k++) h2[k] = 0.0;

  double *d1, *d2;
  CUDA_CHECK(cudaMalloc(&d1, 16*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d2, 16*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d1, h1, 16*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d2, h2, 16*sizeof(double), cudaMemcpyHostToDevice));

  gpu_copy_qmat1_to_qmat2<double>(d1, d2, max_size, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(h2, d2, 16);

  // Diagonal: qmat2[i + 4*i] should equal qmat1[i + 4*i]
  REPORT("double copy_qmat (diag): [0,0]=1",  neq(h2[0+4*0],  1.0));
  REPORT("double copy_qmat (diag): [1,1]=6",  neq(h2[1+4*1],  6.0));
  REPORT("double copy_qmat (diag): [2,2]=11", neq(h2[2+4*2], 11.0));
  REPORT("double copy_qmat (diag): [3,3]=16", neq(h2[3+4*3], 16.0));
  // Off-diagonal: must remain 0 (not copied, bug documented)
  REPORT("double copy_qmat (off-diag [1,0]=0)", neq(h2[1+4*0], 0.0));
  REPORT("double copy_qmat (off-diag [0,1]=0)", neq(h2[0+4*1], 0.0));
  REPORT("double copy_qmat (off-diag [2,1]=0)", neq(h2[2+4*1], 0.0));

  CUDA_CHECK(cudaFree(d1)); CUDA_CHECK(cudaFree(d2));
}

// ============================================================
// gpu_fill_array: fill array with a value
// ============================================================
static void test_fill_array()
{
  printf("\ngpu_fill_array:\n");

  // double: 6 elements set to 3.14
  {
    double hval = 3.14, hout[6];
    double *dval, *dout;
    CUDA_CHECK(cudaMalloc(&dval, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dout, 6*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dval, &hval, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, 6*sizeof(double)));
    gpu_fill_array<double>(dout, dval, 6, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_read_n(hout, dout, 6);
    bool ok = true;
    for (int k=0; k<6; k++) ok = ok && neq(hout[k], 3.14);
    REPORT("double fill_array: all 6 elements == 3.14", ok);
    CUDA_CHECK(cudaFree(dval)); CUDA_CHECK(cudaFree(dout));
  }

  // float: 4 elements set to 2.5f
  {
    float hval = 2.5f, hout[4];
    float *dval, *dout;
    CUDA_CHECK(cudaMalloc(&dval, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, 4*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dval, &hval, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, 4*sizeof(float)));
    gpu_fill_array<float>(dout, dval, 4, 1, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_read_n(hout, dout, 4);
    bool ok = true;
    for (int k=0; k<4; k++) ok = ok && (fabsf(hout[k]-2.5f) < 1e-6f);
    REPORT("float  fill_array: all 4 elements == 2.5", ok);
    CUDA_CHECK(cudaFree(dval)); CUDA_CHECK(cudaFree(dout));
  }
}

// ============================================================
// gpu_copy_qtmp1_to_qtmp1_tmp: 2D copy qtmp1 → qtmp1_tmp
// gemm_dim_k=3, gemm_dim_l=2: qtmp1[i + 3*j] → qtmp1_tmp[i + 3*j]
// ============================================================
static void test_copy_qtmp1_to_qtmp1_tmp()
{
  printf("\ngpu_copy_qtmp1_to_qtmp1_tmp:\n");

  int k=3, l=2;
  double hsrc[6]={1,2,3,4,5,6}, hdst[6]={0};
  double *dsrc, *ddst;
  CUDA_CHECK(cudaMalloc(&dsrc, 6*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&ddst, 6*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dsrc, hsrc, 6*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ddst, 0, 6*sizeof(double)));
  gpu_copy_qtmp1_to_qtmp1_tmp<double>(dsrc, ddst, k, l, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hdst, ddst, 6);
  bool ok = true;
  for (int n=0; n<6; n++) ok = ok && neq(hdst[n], (double)(n+1));
  REPORT("double copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
  CUDA_CHECK(cudaFree(dsrc)); CUDA_CHECK(cudaFree(ddst));

  float hsrcf[6]={1,2,3,4,5,6}, hdstf[6]={0};
  float *dsrcf, *ddstf;
  CUDA_CHECK(cudaMalloc(&dsrcf, 6*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ddstf, 6*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dsrcf, hsrcf, 6*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ddstf, 0, 6*sizeof(float)));
  gpu_copy_qtmp1_to_qtmp1_tmp<float>(dsrcf, ddstf, k, l, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hdstf, ddstf, 6);
  ok = true;
  for (int n=0; n<6; n++) ok = ok && (fabsf(hdstf[n] - (float)(n+1)) < 1e-6f);
  REPORT("float  copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
  CUDA_CHECK(cudaFree(dsrcf)); CUDA_CHECK(cudaFree(ddstf));
}

// ============================================================
// gpu_compute_nnzl_nnzu_val_part1 and part2
//
// na1=4, na=4, npc_n=1, np=1 (np_c=0), np_rem=0
// p_col=[0,0,0,0], idx1=[1,2,3,4], coltyp=[1,3,2,1]
//
// After part1: nnzu_val=[1,0,1,1], nnzl_val=[0,1,1,0]
// After part2 (sequential count): nnzu_val=[1,0,2,3], nnzl_val=[0,1,2,0]
// ============================================================
static void test_compute_nnzl_nnzu()
{
  printf("\ngpu_compute_nnzl_nnzu_val_part1+2:\n");

  int na1=4, na=4, npc_n=1, np_rem=0;
  int h_pcol[4] = {0,0,0,0};
  int h_idx1[4] = {1,2,3,4};
  int h_coltyp[4] = {1,3,2,1};
  int h_nnzu[4] = {0}, h_nnzl[4] = {0};

  int *d_pcol, *d_idx1, *d_coltyp, *d_nnzu, *d_nnzl;
  CUDA_CHECK(cudaMalloc(&d_pcol,   4*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_idx1,   4*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_coltyp, 4*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_nnzu,   4*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_nnzl,   4*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_pcol,   h_pcol,   4*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_idx1,   h_idx1,   4*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_coltyp, h_coltyp, 4*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_nnzu, 0, 4*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_nnzl, 0, 4*sizeof(int)));

  // np=1 → np_c=np-1=0
  gpu_compute_nnzl_nnzu_val_part1(d_pcol, d_idx1, d_coltyp, d_nnzu, d_nnzl,
                                  na, na1, np_rem, npc_n, 0, 0, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
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
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(h_nnzu, d_nnzu, 4);
  dev_read_n(h_nnzl, d_nnzl, 4);
  // Sequential cumulative counts (0s stay 0, 1s become running count)
  REPORT("part2: nnzu_val[0]=1",  h_nnzu[0] == 1);
  REPORT("part2: nnzu_val[1]=0",  h_nnzu[1] == 0);
  REPORT("part2: nnzu_val[2]=2",  h_nnzu[2] == 2);
  REPORT("part2: nnzu_val[3]=3",  h_nnzu[3] == 3);
  REPORT("part2: nnzl_val[0]=0",  h_nnzl[0] == 0);
  REPORT("part2: nnzl_val[1]=1",  h_nnzl[1] == 1);
  REPORT("part2: nnzl_val[2]=2",  h_nnzl[2] == 2);
  REPORT("part2: nnzl_val[3]=0",  h_nnzl[3] == 0);

  CUDA_CHECK(cudaFree(d_pcol));   CUDA_CHECK(cudaFree(d_idx1));
  CUDA_CHECK(cudaFree(d_coltyp)); CUDA_CHECK(cudaFree(d_nnzu));
  CUDA_CHECK(cudaFree(d_nnzl));
}

#ifdef WANT_HALF_PRECISION_REAL
// ============================================================
// gpu_transform_one_column<half_real>: c = alpha*a + beta*b
// a=[1,2,3,4], b=[5,6,7,8], alpha=2, beta=3 → c=[17,22,27,32]
// ============================================================
static void test_transform_one_column_half()
{
  printf("\ngpu_transform_one_column<half_real>:\n");

  float fa[4]={1,2,3,4}, fb[4]={5,6,7,8};
  __half ha[4], hb[4], halpha, hbeta;
  for (int i=0; i<4; i++) ha[i] = __float2half(fa[i]);
  for (int i=0; i<4; i++) hb[i] = __float2half(fb[i]);
  halpha = __float2half(2.0f);
  hbeta  = __float2half(3.0f);

  __half *da, *db, *dc, *dalpha, *dbeta;
  CUDA_CHECK(cudaMalloc(&da,     4*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&db,     4*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dc,     4*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dalpha, sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dbeta,  sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(da,     ha,     4*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db,     hb,     4*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dalpha, &halpha, sizeof(__half),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbeta,  &hbeta,  sizeof(__half),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dc, 0, 4*sizeof(__half)));
  gpu_transform_one_column<half_real>(da, db, dc, dalpha, dbeta, 4, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  __half hc[4];
  dev_read_n(hc, dc, 4);
  REPORT("half transform_one_col: c[0]~17", neqh(hc[0], __float2half(17.0f)));
  REPORT("half transform_one_col: c[1]~22", neqh(hc[1], __float2half(22.0f)));
  REPORT("half transform_one_col: c[2]~27", neqh(hc[2], __float2half(27.0f)));
  REPORT("half transform_one_col: c[3]~32", neqh(hc[3], __float2half(32.0f)));
  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
  CUDA_CHECK(cudaFree(dalpha)); CUDA_CHECK(cudaFree(dbeta));
}

// ============================================================
// gpu_transform_two_columns<half_real>
// col1=[1,2], col2=[3,4], qtrans={2,4,3,5}, ldq=4, l_rows=2
// col1_new=[14,20], col2_new=[18,26]
// ============================================================
static void test_transform_two_columns_half()
{
  printf("\ngpu_transform_two_columns<half_real>:\n");

  int ldq=4, l_rows=2, l_rqs=1, l_rqe=2, lc1=1, lc2=2;

  float fq[4*2]     = {1,2,0,0, 3,4,0,0};
  float fqtrans[4]  = {2,4,3,5};
  __half hq[4*2], hqtrans[4];
  for (int i=0; i<8; i++) hq[i]      = __float2half(fq[i]);
  for (int i=0; i<4; i++) hqtrans[i] = __float2half(fqtrans[i]);

  __half *dq, *dqtrans, *dtmp;
  CUDA_CHECK(cudaMalloc(&dq,      4*2*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dqtrans, 4*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dtmp,    4*sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(dq,      hq,      4*2*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dqtrans, hqtrans, 4*sizeof(__half),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dtmp, 0, 4*sizeof(__half)));

  gpu_transform_two_columns<half_real>(dq, dqtrans, dtmp, ldq, l_rows, l_rqs, l_rqe, lc1, lc2, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  dev_read_n(hq, dq, 4*2);
  REPORT("half transform_two_cols: q_col1[0]~14", neqh(hq[0 + (lc1-1)*ldq], __float2half(14.0f)));
  REPORT("half transform_two_cols: q_col1[1]~20", neqh(hq[1 + (lc1-1)*ldq], __float2half(20.0f)));
  REPORT("half transform_two_cols: q_col2[0]~18", neqh(hq[0 + (lc2-1)*ldq], __float2half(18.0f)));
  REPORT("half transform_two_cols: q_col2[1]~26", neqh(hq[1 + (lc2-1)*ldq], __float2half(26.0f)));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtrans)); CUDA_CHECK(cudaFree(dtmp));
}

// ============================================================
// gpu_copy_qmat1_to_qmat2<half_real>: diagonal-only copy
// max_size=4, qmat1 values 1..16, qmat2 zeros → diagonal {1,6,11,16}
// ============================================================
static void test_copy_qmat1_to_qmat2_half()
{
  printf("\ngpu_copy_qmat1_to_qmat2<half_real>:\n");

  int max_size = 4;
  __half h1[16], h2[16];
  for (int k=0; k<16; k++) h1[k] = __float2half((float)(k+1));
  for (int k=0; k<16; k++) h2[k] = __float2half(0.0f);

  __half *d1, *d2;
  CUDA_CHECK(cudaMalloc(&d1, 16*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d2, 16*sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d1, h1, 16*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d2, h2, 16*sizeof(__half), cudaMemcpyHostToDevice));

  gpu_copy_qmat1_to_qmat2<half_real>(d1, d2, max_size, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(h2, d2, 16);

  REPORT("half copy_qmat (diag): [0,0]~1",  neqh(h2[0+4*0], __float2half(1.0f)));
  REPORT("half copy_qmat (diag): [1,1]~6",  neqh(h2[1+4*1], __float2half(6.0f)));
  REPORT("half copy_qmat (diag): [2,2]~11", neqh(h2[2+4*2], __float2half(11.0f)));
  REPORT("half copy_qmat (diag): [3,3]~16", neqh(h2[3+4*3], __float2half(16.0f)));
  REPORT("half copy_qmat (off-diag [1,0]=0)", neqh(h2[1+4*0], __float2half(0.0f)));

  CUDA_CHECK(cudaFree(d1)); CUDA_CHECK(cudaFree(d2));
}

// ============================================================
// gpu_fill_array<half_real>: fill 4 elements with 2.5
// ============================================================
static void test_fill_array_half()
{
  printf("\ngpu_fill_array<half_real>:\n");

  __half hval = __float2half(2.5f), hout[4];
  __half *dval, *dout;
  CUDA_CHECK(cudaMalloc(&dval, sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dout, 4*sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(dval, &hval, sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dout, 0, 4*sizeof(__half)));
  gpu_fill_array<half_real>(dout, dval, 4, 1, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hout, dout, 4);
  bool ok = true;
  for (int k=0; k<4; k++) ok = ok && neqh(hout[k], __float2half(2.5f));
  REPORT("half fill_array: all 4 elements ~ 2.5", ok);
  CUDA_CHECK(cudaFree(dval)); CUDA_CHECK(cudaFree(dout));
}

// ============================================================
// gpu_copy_qtmp1_to_qtmp1_tmp<half_real>: copy 6 elements
// ============================================================
static void test_copy_qtmp1_to_qtmp1_tmp_half()
{
  printf("\ngpu_copy_qtmp1_to_qtmp1_tmp<half_real>:\n");

  int k=3, l=2;
  __half hsrc[6], hdst[6];
  for (int n=0; n<6; n++) hsrc[n] = __float2half((float)(n+1));
  for (int n=0; n<6; n++) hdst[n] = __float2half(0.0f);

  __half *dsrc, *ddst;
  CUDA_CHECK(cudaMalloc(&dsrc, 6*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&ddst, 6*sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(dsrc, hsrc, 6*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ddst, 0, 6*sizeof(__half)));
  gpu_copy_qtmp1_to_qtmp1_tmp<half_real>(dsrc, ddst, k, l, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  dev_read_n(hdst, ddst, 6);
  bool ok = true;
  for (int n=0; n<6; n++) ok = ok && neqh(hdst[n], __float2half((float)(n+1)));
  REPORT("half copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
  CUDA_CHECK(cudaFree(dsrc)); CUDA_CHECK(cudaFree(ddst));
}
#endif /* WANT_HALF_PRECISION_REAL */

// ============================================================
int main(void)
{
  printf("=== Unit tests for solve_tridi GPU kernels (CUDA) ===\n");

  test_transform_one_column();
  test_transform_two_columns();
  test_update_d();
  test_copy_qmat1_to_qmat2();
  test_fill_array();
  test_copy_qtmp1_to_qtmp1_tmp();
  test_compute_nnzl_nnzu();

#ifdef WANT_HALF_PRECISION_REAL
  printf("\n--- half precision (WANT_HALF_PRECISION_REAL) ---\n");
  test_transform_one_column_half();
  test_transform_two_columns_half();
  test_copy_qmat1_to_qmat2_half();
  test_fill_array_half();
  test_copy_qtmp1_to_qtmp1_tmp_half();
#endif

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
