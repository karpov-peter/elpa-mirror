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

// Unit tests for kernels in gpu_solve_tridi_single_problem.h (CUDA backend).
//
// Kernel 1: gpu_check_monotony<T>
//   Single-threaded insertion sort on d[0..nlen-1].  Correspondingly permutes
//   columns of q (col-major, leading dimension ldq).  qtmp[nlen] is scratch.
//   Key detail: the comparison threshold is stored as 'double dtmp' even when
//   T=float — a deliberate widening that is exact for all float values.
//
// Kernel 2: gpu_construct_full_from_tridi_matrix<T>
//   Fills a symmetric tridiagonal matrix into q (col-major, ldq) from
//   d[0..nlen-1] (diagonal) and e[0..nlen-2] (sub/super-diagonal).
//   q must be pre-zeroed by the caller.

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
#include "../../../../src/solve_tridi/GPU/gpu_solve_tridi_single_problem.h"

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

static bool deq (double a, double b, double tol=1e-10) { return fabs (a-b) < tol; }
static bool feq (float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }

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

template <typename T>
static T *dev_alloc_upload(const T *h, int n) {
  T *d;
  CUDA_CHECK(cudaMalloc(&d, n*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
static T *dev_alloc_zero(int n) {
  T *d;
  CUDA_CHECK(cudaMalloc(&d, n*sizeof(T)));
  CUDA_CHECK(cudaMemset(d, 0, n*sizeof(T)));
  return d;
}


// ============================================================
// gpu_check_monotony — insertion sort of (d, q columns)
//
// The algorithm is single-threaded and runs in-place:
//   for i = 0..nlen-2:
//     if d[i+1] < d[i]:
//       save col i+1 of q into qtmp
//       shift cols and d values right until the insertion point
//       insert d value and column q at position j+1
//
// Note the hardcoded 'double dtmp = d[i+1]' even for T=float — a widening
// that is exact for float values.  The sort is strictly ascending (ties
// are NOT exchanged, so the sort is stable with respect to equal keys).
// ============================================================

// ── Test 1: already sorted — kernel should be a no-op ─────────────────
static void test_check_monotony_already_sorted()
{
  printf("\ngpu_check_monotony — already sorted (no-op):\n");

  // d = [1,2,3,4], q = 4x4 identity (col-major, ldq=4)
  const int nlen = 4, ldq = 4;
  double hd[4] = {1.0, 2.0, 3.0, 4.0};
  double hq[16], hqtmp[4] = {0};
  for (int i=0; i<16; i++) hq[i] = 0.0;
  for (int i=0; i<nlen; i++) hq[i + ldq*i] = 1.0;  // identity

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rd[4], rq[16];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("already_sorted: d[0..3] unchanged: 1,2,3,4",
         deq(rd[0],1) && deq(rd[1],2) && deq(rd[2],3) && deq(rd[3],4));
  // diagonal of identity untouched
  REPORT("already_sorted: q diagonal = 1",
         deq(rq[0+ldq*0],1) && deq(rq[1+ldq*1],1) &&
         deq(rq[2+ldq*2],1) && deq(rq[3+ldq*3],1));
  // off-diagonal still 0
  REPORT("already_sorted: q off-diagonal = 0",
         deq(rq[1+ldq*0],0) && deq(rq[0+ldq*1],0) && deq(rq[3+ldq*2],0));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 2: single adjacent swap ─────────────────────────────────────
//
// d = [1,3,2], q = 3x3 identity.
// Only d[1] and d[2] need to be exchanged.
//
// After sort: d=[1,2,3], q[:,1]=e2=[0,0,1], q[:,2]=e1=[0,1,0]
static void test_check_monotony_single_swap()
{
  printf("\ngpu_check_monotony — single adjacent swap:\n");

  const int nlen = 3, ldq = 3;
  double hd[3]   = {1.0, 3.0, 2.0};
  double hq[9]   = {0};
  double hqtmp[3]= {0};
  for (int i=0; i<nlen; i++) hq[i + ldq*i] = 1.0;  // identity

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rd[3], rq[9];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("single_swap: d sorted to [1,2,3]",
         deq(rd[0],1) && deq(rd[1],2) && deq(rd[2],3));
  // Column 0 of q: unchanged = e0 = [1,0,0]
  REPORT("single_swap: q[:,0] = e0 (unchanged)",
         deq(rq[0+ldq*0],1) && deq(rq[1+ldq*0],0) && deq(rq[2+ldq*0],0));
  // Column 1 of q: originally e2 (eigenvalue 2 → old column 2)
  REPORT("single_swap: q[:,1] = e2 (old col 2, eigenvalue 2)",
         deq(rq[0+ldq*1],0) && deq(rq[1+ldq*1],0) && deq(rq[2+ldq*1],1));
  // Column 2 of q: originally e1 (eigenvalue 3 → old column 1)
  REPORT("single_swap: q[:,2] = e1 (old col 1, eigenvalue 3)",
         deq(rq[0+ldq*2],0) && deq(rq[1+ldq*2],1) && deq(rq[2+ldq*2],0));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 3: fully reverse sorted ────────────────────────────────────
//
// d = [4,3,2,1], q = 4x4 identity.
// After sort: d=[1,2,3,4], column i of q = e_{nlen-1-i} (anti-diagonal).
static void test_check_monotony_reverse_sorted()
{
  printf("\ngpu_check_monotony — fully reverse sorted:\n");

  const int nlen = 4, ldq = 4;
  double hd[4]   = {4.0, 3.0, 2.0, 1.0};
  double hq[16]  = {0};
  double hqtmp[4]= {0};
  for (int i=0; i<nlen; i++) hq[i + ldq*i] = 1.0;

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rd[4], rq[16];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("reverse_sorted: d sorted to [1,2,3,4]",
         deq(rd[0],1) && deq(rd[1],2) && deq(rd[2],3) && deq(rd[3],4));
  // Column i of q should be e_{nlen-1-i}:
  //   q[:,0] = e3 → q[3+4*0]=1, all others 0
  //   q[:,1] = e2 → q[2+4*1]=1
  //   q[:,2] = e1 → q[1+4*2]=1
  //   q[:,3] = e0 → q[0+4*3]=1
  REPORT("reverse_sorted: q[:,0] = e3", deq(rq[3+ldq*0],1) && deq(rq[0+ldq*0],0));
  REPORT("reverse_sorted: q[:,1] = e2", deq(rq[2+ldq*1],1) && deq(rq[3+ldq*1],0));
  REPORT("reverse_sorted: q[:,2] = e1", deq(rq[1+ldq*2],1) && deq(rq[0+ldq*2],0));
  REPORT("reverse_sorted: q[:,3] = e0", deq(rq[0+ldq*3],1) && deq(rq[1+ldq*3],0));
  // Verify some zeros on the old diagonal positions
  REPORT("reverse_sorted: old diagonal entries zero",
         deq(rq[0+ldq*0],0) && deq(rq[1+ldq*1],0) &&
         deq(rq[2+ldq*2],0) && deq(rq[3+ldq*3],0));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 4: non-trivial eigenvectors ─────────────────────────────────
//
// d = [5,1,3], q = [[1,4,7],[2,5,8],[3,6,9]] (col-major).
//   eigenvalue 5 → col 0 = [1,2,3]
//   eigenvalue 1 → col 1 = [4,5,6]
//   eigenvalue 3 → col 2 = [7,8,9]
//
// After sort: d=[1,3,5], q columns reordered as [col1, col2, col0]:
//   col 0 = [4,5,6]
//   col 1 = [7,8,9]
//   col 2 = [1,2,3]
static void test_check_monotony_nontrivial_eigenvecs()
{
  printf("\ngpu_check_monotony — non-trivial eigenvectors:\n");

  const int nlen = 3, ldq = 3;
  // col-major layout: [col0=[1,2,3], col1=[4,5,6], col2=[7,8,9]]
  double hd[3]   = {5.0, 1.0, 3.0};
  double hq[9]   = {1,2,3, 4,5,6, 7,8,9};
  double hqtmp[3]= {0};

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rd[3], rq[9];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("nontrivial: d sorted to [1,3,5]",
         deq(rd[0],1) && deq(rd[1],3) && deq(rd[2],5));
  REPORT("nontrivial: q[:,0]=[4,5,6] (eigenvalue 1)",
         deq(rq[0+ldq*0],4) && deq(rq[1+ldq*0],5) && deq(rq[2+ldq*0],6));
  REPORT("nontrivial: q[:,1]=[7,8,9] (eigenvalue 3)",
         deq(rq[0+ldq*1],7) && deq(rq[1+ldq*1],8) && deq(rq[2+ldq*1],9));
  REPORT("nontrivial: q[:,2]=[1,2,3] (eigenvalue 5)",
         deq(rq[0+ldq*2],1) && deq(rq[1+ldq*2],2) && deq(rq[2+ldq*2],3));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 5: ldq > nlen (padded leading dimension) ─────────────────────
//
// nlen=2, ldq=3.  Only the first nlen rows of each column are active.
// d = [2,1], col0=[10,20,0], col1=[30,40,0]
// After sort: d=[1,2], col0=[30,40], col1=[10,20]
static void test_check_monotony_padded_ldq()
{
  printf("\ngpu_check_monotony — ldq > nlen:\n");

  const int nlen = 2, ldq = 3;
  double hd[2]   = {2.0, 1.0};
  // col-major, 2 cols of width ldq=3
  double hq[6]   = {10, 20, 0,   30, 40, 0};
  double hqtmp[2]= {0};

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rd[2], rq[6];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("padded_ldq: d sorted to [1,2]", deq(rd[0],1) && deq(rd[1],2));
  REPORT("padded_ldq: q[:,0]=[30,40] (eigenvalue 1, ldq stride)",
         deq(rq[0+ldq*0],30) && deq(rq[1+ldq*0],40));
  REPORT("padded_ldq: q[:,1]=[10,20] (eigenvalue 2, ldq stride)",
         deq(rq[0+ldq*1],10) && deq(rq[1+ldq*1],20));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 6: single element (nlen=1) ──────────────────────────────────
//
// No iterations possible (loop runs 0 times).  d and q are unchanged.
static void test_check_monotony_single_element()
{
  printf("\ngpu_check_monotony — single element (nlen=1):\n");

  const int nlen = 1, ldq = 1;
  double hd[1]   = {7.0};
  double hq[1]   = {3.0};
  double hqtmp[1]= {0.0};

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  REPORT("single_element: d[0] unchanged = 7", deq(dev_read(dd), 7.0));
  REPORT("single_element: q[0] unchanged = 3", deq(dev_read(dq), 3.0));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 7: float precision ───────────────────────────────────────────
//
// Same as test 4 but with T=float.  Validates the double-widening path:
//   'double dtmp = d[i+1]' correctly handles float values.
//
// d = [5.0f,1.0f,3.0f], q = [[1,2,3],[4,5,6],[7,8,9]] (col-major)
// After sort: d=[1,3,5], col0=[4,5,6], col1=[7,8,9], col2=[1,2,3]
static void test_check_monotony_float()
{
  printf("\ngpu_check_monotony<float> — non-trivial eigenvectors:\n");

  const int nlen = 3, ldq = 3;
  float hd[3]   = {5.0f, 1.0f, 3.0f};
  float hq[9]   = {1,2,3, 4,5,6, 7,8,9};
  float hqtmp[3]= {0};

  float *dd    = dev_alloc_upload(hd,    nlen);
  float *dq    = dev_alloc_upload(hq,    ldq*nlen);
  float *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<float>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  float rd[3], rq[9];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("float: d sorted to [1,3,5]",
         feq(rd[0],1.0f) && feq(rd[1],3.0f) && feq(rd[2],5.0f));
  REPORT("float: q[:,0]=[4,5,6]",
         feq(rq[0+ldq*0],4) && feq(rq[1+ldq*0],5) && feq(rq[2+ldq*0],6));
  REPORT("float: q[:,1]=[7,8,9]",
         feq(rq[0+ldq*1],7) && feq(rq[1+ldq*1],8) && feq(rq[2+ldq*1],9));
  REPORT("float: q[:,2]=[1,2,3]",
         feq(rq[0+ldq*2],1) && feq(rq[1+ldq*2],2) && feq(rq[2+ldq*2],3));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}

// ── Test 8: duplicate eigenvalues (stable sort) ───────────────────────
//
// d = [2,1,2,1], q = 4x4 identity.
// Insertion sort uses strict '<' so equal keys are not exchanged.
// Analytical result:
//   d=[1,1,2,2], columns permuted: col0=e1, col1=e3, col2=e0, col3=e2
static void test_check_monotony_duplicates()
{
  printf("\ngpu_check_monotony — duplicate eigenvalues (stable):\n");

  const int nlen = 4, ldq = 4;
  double hd[4]   = {2.0, 1.0, 2.0, 1.0};
  double hq[16]  = {0};
  double hqtmp[4]= {0};
  for (int i=0; i<nlen; i++) hq[i + ldq*i] = 1.0;  // identity

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq*nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  gpu_check_monotony<double>(dd, dq, dqtmp, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rd[4], rq[16];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq*nlen);

  REPORT("duplicates: d sorted to [1,1,2,2]",
         deq(rd[0],1) && deq(rd[1],1) && deq(rd[2],2) && deq(rd[3],2));
  // col0=e1: q[1+4*0]=1, all others in col 0 are 0
  REPORT("duplicates: q[:,0] = e1", deq(rq[1+ldq*0],1) && deq(rq[0+ldq*0],0));
  // col1=e3: q[3+4*1]=1
  REPORT("duplicates: q[:,1] = e3", deq(rq[3+ldq*1],1) && deq(rq[0+ldq*1],0));
  // col2=e0: q[0+4*2]=1
  REPORT("duplicates: q[:,2] = e0", deq(rq[0+ldq*2],1) && deq(rq[1+ldq*2],0));
  // col3=e2: q[2+4*3]=1
  REPORT("duplicates: q[:,3] = e2", deq(rq[2+ldq*3],1) && deq(rq[0+ldq*3],0));

  CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dqtmp));
}


// ============================================================
// gpu_construct_full_from_tridi_matrix
//
// Sets q (col-major, ldq) from d (diagonal) and e (sub/super-diagonal).
// q[i + ldq*i]     = d[i]   for i = 0..nlen-1
// q[i+1 + ldq*i]   = e[i]   for i = 0..nlen-2
// q[i + ldq*(i+1)] = e[i]   for i = 0..nlen-2
// All other elements remain at their pre-zeroed values.
// ============================================================

// ── Test 1: nlen=1 (trivial, only diagonal) ───────────────────────────
static void test_construct_tridi_n1()
{
  printf("\ngpu_construct_full_from_tridi_matrix — nlen=1:\n");

  const int nlen = 1, ldq = 1;
  double hd[1] = {7.0};
  // e is empty; allocate a 1-element buffer as a placeholder (kernel uses i<nlen-1)
  double he[1] = {0.0};

  double *dq = dev_alloc_zero<double>(ldq*nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, 1);   // not accessed by kernel when nlen=1

  gpu_construct_full_from_tridi_matrix<double>(dq, dd, de, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  REPORT("n1: q[0,0] = d[0] = 7", deq(dev_read(dq), 7.0));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de));
}

// ── Test 2: nlen=2 ───────────────────────────────────────────────────
//
// d=[3,5], e=[2]
// Expected 2x2 matrix (col-major):
//   q[0+2*0]=3, q[1+2*0]=2
//   q[0+2*1]=2, q[1+2*1]=5
static void test_construct_tridi_n2()
{
  printf("\ngpu_construct_full_from_tridi_matrix — nlen=2:\n");

  const int nlen = 2, ldq = 2;
  double hd[2] = {3.0, 5.0};
  double he[1] = {2.0};

  double *dq = dev_alloc_zero<double>(ldq*nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen-1);

  gpu_construct_full_from_tridi_matrix<double>(dq, dd, de, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rq[4]; dev_read_n(rq, dq, ldq*nlen);

  REPORT("n2: q[0,0] = d[0] = 3",   deq(rq[0+ldq*0], 3.0));
  REPORT("n2: q[1,0] = e[0] = 2",   deq(rq[1+ldq*0], 2.0));
  REPORT("n2: q[0,1] = e[0] = 2",   deq(rq[0+ldq*1], 2.0));
  REPORT("n2: q[1,1] = d[1] = 5",   deq(rq[1+ldq*1], 5.0));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de));
}

// ── Test 3: nlen=4, double ────────────────────────────────────────────
//
// d=[1,2,3,4], e=[5,6,7]
// Expected 4x4 matrix:
//   [ 1  5  0  0 ]
//   [ 5  2  6  0 ]
//   [ 0  6  3  7 ]
//   [ 0  0  7  4 ]
static void test_construct_tridi_n4_double()
{
  printf("\ngpu_construct_full_from_tridi_matrix<double> — nlen=4:\n");

  const int nlen = 4, ldq = 4;
  double hd[4] = {1.0, 2.0, 3.0, 4.0};
  double he[3] = {5.0, 6.0, 7.0};

  double *dq = dev_alloc_zero<double>(ldq*nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen-1);

  gpu_construct_full_from_tridi_matrix<double>(dq, dd, de, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rq[16]; dev_read_n(rq, dq, ldq*nlen);

  // Diagonal
  REPORT("n4d: q[0,0]=1", deq(rq[0+ldq*0], 1.0));
  REPORT("n4d: q[1,1]=2", deq(rq[1+ldq*1], 2.0));
  REPORT("n4d: q[2,2]=3", deq(rq[2+ldq*2], 3.0));
  REPORT("n4d: q[3,3]=4", deq(rq[3+ldq*3], 4.0));
  // Sub-diagonal
  REPORT("n4d: q[1,0]=e[0]=5", deq(rq[1+ldq*0], 5.0));
  REPORT("n4d: q[2,1]=e[1]=6", deq(rq[2+ldq*1], 6.0));
  REPORT("n4d: q[3,2]=e[2]=7", deq(rq[3+ldq*2], 7.0));
  // Super-diagonal (symmetric)
  REPORT("n4d: q[0,1]=e[0]=5", deq(rq[0+ldq*1], 5.0));
  REPORT("n4d: q[1,2]=e[1]=6", deq(rq[1+ldq*2], 6.0));
  REPORT("n4d: q[2,3]=e[2]=7", deq(rq[2+ldq*3], 7.0));
  // Off-tridiagonal elements must be zero
  REPORT("n4d: q[0,2]=0 (off-tridiag)", deq(rq[0+ldq*2], 0.0));
  REPORT("n4d: q[0,3]=0 (off-tridiag)", deq(rq[0+ldq*3], 0.0));
  REPORT("n4d: q[2,0]=0 (off-tridiag)", deq(rq[2+ldq*0], 0.0));
  REPORT("n4d: q[3,0]=0 (off-tridiag)", deq(rq[3+ldq*0], 0.0));
  REPORT("n4d: q[3,1]=0 (off-tridiag)", deq(rq[3+ldq*1], 0.0));
  REPORT("n4d: q[1,3]=0 (off-tridiag)", deq(rq[1+ldq*3], 0.0));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de));
}

// ── Test 4: nlen=3, float ─────────────────────────────────────────────
//
// d=[2.0f,4.0f,6.0f], e=[1.0f,3.0f]
// Expected 3x3 matrix:
//   [ 2  1  0 ]
//   [ 1  4  3 ]
//   [ 0  3  6 ]
static void test_construct_tridi_n3_float()
{
  printf("\ngpu_construct_full_from_tridi_matrix<float> — nlen=3:\n");

  const int nlen = 3, ldq = 3;
  float hd[3] = {2.0f, 4.0f, 6.0f};
  float he[2] = {1.0f, 3.0f};

  float *dq = dev_alloc_zero<float>(ldq*nlen);
  float *dd = dev_alloc_upload(hd, nlen);
  float *de = dev_alloc_upload(he, nlen-1);

  gpu_construct_full_from_tridi_matrix<float>(dq, dd, de, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  float rq[9]; dev_read_n(rq, dq, ldq*nlen);

  REPORT("n3f: q[0,0]=d[0]=2", feq(rq[0+ldq*0], 2.0f));
  REPORT("n3f: q[1,1]=d[1]=4", feq(rq[1+ldq*1], 4.0f));
  REPORT("n3f: q[2,2]=d[2]=6", feq(rq[2+ldq*2], 6.0f));
  REPORT("n3f: q[1,0]=e[0]=1", feq(rq[1+ldq*0], 1.0f));
  REPORT("n3f: q[0,1]=e[0]=1", feq(rq[0+ldq*1], 1.0f));
  REPORT("n3f: q[2,1]=e[1]=3", feq(rq[2+ldq*1], 3.0f));
  REPORT("n3f: q[1,2]=e[1]=3", feq(rq[1+ldq*2], 3.0f));
  REPORT("n3f: q[0,2]=0 (off-tridiag)", feq(rq[0+ldq*2], 0.0f));
  REPORT("n3f: q[2,0]=0 (off-tridiag)", feq(rq[2+ldq*0], 0.0f));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de));
}

// ── Test 5: ldq > nlen (padded allocation) ────────────────────────────
//
// nlen=3, ldq=5, d=[10,20,30], e=[1,2]
// Verifies that off-row padding bytes (row 3 and 4 within each column)
// are not touched, and that the stride ldq is used correctly.
static void test_construct_tridi_padded_ldq()
{
  printf("\ngpu_construct_full_from_tridi_matrix — padded ldq (ldq=5, nlen=3):\n");

  const int nlen = 3, ldq = 5;
  double hd[3] = {10.0, 20.0, 30.0};
  double he[2] = {1.0,  2.0};

  // allocate ldq*nlen = 15 elements, pre-zeroed
  double *dq = dev_alloc_zero<double>(ldq*nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen-1);

  gpu_construct_full_from_tridi_matrix<double>(dq, dd, de, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rq[15]; dev_read_n(rq, dq, ldq*nlen);

  // Diagonal
  REPORT("padded: q[0,0]=d[0]=10", deq(rq[0+ldq*0], 10.0));
  REPORT("padded: q[1,1]=d[1]=20", deq(rq[1+ldq*1], 20.0));
  REPORT("padded: q[2,2]=d[2]=30", deq(rq[2+ldq*2], 30.0));
  // Off-diagonal
  REPORT("padded: q[1,0]=e[0]=1",  deq(rq[1+ldq*0],  1.0));
  REPORT("padded: q[0,1]=e[0]=1",  deq(rq[0+ldq*1],  1.0));
  REPORT("padded: q[2,1]=e[1]=2",  deq(rq[2+ldq*1],  2.0));
  REPORT("padded: q[1,2]=e[1]=2",  deq(rq[1+ldq*2],  2.0));
  // Padding rows (row 3 and 4 within columns) must stay zero
  REPORT("padded: q[3,0]=0 (padding row)", deq(rq[3+ldq*0], 0.0));
  REPORT("padded: q[4,0]=0 (padding row)", deq(rq[4+ldq*0], 0.0));
  REPORT("padded: q[3,1]=0 (padding row)", deq(rq[3+ldq*1], 0.0));
  REPORT("padded: q[3,2]=0 (padding row)", deq(rq[3+ldq*2], 0.0));
  // Off-tridiagonal zeros in the active region
  REPORT("padded: q[0,2]=0 (off-tridiag)", deq(rq[0+ldq*2], 0.0));
  REPORT("padded: q[2,0]=0 (off-tridiag)", deq(rq[2+ldq*0], 0.0));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de));
}

// ── Test 6: symmetry check ────────────────────────────────────────────
//
// Verify that q is exactly symmetric: q[i,j] == q[j,i] for all i,j.
// Uses d=[7,3,1,9,5], e=[2,4,6,8] (nlen=5).
static void test_construct_tridi_symmetry()
{
  printf("\ngpu_construct_full_from_tridi_matrix — symmetry check (nlen=5):\n");

  const int nlen = 5, ldq = 5;
  double hd[5] = {7.0, 3.0, 1.0, 9.0, 5.0};
  double he[4] = {2.0, 4.0, 6.0, 8.0};

  double *dq = dev_alloc_zero<double>(ldq*nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen-1);

  gpu_construct_full_from_tridi_matrix<double>(dq, dd, de, nlen, ldq, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double rq[25]; dev_read_n(rq, dq, ldq*nlen);

  bool sym_ok = true;
  for (int i=0; i<nlen && sym_ok; i++)
    for (int j=0; j<nlen && sym_ok; j++)
      if (!deq(rq[i+ldq*j], rq[j+ldq*i])) sym_ok = false;

  REPORT("symmetry: q[i,j]==q[j,i] for all i,j (nlen=5)", sym_ok);

  // Spot-check a few known values
  REPORT("symmetry: q[0,0]=d[0]=7", deq(rq[0+ldq*0], 7.0));
  REPORT("symmetry: q[1,0]=q[0,1]=e[0]=2",
         deq(rq[1+ldq*0], 2.0) && deq(rq[0+ldq*1], 2.0));
  REPORT("symmetry: q[4,4]=d[4]=5", deq(rq[4+ldq*4], 5.0));
  REPORT("symmetry: q[4,3]=q[3,4]=e[3]=8",
         deq(rq[4+ldq*3], 8.0) && deq(rq[3+ldq*4], 8.0));
  // Non-adjacent off-tridiagonal elements are zero
  REPORT("symmetry: q[0,2]=q[2,0]=0",
         deq(rq[0+ldq*2], 0.0) && deq(rq[2+ldq*0], 0.0));
  REPORT("symmetry: q[0,4]=q[4,0]=0",
         deq(rq[0+ldq*4], 0.0) && deq(rq[4+ldq*0], 0.0));

  CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dd)); CUDA_CHECK(cudaFree(de));
}


// ============================================================
int main(void)
{
  printf("=== Unit tests for gpu_solve_tridi_single_problem.h (CUDA) ===\n");

  printf("\n--- gpu_check_monotony ---\n");
  test_check_monotony_already_sorted();
  test_check_monotony_single_swap();
  test_check_monotony_reverse_sorted();
  test_check_monotony_nontrivial_eigenvecs();
  test_check_monotony_padded_ldq();
  test_check_monotony_single_element();
  test_check_monotony_float();
  test_check_monotony_duplicates();

  printf("\n--- gpu_construct_full_from_tridi_matrix ---\n");
  test_construct_tridi_n1();
  test_construct_tridi_n2();
  test_construct_tridi_n4_double();
  test_construct_tridi_n3_float();
  test_construct_tridi_padded_ldq();
  test_construct_tridi_symmetry();

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
