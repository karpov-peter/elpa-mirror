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
#include <cstring>
#include <sycl/sycl.hpp>
#include <complex>

// ============================================================
// Kernel bodies (copied from sycl_solve_tridi_single_problem.cpp)
// ============================================================

template <typename T>
void gpu_check_monotony_kernel(T *d, T *q, T *qtmp, const int nlen, const int ldq,
                               const sycl::nd_item<1> &it) {
  int j;
  for (int i = 0; i < nlen - 1; i++) {
    if (d[i + 1] < d[i]) {
      double dtmp = d[i + 1];
      for (j = 0; j < nlen; j++) {
        qtmp[j] = q[j + ldq * (i + 1)];
      }
      for (j = i; j >= 0; j--) {
        if (dtmp < d[j]) {
          d[j + 1] = d[j];
          for (int k = 0; k < nlen; k++) {
            q[k + ldq * (j + 1)] = q[k + ldq * j];
          }
        } else {
          break;
        }
      }
      d[j + 1] = dtmp;
      for (int k = 0; k < nlen; k++) {
        q[k + ldq * (j + 1)] = qtmp[k];
      }
    }
  }
}

template <typename T>
void gpu_construct_full_from_tridi_matrix_kernel(T *q, T *d, T *e, const int nlen, const int ldq,
                                                 const sycl::nd_item<1> &it) {
  int i = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);

  if (i >= 0 && i < nlen) {
    q[i + ldq * i] = d[i];
  }
  if (i >= 0 && i < nlen - 1) {
    q[i + 1 + ldq * i] = e[i];
    q[i + ldq * (i + 1)] = e[i];
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
// Self-contained launchers
// ============================================================

template <typename T>
static void launch_check_monotony(T *dd, T *dq, T *dqtmp, int nlen, int ldq) {
  sycl::queue &q = get_queue();
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                 [=](sycl::nd_item<1> it) {
                   gpu_check_monotony_kernel(dd, dq, dqtmp, nlen, ldq, it);
                 }).wait();
}

template <typename T>
static void launch_construct_full(T *dq, T *dd, T *de, int nlen, int ldq) {
  sycl::queue &q = get_queue();
  const int tpb = 64;
  const int blocks = (nlen + tpb - 1) / tpb;
  if (blocks == 0) return;
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * tpb), sycl::range<1>(tpb)),
                 [=](sycl::nd_item<1> it) {
                   gpu_construct_full_from_tridi_matrix_kernel(dq, dd, de, nlen, ldq, it);
                 }).wait();
}

// ============================================================
// Device memory helpers
// ============================================================

template <typename T>
static T dev_read(T *d) {
  sycl::queue &q = get_queue();
  T h;
  q.memcpy(&h, d, sizeof(T)).wait();
  return h;
}

template <typename T>
static void dev_read_n(T *h, T *d, int n) {
  sycl::queue &q = get_queue();
  q.memcpy(h, d, n * sizeof(T)).wait();
}

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
static void dev_free(T *d) {
  sycl::free(d, get_queue());
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
static bool feq(float a,  float b,  float  tol = 1e-5f) { return fabsf(a - b) < tol; }


// ============================================================
// gpu_check_monotony — insertion sort of (d, q columns)
// ============================================================

// ── Test 1: already sorted — kernel should be a no-op ─────────────────
static void test_check_monotony_already_sorted()
{
  printf("\ngpu_check_monotony — already sorted (no-op):\n");

  const int nlen = 4, ldq = 4;
  double hd[4]   = {1.0, 2.0, 3.0, 4.0};
  double hq[16], hqtmp[4] = {0};
  for (int i = 0; i < 16; i++) hq[i] = 0.0;
  for (int i = 0; i < nlen; i++) hq[i + ldq * i] = 1.0;

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  double rd[4], rq[16];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("already_sorted: d[0..3] unchanged: 1,2,3,4",
         deq(rd[0],1) && deq(rd[1],2) && deq(rd[2],3) && deq(rd[3],4));
  REPORT("already_sorted: q diagonal = 1",
         deq(rq[0+ldq*0],1) && deq(rq[1+ldq*1],1) &&
         deq(rq[2+ldq*2],1) && deq(rq[3+ldq*3],1));
  REPORT("already_sorted: q off-diagonal = 0",
         deq(rq[1+ldq*0],0) && deq(rq[0+ldq*1],0) && deq(rq[3+ldq*2],0));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 2: single adjacent swap ─────────────────────────────────────
//
// d = [1,3,2], q = 3x3 identity.
// After sort: d=[1,2,3], q[:,1]=e2=[0,0,1], q[:,2]=e1=[0,1,0]
static void test_check_monotony_single_swap()
{
  printf("\ngpu_check_monotony — single adjacent swap:\n");

  const int nlen = 3, ldq = 3;
  double hd[3]    = {1.0, 3.0, 2.0};
  double hq[9]    = {0};
  double hqtmp[3] = {0};
  for (int i = 0; i < nlen; i++) hq[i + ldq * i] = 1.0;

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  double rd[3], rq[9];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("single_swap: d sorted to [1,2,3]",
         deq(rd[0],1) && deq(rd[1],2) && deq(rd[2],3));
  REPORT("single_swap: q[:,0] = e0 (unchanged)",
         deq(rq[0+ldq*0],1) && deq(rq[1+ldq*0],0) && deq(rq[2+ldq*0],0));
  REPORT("single_swap: q[:,1] = e2 (old col 2, eigenvalue 2)",
         deq(rq[0+ldq*1],0) && deq(rq[1+ldq*1],0) && deq(rq[2+ldq*1],1));
  REPORT("single_swap: q[:,2] = e1 (old col 1, eigenvalue 3)",
         deq(rq[0+ldq*2],0) && deq(rq[1+ldq*2],1) && deq(rq[2+ldq*2],0));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 3: fully reverse sorted ────────────────────────────────────
//
// d = [4,3,2,1], q = 4x4 identity.
// After sort: d=[1,2,3,4], column i of q = e_{nlen-1-i} (anti-diagonal).
static void test_check_monotony_reverse_sorted()
{
  printf("\ngpu_check_monotony — fully reverse sorted:\n");

  const int nlen = 4, ldq = 4;
  double hd[4]    = {4.0, 3.0, 2.0, 1.0};
  double hq[16]   = {0};
  double hqtmp[4] = {0};
  for (int i = 0; i < nlen; i++) hq[i + ldq * i] = 1.0;

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  double rd[4], rq[16];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("reverse_sorted: d sorted to [1,2,3,4]",
         deq(rd[0],1) && deq(rd[1],2) && deq(rd[2],3) && deq(rd[3],4));
  REPORT("reverse_sorted: q[:,0] = e3", deq(rq[3+ldq*0],1) && deq(rq[0+ldq*0],0));
  REPORT("reverse_sorted: q[:,1] = e2", deq(rq[2+ldq*1],1) && deq(rq[3+ldq*1],0));
  REPORT("reverse_sorted: q[:,2] = e1", deq(rq[1+ldq*2],1) && deq(rq[0+ldq*2],0));
  REPORT("reverse_sorted: q[:,3] = e0", deq(rq[0+ldq*3],1) && deq(rq[1+ldq*3],0));
  REPORT("reverse_sorted: old diagonal entries zero",
         deq(rq[0+ldq*0],0) && deq(rq[1+ldq*1],0) &&
         deq(rq[2+ldq*2],0) && deq(rq[3+ldq*3],0));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 4: non-trivial eigenvectors ─────────────────────────────────
//
// d = [5,1,3], q = [[1,4,7],[2,5,8],[3,6,9]] (col-major).
// After sort: d=[1,3,5], col0=[4,5,6], col1=[7,8,9], col2=[1,2,3]
static void test_check_monotony_nontrivial_eigenvecs()
{
  printf("\ngpu_check_monotony — non-trivial eigenvectors:\n");

  const int nlen = 3, ldq = 3;
  double hd[3]    = {5.0, 1.0, 3.0};
  double hq[9]    = {1,2,3, 4,5,6, 7,8,9};
  double hqtmp[3] = {0};

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  double rd[3], rq[9];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("nontrivial: d sorted to [1,3,5]",
         deq(rd[0],1) && deq(rd[1],3) && deq(rd[2],5));
  REPORT("nontrivial: q[:,0]=[4,5,6] (eigenvalue 1)",
         deq(rq[0+ldq*0],4) && deq(rq[1+ldq*0],5) && deq(rq[2+ldq*0],6));
  REPORT("nontrivial: q[:,1]=[7,8,9] (eigenvalue 3)",
         deq(rq[0+ldq*1],7) && deq(rq[1+ldq*1],8) && deq(rq[2+ldq*1],9));
  REPORT("nontrivial: q[:,2]=[1,2,3] (eigenvalue 5)",
         deq(rq[0+ldq*2],1) && deq(rq[1+ldq*2],2) && deq(rq[2+ldq*2],3));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 5: ldq > nlen (padded leading dimension) ─────────────────────
//
// nlen=2, ldq=3.  d=[2,1], col0=[10,20,0], col1=[30,40,0].
// After sort: d=[1,2], col0=[30,40], col1=[10,20].
static void test_check_monotony_padded_ldq()
{
  printf("\ngpu_check_monotony — ldq > nlen:\n");

  const int nlen = 2, ldq = 3;
  double hd[2]    = {2.0, 1.0};
  double hq[6]    = {10, 20, 0,   30, 40, 0};
  double hqtmp[2] = {0};

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  double rd[2], rq[6];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("padded_ldq: d sorted to [1,2]", deq(rd[0],1) && deq(rd[1],2));
  REPORT("padded_ldq: q[:,0]=[30,40] (eigenvalue 1, ldq stride)",
         deq(rq[0+ldq*0],30) && deq(rq[1+ldq*0],40));
  REPORT("padded_ldq: q[:,1]=[10,20] (eigenvalue 2, ldq stride)",
         deq(rq[0+ldq*1],10) && deq(rq[1+ldq*1],20));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 6: single element (nlen=1) ──────────────────────────────────
static void test_check_monotony_single_element()
{
  printf("\ngpu_check_monotony — single element (nlen=1):\n");

  const int nlen = 1, ldq = 1;
  double hd[1]    = {7.0};
  double hq[1]    = {3.0};
  double hqtmp[1] = {0.0};

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  REPORT("single_element: d[0] unchanged = 7", deq(dev_read(dd), 7.0));
  REPORT("single_element: q[0] unchanged = 3", deq(dev_read(dq), 3.0));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 7: float precision ───────────────────────────────────────────
//
// d = [5.0f,1.0f,3.0f], q = [[1,2,3],[4,5,6],[7,8,9]] (col-major).
// After sort: d=[1,3,5], col0=[4,5,6], col1=[7,8,9], col2=[1,2,3]
static void test_check_monotony_float()
{
  printf("\ngpu_check_monotony<float> — non-trivial eigenvectors:\n");

  const int nlen = 3, ldq = 3;
  float hd[3]    = {5.0f, 1.0f, 3.0f};
  float hq[9]    = {1,2,3, 4,5,6, 7,8,9};
  float hqtmp[3] = {0};

  float *dd    = dev_alloc_upload(hd,    nlen);
  float *dq    = dev_alloc_upload(hq,    ldq * nlen);
  float *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  float rd[3], rq[9];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("float: d sorted to [1,3,5]",
         feq(rd[0],1.0f) && feq(rd[1],3.0f) && feq(rd[2],5.0f));
  REPORT("float: q[:,0]=[4,5,6]",
         feq(rq[0+ldq*0],4) && feq(rq[1+ldq*0],5) && feq(rq[2+ldq*0],6));
  REPORT("float: q[:,1]=[7,8,9]",
         feq(rq[0+ldq*1],7) && feq(rq[1+ldq*1],8) && feq(rq[2+ldq*1],9));
  REPORT("float: q[:,2]=[1,2,3]",
         feq(rq[0+ldq*2],1) && feq(rq[1+ldq*2],2) && feq(rq[2+ldq*2],3));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}

// ── Test 8: duplicate eigenvalues (stable sort) ───────────────────────
//
// d = [2,1,2,1], q = 4x4 identity.
// Strict '<' means equal keys are not exchanged.
// Result: d=[1,1,2,2], col0=e1, col1=e3, col2=e0, col3=e2
static void test_check_monotony_duplicates()
{
  printf("\ngpu_check_monotony — duplicate eigenvalues (stable):\n");

  const int nlen = 4, ldq = 4;
  double hd[4]    = {2.0, 1.0, 2.0, 1.0};
  double hq[16]   = {0};
  double hqtmp[4] = {0};
  for (int i = 0; i < nlen; i++) hq[i + ldq * i] = 1.0;

  double *dd    = dev_alloc_upload(hd,    nlen);
  double *dq    = dev_alloc_upload(hq,    ldq * nlen);
  double *dqtmp = dev_alloc_upload(hqtmp, nlen);

  launch_check_monotony(dd, dq, dqtmp, nlen, ldq);

  double rd[4], rq[16];
  dev_read_n(rd, dd, nlen);
  dev_read_n(rq, dq, ldq * nlen);

  REPORT("duplicates: d sorted to [1,1,2,2]",
         deq(rd[0],1) && deq(rd[1],1) && deq(rd[2],2) && deq(rd[3],2));
  REPORT("duplicates: q[:,0] = e1", deq(rq[1+ldq*0],1) && deq(rq[0+ldq*0],0));
  REPORT("duplicates: q[:,1] = e3", deq(rq[3+ldq*1],1) && deq(rq[0+ldq*1],0));
  REPORT("duplicates: q[:,2] = e0", deq(rq[0+ldq*2],1) && deq(rq[1+ldq*2],0));
  REPORT("duplicates: q[:,3] = e2", deq(rq[2+ldq*3],1) && deq(rq[0+ldq*3],0));

  dev_free(dd); dev_free(dq); dev_free(dqtmp);
}


// ============================================================
// gpu_construct_full_from_tridi_matrix
// ============================================================

// ── Test 1: nlen=1 ────────────────────────────────────────────────────
static void test_construct_tridi_n1()
{
  printf("\ngpu_construct_full_from_tridi_matrix — nlen=1:\n");

  const int nlen = 1, ldq = 1;
  double hd[1] = {7.0};
  double he[1] = {0.0};

  double *dq = dev_alloc_zero<double>(ldq * nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, 1);

  launch_construct_full(dq, dd, de, nlen, ldq);

  REPORT("n1: q[0,0] = d[0] = 7", deq(dev_read(dq), 7.0));

  dev_free(dq); dev_free(dd); dev_free(de);
}

// ── Test 2: nlen=2 ────────────────────────────────────────────────────
//
// d=[3,5], e=[2] → q = [[3,2],[2,5]]
static void test_construct_tridi_n2()
{
  printf("\ngpu_construct_full_from_tridi_matrix — nlen=2:\n");

  const int nlen = 2, ldq = 2;
  double hd[2] = {3.0, 5.0};
  double he[1] = {2.0};

  double *dq = dev_alloc_zero<double>(ldq * nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen - 1);

  launch_construct_full(dq, dd, de, nlen, ldq);

  double rq[4]; dev_read_n(rq, dq, ldq * nlen);

  REPORT("n2: q[0,0] = d[0] = 3", deq(rq[0+ldq*0], 3.0));
  REPORT("n2: q[1,0] = e[0] = 2", deq(rq[1+ldq*0], 2.0));
  REPORT("n2: q[0,1] = e[0] = 2", deq(rq[0+ldq*1], 2.0));
  REPORT("n2: q[1,1] = d[1] = 5", deq(rq[1+ldq*1], 5.0));

  dev_free(dq); dev_free(dd); dev_free(de);
}

// ── Test 3: nlen=4, double ────────────────────────────────────────────
//
// d=[1,2,3,4], e=[5,6,7]
// Expected 4x4 symmetric tridiagonal matrix.
static void test_construct_tridi_n4_double()
{
  printf("\ngpu_construct_full_from_tridi_matrix<double> — nlen=4:\n");

  const int nlen = 4, ldq = 4;
  double hd[4] = {1.0, 2.0, 3.0, 4.0};
  double he[3] = {5.0, 6.0, 7.0};

  double *dq = dev_alloc_zero<double>(ldq * nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen - 1);

  launch_construct_full(dq, dd, de, nlen, ldq);

  double rq[16]; dev_read_n(rq, dq, ldq * nlen);

  REPORT("n4d: q[0,0]=1", deq(rq[0+ldq*0], 1.0));
  REPORT("n4d: q[1,1]=2", deq(rq[1+ldq*1], 2.0));
  REPORT("n4d: q[2,2]=3", deq(rq[2+ldq*2], 3.0));
  REPORT("n4d: q[3,3]=4", deq(rq[3+ldq*3], 4.0));
  REPORT("n4d: q[1,0]=e[0]=5", deq(rq[1+ldq*0], 5.0));
  REPORT("n4d: q[2,1]=e[1]=6", deq(rq[2+ldq*1], 6.0));
  REPORT("n4d: q[3,2]=e[2]=7", deq(rq[3+ldq*2], 7.0));
  REPORT("n4d: q[0,1]=e[0]=5", deq(rq[0+ldq*1], 5.0));
  REPORT("n4d: q[1,2]=e[1]=6", deq(rq[1+ldq*2], 6.0));
  REPORT("n4d: q[2,3]=e[2]=7", deq(rq[2+ldq*3], 7.0));
  REPORT("n4d: q[0,2]=0 (off-tridiag)", deq(rq[0+ldq*2], 0.0));
  REPORT("n4d: q[0,3]=0 (off-tridiag)", deq(rq[0+ldq*3], 0.0));
  REPORT("n4d: q[2,0]=0 (off-tridiag)", deq(rq[2+ldq*0], 0.0));
  REPORT("n4d: q[3,0]=0 (off-tridiag)", deq(rq[3+ldq*0], 0.0));
  REPORT("n4d: q[3,1]=0 (off-tridiag)", deq(rq[3+ldq*1], 0.0));
  REPORT("n4d: q[1,3]=0 (off-tridiag)", deq(rq[1+ldq*3], 0.0));

  dev_free(dq); dev_free(dd); dev_free(de);
}

// ── Test 4: nlen=3, float ─────────────────────────────────────────────
//
// d=[2.0f,4.0f,6.0f], e=[1.0f,3.0f]
static void test_construct_tridi_n3_float()
{
  printf("\ngpu_construct_full_from_tridi_matrix<float> — nlen=3:\n");

  const int nlen = 3, ldq = 3;
  float hd[3] = {2.0f, 4.0f, 6.0f};
  float he[2] = {1.0f, 3.0f};

  float *dq = dev_alloc_zero<float>(ldq * nlen);
  float *dd = dev_alloc_upload(hd, nlen);
  float *de = dev_alloc_upload(he, nlen - 1);

  launch_construct_full(dq, dd, de, nlen, ldq);

  float rq[9]; dev_read_n(rq, dq, ldq * nlen);

  REPORT("n3f: q[0,0]=d[0]=2", feq(rq[0+ldq*0], 2.0f));
  REPORT("n3f: q[1,1]=d[1]=4", feq(rq[1+ldq*1], 4.0f));
  REPORT("n3f: q[2,2]=d[2]=6", feq(rq[2+ldq*2], 6.0f));
  REPORT("n3f: q[1,0]=e[0]=1", feq(rq[1+ldq*0], 1.0f));
  REPORT("n3f: q[0,1]=e[0]=1", feq(rq[0+ldq*1], 1.0f));
  REPORT("n3f: q[2,1]=e[1]=3", feq(rq[2+ldq*1], 3.0f));
  REPORT("n3f: q[1,2]=e[1]=3", feq(rq[1+ldq*2], 3.0f));
  REPORT("n3f: q[0,2]=0 (off-tridiag)", feq(rq[0+ldq*2], 0.0f));
  REPORT("n3f: q[2,0]=0 (off-tridiag)", feq(rq[2+ldq*0], 0.0f));

  dev_free(dq); dev_free(dd); dev_free(de);
}

// ── Test 5: ldq > nlen (padded allocation) ────────────────────────────
//
// nlen=3, ldq=5, d=[10,20,30], e=[1,2]
static void test_construct_tridi_padded_ldq()
{
  printf("\ngpu_construct_full_from_tridi_matrix — padded ldq (ldq=5, nlen=3):\n");

  const int nlen = 3, ldq = 5;
  double hd[3] = {10.0, 20.0, 30.0};
  double he[2] = {1.0,  2.0};

  double *dq = dev_alloc_zero<double>(ldq * nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen - 1);

  launch_construct_full(dq, dd, de, nlen, ldq);

  double rq[15]; dev_read_n(rq, dq, ldq * nlen);

  REPORT("padded: q[0,0]=d[0]=10", deq(rq[0+ldq*0], 10.0));
  REPORT("padded: q[1,1]=d[1]=20", deq(rq[1+ldq*1], 20.0));
  REPORT("padded: q[2,2]=d[2]=30", deq(rq[2+ldq*2], 30.0));
  REPORT("padded: q[1,0]=e[0]=1",  deq(rq[1+ldq*0],  1.0));
  REPORT("padded: q[0,1]=e[0]=1",  deq(rq[0+ldq*1],  1.0));
  REPORT("padded: q[2,1]=e[1]=2",  deq(rq[2+ldq*1],  2.0));
  REPORT("padded: q[1,2]=e[1]=2",  deq(rq[1+ldq*2],  2.0));
  REPORT("padded: q[3,0]=0 (padding row)", deq(rq[3+ldq*0], 0.0));
  REPORT("padded: q[4,0]=0 (padding row)", deq(rq[4+ldq*0], 0.0));
  REPORT("padded: q[3,1]=0 (padding row)", deq(rq[3+ldq*1], 0.0));
  REPORT("padded: q[3,2]=0 (padding row)", deq(rq[3+ldq*2], 0.0));
  REPORT("padded: q[0,2]=0 (off-tridiag)", deq(rq[0+ldq*2], 0.0));
  REPORT("padded: q[2,0]=0 (off-tridiag)", deq(rq[2+ldq*0], 0.0));

  dev_free(dq); dev_free(dd); dev_free(de);
}

// ── Test 6: symmetry check ────────────────────────────────────────────
//
// d=[7,3,1,9,5], e=[2,4,6,8] (nlen=5).
// Exhaustive q[i,j]==q[j,i] plus targeted spot checks.
static void test_construct_tridi_symmetry()
{
  printf("\ngpu_construct_full_from_tridi_matrix — symmetry check (nlen=5):\n");

  const int nlen = 5, ldq = 5;
  double hd[5] = {7.0, 3.0, 1.0, 9.0, 5.0};
  double he[4] = {2.0, 4.0, 6.0, 8.0};

  double *dq = dev_alloc_zero<double>(ldq * nlen);
  double *dd = dev_alloc_upload(hd, nlen);
  double *de = dev_alloc_upload(he, nlen - 1);

  launch_construct_full(dq, dd, de, nlen, ldq);

  double rq[25]; dev_read_n(rq, dq, ldq * nlen);

  bool sym_ok = true;
  for (int i = 0; i < nlen && sym_ok; i++)
    for (int j = 0; j < nlen && sym_ok; j++)
      if (!deq(rq[i+ldq*j], rq[j+ldq*i])) sym_ok = false;

  REPORT("symmetry: q[i,j]==q[j,i] for all i,j (nlen=5)", sym_ok);
  REPORT("symmetry: q[0,0]=d[0]=7", deq(rq[0+ldq*0], 7.0));
  REPORT("symmetry: q[1,0]=q[0,1]=e[0]=2",
         deq(rq[1+ldq*0], 2.0) && deq(rq[0+ldq*1], 2.0));
  REPORT("symmetry: q[4,4]=d[4]=5", deq(rq[4+ldq*4], 5.0));
  REPORT("symmetry: q[4,3]=q[3,4]=e[3]=8",
         deq(rq[4+ldq*3], 8.0) && deq(rq[3+ldq*4], 8.0));
  REPORT("symmetry: q[0,2]=q[2,0]=0",
         deq(rq[0+ldq*2], 0.0) && deq(rq[2+ldq*0], 0.0));
  REPORT("symmetry: q[0,4]=q[4,0]=0",
         deq(rq[0+ldq*4], 0.0) && deq(rq[4+ldq*0], 0.0));

  dev_free(dq); dev_free(dd); dev_free(de);
}


// ============================================================
int main(void)
{
  printf("=== Unit tests for gpu_solve_tridi_single_problem.h (Intel SYCL) ===\n");

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
  fprintf(stderr, "Error: this test requires WITH_SYCL_GPU_VERSION\n");
  abort();
}

#endif /* WITH_SYCL_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
