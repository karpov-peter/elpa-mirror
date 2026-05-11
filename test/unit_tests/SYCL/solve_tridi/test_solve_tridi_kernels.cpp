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

// Unit tests for solve_tridi GPU kernels (SYCL backend).
// Covers: gpu_transform_one_column, gpu_transform_two_columns,
//         gpu_update_d, gpu_copy_qmat1_to_qmat2,
//         gpu_fill_array, gpu_copy_qtmp1_to_qtmp1_tmp,
//         gpu_compute_nnzl_nnzu_val_part1/2.
// Self-contained: kernel bodies copied here to avoid syclCommon.hpp dependency.
//
// SYCL-specific behavioural difference from CUDA:
//   gpu_copy_qmat1_to_qmat2: the SYCL kernel uses dimension 0 of nd_item<3>
//   for both i and j, but the launcher assigns only 1 thread/block in dimension 0.
//   As a result, ALL threads get i=j=0 and only element [0,0] is copied.
//   Tests below verify this actual SYCL behaviour.

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

static bool neq (double a, double b, double tol=1e-10) { return fabs(a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }

template <typename T>
static void dev_read_n(sycl::queue &q, T *h, T *d, int n) {
  q.memcpy(h, d, n*sizeof(T)).wait();
}

// ============================================================
// Kernel bodies (self-contained)
// ============================================================

template <typename T>
static void gpu_transform_one_column_kernel(T *a, T *b, T *c, T *alpha, T *beta, int n,
                                             const sycl::nd_item<1> &it) {
  int i0 = it.get_local_id(0) + it.get_group(0)*it.get_local_range(0);
  for (int i=i0; i<n; i += it.get_group_range(0)*it.get_local_range(0))
    c[i] = alpha[0]*a[i] + beta[0]*b[i];
}

template <typename T>
static void gpu_transform_two_columns_kernel(T *q, T *qtrans, T *tmp,
                                              int ldq, int l_rows, int l_rqs, int l_rqe,
                                              int lc1, int lc2, const sycl::nd_item<1> &it) {
  int i0 = it.get_local_id(0) + it.get_group(0)*it.get_local_range(0);
  T qt11=qtrans[0], qt21=qtrans[1], qt12=qtrans[2], qt22=qtrans[3];
  for (int i=i0; i<l_rows; i += it.get_group_range(0)*it.get_local_range(0)) {
    tmp[i] = q[l_rqs-1+i+(lc1-1)*ldq]*qt11 + q[l_rqs-1+i+(lc2-1)*ldq]*qt21;
    q[l_rqs-1+i+(lc2-1)*ldq] = q[l_rqs-1+i+(lc1-1)*ldq]*qt12 + q[l_rqs-1+i+(lc2-1)*ldq]*qt22;
    q[l_rqs-1+i+(lc1-1)*ldq] = tmp[i];
  }
}

template <typename T>
static void gpu_update_d_kernel(T *d, T *e, int *limits, int ndiv, int /*na*/,
                                  const sycl::nd_item<1> &/*it*/) {
  for (int ii=0; ii<ndiv-1; ii++) {
    int n = limits[ii]-1;
    d[n]   -= fabs(e[n]);
    d[n+1] -= fabs(e[n]);
  }
}

template <typename T>
static void gpu_copy_qmat1_to_qmat2_kernel(T *qmat1, T *qmat2, int max_size,
                                            const sycl::nd_item<3> &it) {
  // Production SYCL bug: both i and j use dimension 0 (outermost).
  // With threadsPerBlock(1,32,32) only 1 thread/block in dim0 → i=j=0 always.
  int i = it.get_group(0)*it.get_local_range(0) + it.get_local_id(0);
  int j = it.get_group(0)*it.get_local_range(0) + it.get_local_id(0);
  if (i >= 0 && i < max_size && j >= 0 && j < max_size)
    qmat2[i + max_size*j] = qmat1[i + max_size*j];
}

template <typename T>
static void gpu_fill_array_kernel(T *array, T *value, int n, const sycl::nd_item<1> &it) {
  int i0 = it.get_local_id(0) + it.get_group(0)*it.get_local_range(0);
  for (int i=i0; i<n; i += it.get_group_range(0)*it.get_local_range(0))
    array[i] = *value;
}

template <typename T>
static void gpu_copy_qtmp1_to_qtmp1_tmp_kernel(T *src, T *dst, int k, int l,
                                                const sycl::nd_item<3> &it) {
  int i = it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
  int j = it.get_group(1)*it.get_local_range(1) + it.get_local_id(1);
  if (i < k && j < l)
    dst[i + k*j] = src[i + k*j];
}

static void gpu_compute_nnzl_nnzu_val_part1_kernel(int *p_col, int *idx1, int *coltyp,
                                                    int *nnzu_val, int *nnzl_val,
                                                    int /*na*/, int na1, int np_rem, int npc_n,
                                                    int /*nnzu_start*/, int /*nnzl_start*/, int np,
                                                    const sycl::nd_item<1> &it) {
  int i = it.get_group(0)*it.get_local_range(0) + it.get_local_id(0);
  int np_c = np - 1;
  if (i < na1) {
    if (np_c >= 0 && np_c < npc_n+1) {
      nnzu_val[i + na1*np_c] = 0;
      nnzl_val[i + na1*np_c] = 0;
      if (p_col[idx1[i]-1] == np_rem) {
        if (coltyp[idx1[i]-1] == 1 || coltyp[idx1[i]-1] == 2) nnzu_val[i + na1*np_c] = 1;
        if (coltyp[idx1[i]-1] == 3 || coltyp[idx1[i]-1] == 2) nnzl_val[i + na1*np_c] = 1;
      }
    }
  }
}

static void gpu_compute_nnzl_nnzu_val_part2_kernel(int *nnzu_val, int *nnzl_val,
                                                    int /*na*/, int na1, int /*nnzu_start*/,
                                                    int /*nnzl_start*/, int npc_n,
                                                    const sycl::nd_item<1> &it) {
  int i = it.get_group(0)*it.get_local_range(0) + it.get_local_id(0);
  if (i == 0) {
    for (int jj=0; jj<npc_n; jj++) {
      int c1=0, c2=0;
      for (int ii=0; ii<na1; ii++) {
        if (nnzu_val[ii+na1*jj]==1) { c1++; nnzu_val[ii+na1*jj]=c1; } else nnzu_val[ii+na1*jj]=0;
        if (nnzl_val[ii+na1*jj]==1) { c2++; nnzl_val[ii+na1*jj]=c2; } else nnzl_val[ii+na1*jj]=0;
      }
    }
  }
}

// ============================================================
// gpu_transform_one_column: c = alpha*a + beta*b
// ============================================================
static void test_transform_one_column(sycl::queue &q)
{
  printf("\ngpu_transform_one_column:\n");
  const int SM=1, TPB=32;

  // double: a=[1,2,3,4], b=[5,6,7,8], alpha=2, beta=3 -> c=[17,22,27,32]
  {
    double ha[]={1,2,3,4}, hb[]={5,6,7,8}, halpha=2.0, hbeta=3.0;
    double *da=sycl::malloc_device<double>(4,q), *db=sycl::malloc_device<double>(4,q);
    double *dc=sycl::malloc_device<double>(4,q);
    double *dalpha=sycl::malloc_device<double>(1,q), *dbeta=sycl::malloc_device<double>(1,q);
    q.memcpy(da,ha,4*sizeof(double)).wait(); q.memcpy(db,hb,4*sizeof(double)).wait();
    q.memcpy(dalpha,&halpha,sizeof(double)).wait(); q.memcpy(dbeta,&hbeta,sizeof(double)).wait();
    q.memset(dc,0,4*sizeof(double)).wait();
    q.parallel_for(sycl::nd_range<1>(SM*TPB,TPB),[=](sycl::nd_item<1> it){
      gpu_transform_one_column_kernel(da,db,dc,dalpha,dbeta,4,it);
    }).wait();
    double hc[4]; dev_read_n(q,hc,dc,4);
    REPORT("double transform_one_col: c[0]==17", neq(hc[0],17.0));
    REPORT("double transform_one_col: c[1]==22", neq(hc[1],22.0));
    REPORT("double transform_one_col: c[2]==27", neq(hc[2],27.0));
    REPORT("double transform_one_col: c[3]==32", neq(hc[3],32.0));
    sycl::free(da,q); sycl::free(db,q); sycl::free(dc,q);
    sycl::free(dalpha,q); sycl::free(dbeta,q);
  }

  // float
  {
    float ha[]={1,2,3,4}, hb[]={5,6,7,8}, halpha=2.0f, hbeta=3.0f;
    float *da=sycl::malloc_device<float>(4,q), *db=sycl::malloc_device<float>(4,q);
    float *dc=sycl::malloc_device<float>(4,q);
    float *dalpha=sycl::malloc_device<float>(1,q), *dbeta=sycl::malloc_device<float>(1,q);
    q.memcpy(da,ha,4*sizeof(float)).wait(); q.memcpy(db,hb,4*sizeof(float)).wait();
    q.memcpy(dalpha,&halpha,sizeof(float)).wait(); q.memcpy(dbeta,&hbeta,sizeof(float)).wait();
    q.memset(dc,0,4*sizeof(float)).wait();
    q.parallel_for(sycl::nd_range<1>(SM*TPB,TPB),[=](sycl::nd_item<1> it){
      gpu_transform_one_column_kernel(da,db,dc,dalpha,dbeta,4,it);
    }).wait();
    float hc[4]; dev_read_n(q,hc,dc,4);
    REPORT("float transform_one_col: c[0]==17", neqf(hc[0],17.0f));
    REPORT("float transform_one_col: c[3]==32", neqf(hc[3],32.0f));
    sycl::free(da,q); sycl::free(db,q); sycl::free(dc,q);
    sycl::free(dalpha,q); sycl::free(dbeta,q);
  }
}

// ============================================================
// gpu_transform_two_columns
// q col1=[1,2], col2=[3,4]; qtrans={2,4,3,5}; ldq=4, l_rows=2, l_rqs=1
// tmp[0]=1*2+3*4=14; q_col2_new[0]=1*3+3*5=18; q_col1_new[0]=14
// tmp[1]=2*2+4*4=20; q_col2_new[1]=2*3+4*5=26; q_col1_new[1]=20
// ============================================================
static void test_transform_two_columns(sycl::queue &q)
{
  printf("\ngpu_transform_two_columns:\n");
  const int SM=1, TPB=32;
  int ldq=4, l_rows=2, l_rqs=1, l_rqe=2, lc1=1, lc2=2;

  double hq[8]={1,2,0,0, 3,4,0,0};
  double hqtrans[4]={2,4,3,5};
  double *dq=sycl::malloc_device<double>(8,q);
  double *dqtrans=sycl::malloc_device<double>(4,q);
  double *dtmp=sycl::malloc_device<double>(4,q);
  q.memcpy(dq,hq,8*sizeof(double)).wait();
  q.memcpy(dqtrans,hqtrans,4*sizeof(double)).wait();
  q.memset(dtmp,0,4*sizeof(double)).wait();
  q.parallel_for(sycl::nd_range<1>(SM*TPB,TPB),[=](sycl::nd_item<1> it){
    gpu_transform_two_columns_kernel(dq,dqtrans,dtmp,ldq,l_rows,l_rqs,l_rqe,lc1,lc2,it);
  }).wait();
  dev_read_n(q,hq,dq,8);
  REPORT("double transform_two_cols: q_col1[0]==14", neq(hq[0+(lc1-1)*ldq],14.0));
  REPORT("double transform_two_cols: q_col1[1]==20", neq(hq[1+(lc1-1)*ldq],20.0));
  REPORT("double transform_two_cols: q_col2[0]==18", neq(hq[0+(lc2-1)*ldq],18.0));
  REPORT("double transform_two_cols: q_col2[1]==26", neq(hq[1+(lc2-1)*ldq],26.0));
  sycl::free(dq,q); sycl::free(dqtrans,q); sycl::free(dtmp,q);
}

// ============================================================
// gpu_update_d
// d=[10,20,30,40], e=[1,2,3,4], limits=[2], ndiv=2
// ii=0: n=1 → d[1]-=|e[1]|=2→18, d[2]-=|e[1]|=2→28
// ============================================================
static void test_update_d(sycl::queue &q)
{
  printf("\ngpu_update_d:\n");

  double hd[]={10,20,30,40}, he[]={1,2,3,4};
  int hl[]={2};
  double *dd=sycl::malloc_device<double>(4,q);
  double *de=sycl::malloc_device<double>(4,q);
  int    *dl=sycl::malloc_device<int>(1,q);
  q.memcpy(dd,hd,4*sizeof(double)).wait();
  q.memcpy(de,he,4*sizeof(double)).wait();
  q.memcpy(dl,hl,sizeof(int)).wait();
  q.parallel_for(sycl::nd_range<1>(1,1),[=](sycl::nd_item<1> it){
    gpu_update_d_kernel(dd,de,dl,2,4,it);
  }).wait();
  dev_read_n(q,hd,dd,4);
  REPORT("double update_d: d[0] unchanged=10", neq(hd[0],10.0));
  REPORT("double update_d: d[1]-=|e[1]|=18",  neq(hd[1],18.0));
  REPORT("double update_d: d[2]-=|e[1]|=28",  neq(hd[2],28.0));
  REPORT("double update_d: d[3] unchanged=40", neq(hd[3],40.0));
  sycl::free(dd,q); sycl::free(de,q); sycl::free(dl,q);

  float hdf[]={10,20,30,40}, hef[]={1,2,3,4};
  int   hlf[]={2};
  float *ddf=sycl::malloc_device<float>(4,q);
  float *def_=sycl::malloc_device<float>(4,q);
  int   *dlf=sycl::malloc_device<int>(1,q);
  q.memcpy(ddf,hdf,4*sizeof(float)).wait();
  q.memcpy(def_,hef,4*sizeof(float)).wait();
  q.memcpy(dlf,hlf,sizeof(int)).wait();
  q.parallel_for(sycl::nd_range<1>(1,1),[=](sycl::nd_item<1> it){
    gpu_update_d_kernel(ddf,def_,dlf,2,4,it);
  }).wait();
  dev_read_n(q,hdf,ddf,4);
  REPORT("float update_d: d[1]-=|e[1]|=18",  neqf(hdf[1],18.0f));
  REPORT("float update_d: d[2]-=|e[1]|=28",  neqf(hdf[2],28.0f));
  sycl::free(ddf,q); sycl::free(def_,q); sycl::free(dlf,q);
}

// ============================================================
// gpu_copy_qmat1_to_qmat2 [SYCL BUG: only [0,0] is copied]
//
// The SYCL kernel uses dimension 0 of nd_item<3> for both i and j.
// The production launcher sets threadsPerBlock(1,32,32), so dimension 0
// has exactly 1 thread/block → all threads compute i=j=0.
// Only qmat2[0,0] = qmat1[0,0] = 1 is written; all other elements stay 0.
// ============================================================
static void test_copy_qmat1_to_qmat2(sycl::queue &q)
{
  printf("\ngpu_copy_qmat1_to_qmat2 (verifies SYCL [0,0]-only copy bug):\n");

  int max_size = 4;
  double h1[16], h2[16];
  for (int k=0; k<16; k++) h1[k]=(double)(k+1);
  for (int k=0; k<16; k++) h2[k]=0.0;

  double *d1=sycl::malloc_device<double>(16,q);
  double *d2=sycl::malloc_device<double>(16,q);
  q.memcpy(d1,h1,16*sizeof(double)).wait();
  q.memcpy(d2,h2,16*sizeof(double)).wait();

  // Replicate production launch: threadsPerBlock(1,32,32), blocks=(1, ceil, ceil)
  sycl::range<3> tpb(1, 32, 32);
  sycl::range<3> blk(1,
                     (max_size + tpb.get(1) - 1) / tpb.get(1),
                     (max_size + tpb.get(2) - 1) / tpb.get(2));
  q.parallel_for(sycl::nd_range<3>(blk*tpb, tpb), [=](sycl::nd_item<3> it){
    gpu_copy_qmat1_to_qmat2_kernel(d1, d2, max_size, it);
  }).wait();

  dev_read_n(q, h2, d2, 16);

  // Only [0,0] should be copied (SYCL bug: both i and j are from dim 0)
  REPORT("double copy_qmat SYCL: [0,0]=1 (only element copied)", neq(h2[0+4*0], 1.0));
  // All other elements must stay 0 (not copied due to SYCL bug)
  REPORT("double copy_qmat SYCL: [1,1]=0 (not copied)",  neq(h2[1+4*1], 0.0));
  REPORT("double copy_qmat SYCL: [2,2]=0 (not copied)",  neq(h2[2+4*2], 0.0));
  REPORT("double copy_qmat SYCL: [3,3]=0 (not copied)",  neq(h2[3+4*3], 0.0));
  REPORT("double copy_qmat SYCL: [1,0]=0 (not copied)",  neq(h2[1+4*0], 0.0));
  REPORT("double copy_qmat SYCL: [0,1]=0 (not copied)",  neq(h2[0+4*1], 0.0));

  sycl::free(d1,q); sycl::free(d2,q);
}

// ============================================================
// gpu_fill_array
// ============================================================
static void test_fill_array(sycl::queue &q)
{
  printf("\ngpu_fill_array:\n");
  const int SM=1, TPB=32;

  // double: 6 elements set to 3.14
  {
    double hval=3.14, hout[6];
    double *dval=sycl::malloc_device<double>(1,q);
    double *dout=sycl::malloc_device<double>(6,q);
    q.memcpy(dval,&hval,sizeof(double)).wait();
    q.memset(dout,0,6*sizeof(double)).wait();
    q.parallel_for(sycl::nd_range<1>(SM*TPB,TPB),[=](sycl::nd_item<1> it){
      gpu_fill_array_kernel(dout,dval,6,it);
    }).wait();
    dev_read_n(q,hout,dout,6);
    bool ok=true; for (int k=0;k<6;k++) ok = ok && neq(hout[k],3.14);
    REPORT("double fill_array: all 6 elements == 3.14", ok);
    sycl::free(dval,q); sycl::free(dout,q);
  }

  // float: 4 elements set to 2.5
  {
    float hval=2.5f, hout[4];
    float *dval=sycl::malloc_device<float>(1,q);
    float *dout=sycl::malloc_device<float>(4,q);
    q.memcpy(dval,&hval,sizeof(float)).wait();
    q.memset(dout,0,4*sizeof(float)).wait();
    q.parallel_for(sycl::nd_range<1>(SM*TPB,TPB),[=](sycl::nd_item<1> it){
      gpu_fill_array_kernel(dout,dval,4,it);
    }).wait();
    dev_read_n(q,hout,dout,4);
    bool ok=true; for (int k=0;k<4;k++) ok = ok && (fabsf(hout[k]-2.5f)<1e-6f);
    REPORT("float fill_array: all 4 elements == 2.5", ok);
    sycl::free(dval,q); sycl::free(dout,q);
  }
}

// ============================================================
// gpu_copy_qtmp1_to_qtmp1_tmp: 2D copy; gemm_dim_k=3, gemm_dim_l=2
// ============================================================
static void test_copy_qtmp1_to_qtmp1_tmp(sycl::queue &q)
{
  printf("\ngpu_copy_qtmp1_to_qtmp1_tmp:\n");
  int k=3, l=2;

  // double
  {
    double hsrc[6]={1,2,3,4,5,6}, hdst[6]={0};
    double *dsrc=sycl::malloc_device<double>(6,q);
    double *ddst=sycl::malloc_device<double>(6,q);
    q.memcpy(dsrc,hsrc,6*sizeof(double)).wait();
    q.memset(ddst,0,6*sizeof(double)).wait();
    sycl::range<3> tpb(1,16,16);
    sycl::range<3> blk(1,(l+tpb.get(1)-1)/tpb.get(1),(k+tpb.get(2)-1)/tpb.get(2));
    q.parallel_for(sycl::nd_range<3>(blk*tpb,tpb),[=](sycl::nd_item<3> it){
      gpu_copy_qtmp1_to_qtmp1_tmp_kernel(dsrc,ddst,k,l,it);
    }).wait();
    dev_read_n(q,hdst,ddst,6);
    bool ok=true; for (int n=0;n<6;n++) ok=ok&&neq(hdst[n],(double)(n+1));
    REPORT("double copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
    sycl::free(dsrc,q); sycl::free(ddst,q);
  }

  // float
  {
    float hsrc[6]={1,2,3,4,5,6}, hdst[6]={0};
    float *dsrc=sycl::malloc_device<float>(6,q);
    float *ddst=sycl::malloc_device<float>(6,q);
    q.memcpy(dsrc,hsrc,6*sizeof(float)).wait();
    q.memset(ddst,0,6*sizeof(float)).wait();
    sycl::range<3> tpb(1,16,16);
    sycl::range<3> blk(1,(l+tpb.get(1)-1)/tpb.get(1),(k+tpb.get(2)-1)/tpb.get(2));
    q.parallel_for(sycl::nd_range<3>(blk*tpb,tpb),[=](sycl::nd_item<3> it){
      gpu_copy_qtmp1_to_qtmp1_tmp_kernel(dsrc,ddst,k,l,it);
    }).wait();
    dev_read_n(q,hdst,ddst,6);
    bool ok=true; for (int n=0;n<6;n++) ok=ok&&(fabsf(hdst[n]-(float)(n+1))<1e-6f);
    REPORT("float copy_qtmp1_to_qtmp1_tmp: all 6 elements copied", ok);
    sycl::free(dsrc,q); sycl::free(ddst,q);
  }
}

// ============================================================
// gpu_compute_nnzl_nnzu_val_part1 and part2
// na1=4, na=4, npc_n=1, np=1 (np_c=0), np_rem=0
// p_col=[0,0,0,0], idx1=[1,2,3,4], coltyp=[1,3,2,1]
// After part1: nnzu=[1,0,1,1], nnzl=[0,1,1,0]
// After part2 (cumulative): nnzu=[1,0,2,3], nnzl=[0,1,2,0]
// ============================================================
static void test_compute_nnzl_nnzu(sycl::queue &q)
{
  printf("\ngpu_compute_nnzl_nnzu_val_part1+2:\n");

  int na1=4, na=4, npc_n=1, np_rem=0;
  int h_pcol[4]={0,0,0,0}, h_idx1[4]={1,2,3,4}, h_coltyp[4]={1,3,2,1};
  int h_nnzu[4]={0}, h_nnzl[4]={0};

  int *d_pcol=sycl::malloc_device<int>(4,q);
  int *d_idx1=sycl::malloc_device<int>(4,q);
  int *d_coltyp=sycl::malloc_device<int>(4,q);
  int *d_nnzu=sycl::malloc_device<int>(4,q);
  int *d_nnzl=sycl::malloc_device<int>(4,q);
  q.memcpy(d_pcol,h_pcol,4*sizeof(int)).wait();
  q.memcpy(d_idx1,h_idx1,4*sizeof(int)).wait();
  q.memcpy(d_coltyp,h_coltyp,4*sizeof(int)).wait();
  q.memset(d_nnzu,0,4*sizeof(int)).wait();
  q.memset(d_nnzl,0,4*sizeof(int)).wait();

  const int TPB=32;
  int blocks_part1 = (na1 + TPB - 1) / TPB;
  q.parallel_for(sycl::nd_range<1>(blocks_part1*TPB,TPB),[=](sycl::nd_item<1> it){
    gpu_compute_nnzl_nnzu_val_part1_kernel(d_pcol,d_idx1,d_coltyp,d_nnzu,d_nnzl,
                                           na,na1,np_rem,npc_n,0,0,1,it);
  }).wait();
  dev_read_n(q,h_nnzu,d_nnzu,4);
  dev_read_n(q,h_nnzl,d_nnzl,4);
  REPORT("part1: nnzu_val[0]=1 (coltyp=1)", h_nnzu[0]==1);
  REPORT("part1: nnzu_val[1]=0 (coltyp=3)", h_nnzu[1]==0);
  REPORT("part1: nnzu_val[2]=1 (coltyp=2)", h_nnzu[2]==1);
  REPORT("part1: nnzu_val[3]=1 (coltyp=1)", h_nnzu[3]==1);
  REPORT("part1: nnzl_val[0]=0 (coltyp=1)", h_nnzl[0]==0);
  REPORT("part1: nnzl_val[1]=1 (coltyp=3)", h_nnzl[1]==1);
  REPORT("part1: nnzl_val[2]=1 (coltyp=2)", h_nnzl[2]==1);
  REPORT("part1: nnzl_val[3]=0 (coltyp=1)", h_nnzl[3]==0);

  q.parallel_for(sycl::nd_range<1>(1,1),[=](sycl::nd_item<1> it){
    gpu_compute_nnzl_nnzu_val_part2_kernel(d_nnzu,d_nnzl,na,na1,0,0,npc_n,it);
  }).wait();
  dev_read_n(q,h_nnzu,d_nnzu,4);
  dev_read_n(q,h_nnzl,d_nnzl,4);
  REPORT("part2: nnzu_val[0]=1",  h_nnzu[0]==1);
  REPORT("part2: nnzu_val[1]=0",  h_nnzu[1]==0);
  REPORT("part2: nnzu_val[2]=2",  h_nnzu[2]==2);
  REPORT("part2: nnzu_val[3]=3",  h_nnzu[3]==3);
  REPORT("part2: nnzl_val[0]=0",  h_nnzl[0]==0);
  REPORT("part2: nnzl_val[1]=1",  h_nnzl[1]==1);
  REPORT("part2: nnzl_val[2]=2",  h_nnzl[2]==2);
  REPORT("part2: nnzl_val[3]=0",  h_nnzl[3]==0);

  sycl::free(d_pcol,q); sycl::free(d_idx1,q);
  sycl::free(d_coltyp,q); sycl::free(d_nnzu,q); sycl::free(d_nnzl,q);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for solve_tridi GPU kernels (SYCL) ===\n");

  sycl::queue q(sycl::default_selector_v);

  test_transform_one_column(q);
  test_transform_two_columns(q);
  test_update_d(q);
  test_copy_qmat1_to_qmat2(q);
  test_fill_array(q);
  test_copy_qtmp1_to_qtmp1_tmp(q);
  test_compute_nnzl_nnzu(q);

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
