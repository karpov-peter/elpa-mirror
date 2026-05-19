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

// local_buffer: SYCL shared-memory accessor (same helper as production code)
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20230000
template <typename T> using local_buffer = sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local>;
#else
template <typename T> using local_buffer = sycl::local_accessor<T>;
#endif

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-70s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool neq (double a, double b, double tol=1e-10) { return fabs (a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }

template <typename T>
static T dev_read(sycl::queue &q, T *d) {
  T h{};
  q.memcpy(&h, d, sizeof(T)).wait();
  return h;
}

template <typename T>
static void dev_read_n(sycl::queue &q, T *h, T *d, int n) {
  q.memcpy(h, d, n*sizeof(T)).wait();
}

// ============================================================
// Kernel bodies (self-contained, no getQueueOrDefault)
// ============================================================

template <typename T>
static void gpu_hh_transform_kernel(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev, T *tau_dev) {
  auto alpha_r = elpaDeviceRealPart(*alpha_dev);
  auto alpha_i = elpaDeviceImagPart(*alpha_dev);

  if (elpaDeviceRealPart(*xnorm_sq_dev)==0.0 && alpha_i==0.0) {
    if (alpha_r >= 0.0) *tau_dev = elpaDeviceNumber<T>(0.0);
    else {
      *tau_dev = elpaDeviceNumber<T>(2.0);
      *alpha_dev = elpaDeviceMultiply(*alpha_dev, elpaDeviceNumber<T>(-1.0));
    }
    *xf_dev = elpaDeviceNumber<T>(0.0);
  } else {
    T beta = elpaDeviceNumber<T>(elpaDeviceSign(
        elpaDeviceSqrt(alpha_r*alpha_r + alpha_i*alpha_i + elpaDeviceRealPart(*xnorm_sq_dev)),
        elpaDeviceRealPart(*alpha_dev)));
    *alpha_dev = elpaDeviceAdd(*alpha_dev, beta);
    if (elpaDeviceRealPart(beta) < 0) {
      *tau_dev = elpaDeviceDivide(*alpha_dev, beta);
      beta = elpaDeviceMultiply(beta, elpaDeviceNumber<T>(-1.0));
    } else {
      alpha_r = alpha_i * alpha_i / elpaDeviceRealPart(*alpha_dev);
      alpha_r = alpha_r + elpaDeviceRealPart(*xnorm_sq_dev) / elpaDeviceRealPart(*alpha_dev);
      *tau_dev = elpaDeviceDivide(elpaDeviceNumberFromRealImag<T>(alpha_r, -alpha_i), beta);
      *alpha_dev = elpaDeviceNumberFromRealImag<T>(-alpha_r, alpha_i);
    }
    *xf_dev = elpaDeviceDivide(elpaDeviceNumber<T>(1.0), *alpha_dev);
    *alpha_dev = beta;
  }
}

template <typename T>
static void gpu_dot_product_kernel(int n, T *x_dev, int incx, T *y_dev, int incy, T *result_dev,
                                    sycl::nd_item<1> it, local_buffer<T> cache) {
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  T temp = elpaDeviceNumber<T>(0.0);
  for (int i=tid; i<n; i += it.get_local_range(0)*it.get_group_range(0))
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(x_dev[i*incx]), y_dev[i*incy]));
  cache[it.get_local_id(0)] = temp;
  it.barrier();
  for (int i=it.get_local_range(0)/2; i>0; i/=2) {
    if (it.get_local_id(0) < i)
      cache[it.get_local_id(0)] = elpaDeviceAdd(cache[it.get_local_id(0)], cache[it.get_local_id(0)+i]);
    it.barrier();
  }
  if (it.get_local_id(0)==0) atomicAdd(&result_dev[0], cache[0]);
}

template <typename T>
static void gpu_dot_product_and_assign_kernel(T *v_row_dev, int l_rows, int isOurProcessRow, T *aux1_dev,
                                               sycl::nd_item<1> it, local_buffer<T> cache) {
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  T temp = elpaDeviceNumber<T>(0.0);
  for (int i=tid; i<l_rows-1; i += it.get_local_range(0)*it.get_group_range(0))
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(v_row_dev[i]), v_row_dev[i]));
  cache[it.get_local_id(0)] = temp;
  it.barrier();
  for (int i=it.get_local_range(0)/2; i>0; i/=2) {
    if (it.get_local_id(0) < i)
      cache[it.get_local_id(0)] = elpaDeviceAdd(cache[it.get_local_id(0)], cache[it.get_local_id(0)+i]);
    it.barrier();
  }
  if (it.get_local_id(0)==0) atomicAdd(&aux1_dev[0], cache[0]);
  if (tid==0) {
    if (isOurProcessRow) {
      aux1_dev[1] = v_row_dev[l_rows-1];
    } else {
      if (l_rows>0) atomicAdd(&aux1_dev[0], elpaDeviceMultiply(elpaDeviceComplexConjugate(v_row_dev[l_rows-1]), v_row_dev[l_rows-1]));
      aux1_dev[1] = elpaDeviceNumber<T>(0.0);
    }
  }
}

// ============================================================
// gpu_hh_transform
// Case A: xnorm_sq=0, alpha=4 (>0) -> tau=0, xf=0, alpha=4
// Case B: xnorm_sq=0, alpha=-4 (<0) -> tau=2, xf=0, alpha=4
// Case C: alpha=3, xnorm_sq=16 -> alpha_out=5, tau=0.4, xf=-0.5
// ============================================================
template <typename T>
static void test_hh_transform(sycl::queue &q, const char *tname) {
  printf("\ngpu_hh_transform<%s>:\n", tname);

  T *alpha = sycl::malloc_device<T>(1, q);
  T *xnorm = sycl::malloc_device<T>(1, q);
  T *xf    = sycl::malloc_device<T>(1, q);
  T *tau   = sycl::malloc_device<T>(1, q);

  auto upload = [&](T *d, T val) { q.memcpy(d, &val, sizeof(T)).wait(); };

  // Case A
  {
    T ha = elpaHostNumberFromInt<T>(4);
    T hz = elpaHostNumberFromInt<T>(0);
    upload(alpha, ha); upload(xnorm, hz);
    q.single_task([=](){ gpu_hh_transform_kernel(alpha, xnorm, xf, tau); }).wait();
    T h_tau = dev_read(q, tau); T h_xf = dev_read(q, xf); T h_alpha_out = dev_read(q, alpha);
    char lbl[128]; snprintf(lbl,sizeof(lbl),"%s Case A (xnorm=0,alpha>0): tau=0", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(h_tau), 0.0) &&
                neq((double)elpaDeviceRealPart(h_xf),  0.0) &&
                neq((double)elpaDeviceRealPart(h_alpha_out), 4.0));
  }

  // Case B
  {
    T ha = elpaHostNumberFromInt<T>(-4);
    T hz = elpaHostNumberFromInt<T>(0);
    upload(alpha, ha); upload(xnorm, hz);
    q.single_task([=](){ gpu_hh_transform_kernel(alpha, xnorm, xf, tau); }).wait();
    T h_tau = dev_read(q, tau); T h_xf = dev_read(q, xf); T h_alpha_out = dev_read(q, alpha);
    char lbl[128]; snprintf(lbl,sizeof(lbl),"%s Case B (xnorm=0,alpha<0): tau=2,alpha=4", tname);
    REPORT(lbl, neq((double)elpaDeviceRealPart(h_tau), 2.0) &&
                neq((double)elpaDeviceRealPart(h_xf),  0.0) &&
                neq((double)elpaDeviceRealPart(h_alpha_out), 4.0));
  }

  // Case C
  {
    T ha = elpaHostNumberFromInt<T>(3);
    T hx = elpaHostNumberFromInt<T>(16);
    upload(alpha, ha); upload(xnorm, hx);
    q.single_task([=](){ gpu_hh_transform_kernel(alpha, xnorm, xf, tau); }).wait();
    T h_tau = dev_read(q, tau); T h_xf = dev_read(q, xf); T h_alpha_out = dev_read(q, alpha);
    char lbl1[128], lbl2[128], lbl3[128];
    snprintf(lbl1,sizeof(lbl1),"%s Case C: alpha_out=5.0", tname);
    snprintf(lbl2,sizeof(lbl2),"%s Case C: tau_out=0.4",   tname);
    snprintf(lbl3,sizeof(lbl3),"%s Case C: xf_out=-0.5",   tname);
    REPORT(lbl1, neq((double)elpaDeviceRealPart(h_alpha_out), 5.0, 1e-6));
    REPORT(lbl2, neq((double)elpaDeviceRealPart(h_tau),       0.4, 1e-6));
    REPORT(lbl3, neq((double)elpaDeviceRealPart(h_xf),       -0.5, 1e-6));
  }

  sycl::free(alpha, q); sycl::free(xnorm, q);
  sycl::free(xf, q);    sycl::free(tau, q);
}

// ============================================================
// gpu_dot_product: result = sum_i conj(x[i*incx]) * y[i*incy]
// Use 1 workgroup of 32 threads (MIN_THREADS_PER_BLOCK for SYCL).
// ============================================================
static void test_dot_product(sycl::queue &q)
{
  printf("\ngpu_dot_product:\n");
  const int TPB = 32; // power of 2, fits MIN_THREADS_PER_BLOCK
  const int SM = 1;

  // double: x=[1,2,3], y=[4,5,6], result=32
  {
    double hx[]={1,2,3}, hy[]={4,5,6};
    double *dx = sycl::malloc_device<double>(3, q);
    double *dy = sycl::malloc_device<double>(3, q);
    double *dr = sycl::malloc_device<double>(1, q);
    q.memcpy(dx, hx, 3*sizeof(double)).wait();
    q.memcpy(dy, hy, 3*sizeof(double)).wait();
    double zero=0.0; q.memcpy(dr, &zero, sizeof(double)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<double> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_kernel(3, dx, 1, dy, 1, dr, it, cache);
      });
    }).wait();
    REPORT("double: dot([1,2,3],[4,5,6])==32", neq(dev_read(q, dr), 32.0));
    sycl::free(dx,q); sycl::free(dy,q); sycl::free(dr,q);
  }

  // float
  {
    float hx[]={1,2,3}, hy[]={4,5,6};
    float *dx = sycl::malloc_device<float>(3, q);
    float *dy = sycl::malloc_device<float>(3, q);
    float *dr = sycl::malloc_device<float>(1, q);
    q.memcpy(dx, hx, 3*sizeof(float)).wait();
    q.memcpy(dy, hy, 3*sizeof(float)).wait();
    float zerof=0.0f; q.memcpy(dr, &zerof, sizeof(float)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<float> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_kernel(3, dx, 1, dy, 1, dr, it, cache);
      });
    }).wait();
    REPORT("float: dot([1,2,3],[4,5,6])==32", neqf(dev_read(q, dr), 32.0f));
    sycl::free(dx,q); sycl::free(dy,q); sycl::free(dr,q);
  }

  // double_complex: conj([1+1i,2+2i])·[3+3i,4+4i] = 6+16 = 22+0i
  {
    gpuDoubleComplex hx[]={make_gpuDoubleComplex(1,1), make_gpuDoubleComplex(2,2)};
    gpuDoubleComplex hy[]={make_gpuDoubleComplex(3,3), make_gpuDoubleComplex(4,4)};
    gpuDoubleComplex zero_c = make_gpuDoubleComplex(0,0);
    gpuDoubleComplex *dx = sycl::malloc_device<gpuDoubleComplex>(2, q);
    gpuDoubleComplex *dy = sycl::malloc_device<gpuDoubleComplex>(2, q);
    gpuDoubleComplex *dr = sycl::malloc_device<gpuDoubleComplex>(1, q);
    q.memcpy(dx, hx, 2*sizeof(gpuDoubleComplex)).wait();
    q.memcpy(dy, hy, 2*sizeof(gpuDoubleComplex)).wait();
    q.memcpy(dr, &zero_c, sizeof(gpuDoubleComplex)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<gpuDoubleComplex> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_kernel(2, dx, 1, dy, 1, dr, it, cache);
      });
    }).wait();
    gpuDoubleComplex h = dev_read(q, dr);
    REPORT("double_complex: conj(x).y real==22", neq(h.real(), 22.0));
    REPORT("double_complex: conj(x).y imag==0",  neq(h.imag(),  0.0));
    sycl::free(dx,q); sycl::free(dy,q); sycl::free(dr,q);
  }

  // float_complex
  {
    gpuFloatComplex hx[]={make_gpuFloatComplex(1,1), make_gpuFloatComplex(2,2)};
    gpuFloatComplex hy[]={make_gpuFloatComplex(3,3), make_gpuFloatComplex(4,4)};
    gpuFloatComplex zero_cf = make_gpuFloatComplex(0,0);
    gpuFloatComplex *dx = sycl::malloc_device<gpuFloatComplex>(2, q);
    gpuFloatComplex *dy = sycl::malloc_device<gpuFloatComplex>(2, q);
    gpuFloatComplex *dr = sycl::malloc_device<gpuFloatComplex>(1, q);
    q.memcpy(dx, hx, 2*sizeof(gpuFloatComplex)).wait();
    q.memcpy(dy, hy, 2*sizeof(gpuFloatComplex)).wait();
    q.memcpy(dr, &zero_cf, sizeof(gpuFloatComplex)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<gpuFloatComplex> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_kernel(2, dx, 1, dy, 1, dr, it, cache);
      });
    }).wait();
    gpuFloatComplex h = dev_read(q, dr);
    REPORT("float_complex: conj(x).y real==22", neqf((float)h.real(), 22.0f));
    REPORT("float_complex: conj(x).y imag==0",  neqf((float)h.imag(),  0.0f));
    sycl::free(dx,q); sycl::free(dy,q); sycl::free(dr,q);
  }

  // stride test: incx=2, incy=2; effective x=[1,2,3], y=[4,5,6]
  {
    double hx[]={1,0,2,0,3}, hy[]={4,0,5,0,6};
    double *dx = sycl::malloc_device<double>(5, q);
    double *dy = sycl::malloc_device<double>(5, q);
    double *dr = sycl::malloc_device<double>(1, q);
    q.memcpy(dx, hx, 5*sizeof(double)).wait();
    q.memcpy(dy, hy, 5*sizeof(double)).wait();
    double zero=0.0; q.memcpy(dr, &zero, sizeof(double)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<double> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_kernel(3, dx, 2, dy, 2, dr, it, cache);
      });
    }).wait();
    REPORT("double: dot with stride 2 == 32", neq(dev_read(q, dr), 32.0));
    sycl::free(dx,q); sycl::free(dy,q); sycl::free(dr,q);
  }
}

// ============================================================
// gpu_dot_product_and_assign
// v=[1,2,3,4], l_rows=4
// isOurProcessRow=1: aux[0]=1+4+9=14, aux[1]=4
// isOurProcessRow=0: aux[0]=1+4+9+16=30, aux[1]=0
// ============================================================
static void test_dot_product_and_assign(sycl::queue &q)
{
  printf("\ngpu_dot_product_and_assign:\n");
  const int TPB = 32;
  const int SM  = 1;

  double hv[]={1,2,3,4};
  double *dv  = sycl::malloc_device<double>(4, q);
  double *aux = sycl::malloc_device<double>(2, q);
  q.memcpy(dv, hv, 4*sizeof(double)).wait();

  // isOurProcessRow=1
  {
    double zeros[2]={0,0}; q.memcpy(aux, zeros, 2*sizeof(double)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<double> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_and_assign_kernel(dv, 4, 1, aux, it, cache);
      });
    }).wait();
    double ha[2]; dev_read_n(q, ha, aux, 2);
    REPORT("double isOurProcessRow=1: aux[0]==14 (dot of first 3)", neq(ha[0], 14.0));
    REPORT("double isOurProcessRow=1: aux[1]==4  (last element)",   neq(ha[1],  4.0));
  }

  // isOurProcessRow=0
  {
    double zeros[2]={0,0}; q.memcpy(aux, zeros, 2*sizeof(double)).wait();
    q.submit([&](sycl::handler &cgh){
      local_buffer<double> cache(sycl::range<1>(TPB), cgh);
      cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
        gpu_dot_product_and_assign_kernel(dv, 4, 0, aux, it, cache);
      });
    }).wait();
    double ha[2]; dev_read_n(q, ha, aux, 2);
    REPORT("double isOurProcessRow=0: aux[0]==30 (dot of all 4)", neq(ha[0], 30.0));
    REPORT("double isOurProcessRow=0: aux[1]==0  (set to zero)",  neq(ha[1],  0.0));
  }

  sycl::free(dv,q); sycl::free(aux,q);

  // float
  float hvf[]={1,2,3,4};
  float *dvf  = sycl::malloc_device<float>(4, q);
  float *auxf = sycl::malloc_device<float>(2, q);
  q.memcpy(dvf, hvf, 4*sizeof(float)).wait();
  float zerosf[2]={0,0}; q.memcpy(auxf, zerosf, 2*sizeof(float)).wait();
  q.submit([&](sycl::handler &cgh){
    local_buffer<float> cache(sycl::range<1>(TPB), cgh);
    cgh.parallel_for(sycl::nd_range<1>(SM*TPB, TPB), [=](sycl::nd_item<1> it){
      gpu_dot_product_and_assign_kernel(dvf, 4, 1, auxf, it, cache);
    });
  }).wait();
  float haf[2]; dev_read_n(q, haf, auxf, 2);
  REPORT("float isOurProcessRow=1: aux[0]==14", neqf(haf[0], 14.0f));
  REPORT("float isOurProcessRow=1: aux[1]==4",  neqf(haf[1],  4.0f));
  sycl::free(dvf,q); sycl::free(auxf,q);
}

// ============================================================
// Kernel 1: gpu_copy_and_set_zeros_kernel
// ============================================================

template <typename T, typename T_real>
static void gpu_copy_and_set_zeros_kernel(T *v_row_dev, T *u_col_dev, const T *a_dev,
                         T *aux1_dev, T *vav_dev, T_real *d_vec_dev,
                         int l_rows, int l_cols, int matrixRows, int istep,
                         int isOurProcessRow, int isOurProcessCol, int isOurProcessCol_prev,
                         int isSkewsymmetric, int useCCL,
                         sycl::nd_item<1> item_ct1){
  int tid = item_ct1.get_local_id(0) + item_ct1.get_group(0) * item_ct1.get_local_range(0);
  if (isOurProcessCol_prev) {
    for (int i_row = tid; i_row < l_rows; i_row += item_ct1.get_local_range(0)*item_ct1.get_group_range(0))
      v_row_dev[i_row] = a_dev[i_row + matrixRows * l_cols];
  }
  if (l_cols > l_rows) {
    T zero = elpaDeviceNumber<T>(0.0);
    for (int i = l_rows + tid; i < l_cols; i += item_ct1.get_local_range(0)*item_ct1.get_group_range(0))
      u_col_dev[i] = zero;
  }
  if (tid==0) {
    aux1_dev[0] = elpaDeviceNumber<T>(0.0);
    if (useCCL) *vav_dev = elpaDeviceNumber<T>(0.0);
    if (isOurProcessRow && isOurProcessCol) {
      if (isSkewsymmetric) d_vec_dev[istep-1-1] = 0.0;
      else d_vec_dev[istep-1-1] = elpaDeviceRealPart(a_dev[(l_rows-1) + matrixRows*(l_cols-1)]);
    }
  }
}

static void test_copy_and_set_zeros(sycl::queue &q)
{
  printf("\ngpu_copy_and_set_zeros:\n");
  const int TPB = 32;

  // l_rows=3, l_cols=2, matrixRows=3, istep=2
  // a_dev[0..8] = {1,2,3,4,5,6,7,8,9}
  // isOurProcessCol_prev=1: v_row[i] = a[i + 3*2] = a[i+6] => {7,8,9}
  // d_vec[0] = real(a[(l_rows-1)+matrixRows*(l_cols-1)]) = real(a[2+3*1]) = a[5] = 6
  const int l_rows=3, l_cols=2, matrixRows=3, istep=2;
  double ha[9] = {1,2,3,4,5,6,7,8,9};
  double zeros3[3] = {0,0,0};
  double zeros1[1] = {0};
  double hd[3] = {0,0,0};

  double *da    = sycl::malloc_device<double>(9, q);
  double *dv    = sycl::malloc_device<double>(3, q);
  double *du    = sycl::malloc_device<double>(3, q);
  double *daux1 = sycl::malloc_device<double>(1, q);
  double *dvav  = sycl::malloc_device<double>(1, q);
  double *dd    = sycl::malloc_device<double>(3, q);

  q.memcpy(da, ha, 9*sizeof(double)).wait();
  q.memcpy(dv, zeros3, 3*sizeof(double)).wait();
  q.memcpy(du, zeros3, 3*sizeof(double)).wait();
  q.memcpy(daux1, zeros1, sizeof(double)).wait();
  q.memcpy(dvav, zeros1, sizeof(double)).wait();
  q.memcpy(dd, hd, 3*sizeof(double)).wait();

  q.submit([&](sycl::handler &cgh){
    cgh.parallel_for(sycl::nd_range<1>(TPB, TPB), [=](sycl::nd_item<1> it){
      gpu_copy_and_set_zeros_kernel<double,double>(dv, du, da, daux1, dvav, dd,
          l_rows, l_cols, matrixRows, istep, 1, 1, 1, 0, 0, it);
    });
  }).wait();

  double hv[3], haux, hdvec[3];
  dev_read_n(q, hv, dv, 3);
  haux = dev_read(q, daux1);
  dev_read_n(q, hdvec, dd, 3);

  REPORT("copy_and_set_zeros: v_row[0]==7", neq(hv[0], 7.0));
  REPORT("copy_and_set_zeros: v_row[1]==8", neq(hv[1], 8.0));
  REPORT("copy_and_set_zeros: v_row[2]==9", neq(hv[2], 9.0));
  REPORT("copy_and_set_zeros: aux1[0]==0",  neq(haux, 0.0));
  REPORT("copy_and_set_zeros: d_vec[0]==6", neq(hdvec[0], 6.0));

  sycl::free(da,q); sycl::free(dv,q); sycl::free(du,q);
  sycl::free(daux1,q); sycl::free(dvav,q); sycl::free(dd,q);
}

// ============================================================
// Kernel 2: gpu_set_e_vec_scale_set_one_store_v_row (value overload)
// ============================================================

template <typename T, typename T_real>
static void gpu_set_e_vec_scale_set_one_store_v_row_kernel_val(
    T_real *e_vec_dev, T *vrl_dev, T *a_dev, T *v_row_dev, T *tau_dev, T xf_val,
    int l_rows, int l_cols, int matrixRows, int istep, int isOurProcessRow, int useCCL,
    sycl::nd_item<1> it)
{
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  if (useCCL && tid==0) {
    if (isOurProcessRow) e_vec_dev[istep-1-1] = elpaDeviceRealPart(*vrl_dev);
    v_row_dev[l_rows+1-1] = tau_dev[istep-1];
  }
  int index_global = tid;
  while (index_global < l_rows) {
    v_row_dev[index_global] = elpaDeviceMultiply(v_row_dev[index_global], xf_val);
    index_global += it.get_local_range(0) * it.get_group_range(0);
  }
  if (isOurProcessRow && (index_global - it.get_local_range(0)*it.get_group_range(0) == l_rows - 1))
    v_row_dev[l_rows-1] = elpaDeviceNumber<T>(1.0);
  int i_row = tid;
  while (i_row < l_rows) {
    a_dev[i_row + matrixRows*l_cols] = v_row_dev[i_row];
    i_row += it.get_local_range(0) * it.get_group_range(0);
  }
}

static void test_set_e_vec_scale_set_one_store_v_row(sycl::queue &q)
{
  printf("\ngpu_set_e_vec_scale_set_one_store_v_row:\n");
  const int TPB = 32;

  // l_rows=3, l_cols=2, matrixRows=3, istep=2, useCCL=0, isOurProcessRow=1, xf=0.5
  // v_row = {2,4,6}: after scale *0.5 = {1,2,3}; last element set to 1.0 -> {1,2,1}
  // a[i+3*2] = v_row[i]: a[6]=1, a[7]=2, a[8]=1
  const int l_rows=3, l_cols=2, matrixRows=3, istep=2;
  double hv[3] = {2,4,6};
  double ha[9] = {0,0,0,0,0,0,0,0,0};
  double htau[2] = {0,99};
  double he_vec[2] = {0,0};
  double hvrl = 0;

  double *dv    = sycl::malloc_device<double>(3, q);
  double *da    = sycl::malloc_device<double>(9, q);
  double *dtau  = sycl::malloc_device<double>(2, q);
  double *de    = sycl::malloc_device<double>(2, q);
  double *dvrl  = sycl::malloc_device<double>(1, q);

  q.memcpy(dv, hv, 3*sizeof(double)).wait();
  q.memcpy(da, ha, 9*sizeof(double)).wait();
  q.memcpy(dtau, htau, 2*sizeof(double)).wait();
  q.memcpy(de, he_vec, 2*sizeof(double)).wait();
  q.memcpy(dvrl, &hvrl, sizeof(double)).wait();

  double xf_val = 0.5;
  q.submit([&](sycl::handler &cgh){
    cgh.parallel_for(sycl::nd_range<1>(TPB, TPB), [=](sycl::nd_item<1> it){
      gpu_set_e_vec_scale_set_one_store_v_row_kernel_val<double,double>(
          de, dvrl, da, dv, dtau, xf_val,
          l_rows, l_cols, matrixRows, istep, 1, 0, it);
    });
  }).wait();

  double hv_out[3], ha_out[9];
  dev_read_n(q, hv_out, dv, 3);
  dev_read_n(q, ha_out, da, 9);

  REPORT("set_e_vec_scale: v_row[0]==1", neq(hv_out[0], 1.0));
  REPORT("set_e_vec_scale: v_row[1]==2", neq(hv_out[1], 2.0));
  REPORT("set_e_vec_scale: v_row[2]==1 (set to 1.0)", neq(hv_out[2], 1.0));
  REPORT("set_e_vec_scale: a[6]==1 (stored)", neq(ha_out[6], 1.0));
  REPORT("set_e_vec_scale: a[7]==2 (stored)", neq(ha_out[7], 2.0));
  REPORT("set_e_vec_scale: a[8]==1 (stored)", neq(ha_out[8], 1.0));

  sycl::free(dv,q); sycl::free(da,q); sycl::free(dtau,q);
  sycl::free(de,q); sycl::free(dvrl,q);
}

// ============================================================
// Kernel 3: gpu_store_u_v_in_uv_vu (value overload)
// ============================================================

template <typename T>
static void gpu_store_u_v_in_uv_vu_kernel_val(
    T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
    T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev,
    T vav_val, T tau_val,
    int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols,
    int istep, int useCCL, sycl::nd_item<1> it)
{
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  T conjg_tau = elpaDeviceComplexConjugate(tau_val);
  T conjg_tau_v_row_dev, conjg_tau_v_col_dev;
  if (useCCL && tid==0) tau_dev[istep-1] = v_row_dev[l_rows+1-1];
  T vav = vav_val;
  int i_row = tid;
  while (i_row < l_rows) {
    conjg_tau_v_row_dev = elpaDeviceMultiply(conjg_tau, v_row_dev[i_row]);
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+0)] = conjg_tau_v_row_dev;
    conjg_tau_v_row_dev = elpaDeviceMultiply(conjg_tau_v_row_dev, elpaDeviceNumber<T>(0.5));
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+1)] = elpaDeviceSubtract(elpaDeviceMultiply(conjg_tau_v_row_dev, vav), u_row_dev[i_row]);
    i_row += it.get_local_range(0) * it.get_group_range(0);
  }
  int i_col = tid;
  while (i_col < l_cols) {
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau, v_col_dev[i_col]);
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+1)] = conjg_tau_v_col_dev;
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau_v_col_dev, elpaDeviceNumber<T>(0.5));
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+0)] = elpaDeviceSubtract(elpaDeviceMultiply(conjg_tau_v_col_dev, vav), u_col_dev[i_col]);
    i_col += it.get_local_range(0) * it.get_group_range(0);
  }
}

static void test_store_u_v_in_uv_vu(sycl::queue &q)
{
  printf("\ngpu_store_u_v_in_uv_vu:\n");
  const int TPB = 32;

  // l_rows=2, l_cols=2, n_stored_vecs=1, max_local_rows=2, max_local_cols=2, istep=2
  // tau_val=1.0, vav_val=2.0
  // conjg_tau = conj(1.0) = 1.0
  // vu[i + 2*(2*1+0)] = vu[i+4] = conjg_tau*v_row[i] = v_row[i]
  // vu[i + 2*(2*1+1)] = vu[i+6] = 0.5*conjg_tau*vav*v_row[i] - u_row[i]
  // v_row={1,2}, u_row={3,4}, vav=2.0, tau=1.0:
  //   vu[4]=1, vu[5]=2
  //   vu[6] = 0.5*1.0*2.0*1 - 3 = -2, vu[7] = 0.5*1.0*2.0*2 - 4 = -2
  const int l_rows=2, l_cols=2, n_stored_vecs=1, max_local_rows=2, max_local_cols=2, istep=2;
  double hv_row[2]={1,2}, hu_row[2]={3,4};
  double hv_col[2]={5,6}, hu_col[2]={7,8};
  double htau[2]={0,0};
  double haux[4]={0,0,0,0};
  // vu_stored_rows size = max_local_rows*(2*n_stored_vecs+2) = 2*4 = 8
  // uv_stored_cols size = max_local_cols*(2*n_stored_vecs+2) = 2*4 = 8
  double zeros8[8]={0,0,0,0,0,0,0,0};

  double *dv_row = sycl::malloc_device<double>(2, q);
  double *du_row = sycl::malloc_device<double>(2, q);
  double *dv_col = sycl::malloc_device<double>(2, q);
  double *du_col = sycl::malloc_device<double>(2, q);
  double *dtau   = sycl::malloc_device<double>(2, q);
  double *daux   = sycl::malloc_device<double>(4, q);
  double *dvu    = sycl::malloc_device<double>(8, q);
  double *duv    = sycl::malloc_device<double>(8, q);

  q.memcpy(dv_row, hv_row, 2*sizeof(double)).wait();
  q.memcpy(du_row, hu_row, 2*sizeof(double)).wait();
  q.memcpy(dv_col, hv_col, 2*sizeof(double)).wait();
  q.memcpy(du_col, hu_col, 2*sizeof(double)).wait();
  q.memcpy(dtau, htau, 2*sizeof(double)).wait();
  q.memcpy(daux, haux, 4*sizeof(double)).wait();
  q.memcpy(dvu, zeros8, 8*sizeof(double)).wait();
  q.memcpy(duv, zeros8, 8*sizeof(double)).wait();

  double vav_val = 2.0, tau_val = 1.0;
  q.submit([&](sycl::handler &cgh){
    cgh.parallel_for(sycl::nd_range<1>(TPB, TPB), [=](sycl::nd_item<1> it){
      gpu_store_u_v_in_uv_vu_kernel_val<double>(
          dvu, duv, dv_row, du_row, dv_col, du_col, dtau, daux,
          vav_val, tau_val,
          l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, 0, it);
    });
  }).wait();

  double hvu[8];
  dev_read_n(q, hvu, dvu, 8);

  // vu[i + 2*(2*1+0)] = vu[i+4]: slot for conjg_tau*v_row
  REPORT("store_u_v_in_uv_vu: vu[4]==1 (conjg_tau*v_row[0])", neq(hvu[4], 1.0));
  REPORT("store_u_v_in_uv_vu: vu[5]==2 (conjg_tau*v_row[1])", neq(hvu[5], 2.0));
  // vu[i + 2*(2*1+1)] = vu[i+6]: 0.5*conjg_tau*vav*v_row[i] - u_row[i]
  REPORT("store_u_v_in_uv_vu: vu[6]==-2 (0.5*1*2*1-3)", neq(hvu[6], -2.0));
  REPORT("store_u_v_in_uv_vu: vu[7]==-2 (0.5*1*2*2-4)", neq(hvu[7], -2.0));

  sycl::free(dv_row,q); sycl::free(du_row,q);
  sycl::free(dv_col,q); sycl::free(du_col,q);
  sycl::free(dtau,q); sycl::free(daux,q);
  sycl::free(dvu,q); sycl::free(duv,q);
}

// ============================================================
// Kernel 4: gpu_update_matrix_element_add (uses local_buffer)
// ============================================================

template <typename T, typename T_real>
static void gpu_update_matrix_element_add_kernel(
    T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev,
    int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols,
    int istep, int n_stored_vecs, int isSkewsymmetric,
    sycl::nd_item<1> it, local_buffer<T> cache)
{
  int tid = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  if (n_stored_vecs > 0) {
    T temp = elpaDeviceNumber<T>(0.0);
    int index_n = tid;
    while (index_n < 2*n_stored_vecs) {
      temp = elpaDeviceAdd(temp, elpaDeviceMultiply(
          elpaDeviceComplexConjugate(vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n]),
          uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]));
      index_n += it.get_local_range(0)*it.get_group_range(0);
    }
    cache[it.get_local_id(0)] = temp;
    it.barrier();
    int i = it.get_local_range(0)/2;
    while (i > 0) {
      if (it.get_local_id(0) < i)
        cache[it.get_local_id(0)] = elpaDeviceAdd(cache[it.get_local_id(0)], cache[it.get_local_id(0)+i]);
      it.barrier();
      i /= 2;
    }
    if (it.get_local_id(0) == 0) {
      atomicAdd(&a_dev[(l_rows-1) + matrixRows*(l_cols-1)], cache[0]);
      if (!isSkewsymmetric)
        atomicAdd(&d_vec_dev[istep-1-1], elpaDeviceRealPart(cache[0]));
    }
  }
}

static void test_update_matrix_element_add(sycl::queue &q)
{
  printf("\ngpu_update_matrix_element_add:\n");
  const int TPB = 32;

  // n_stored_vecs=2, l_rows=2, l_cols=2, matrixRows=2, max_local_rows=2, max_local_cols=2
  // istep=2, isSkewsymmetric=0
  // 2*n_stored_vecs=4; index_n=0..3
  // vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n] = vu[1+2*index_n]: vu[1]=1, vu[3]=2, vu[5]=3, vu[7]=4
  // uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n] = uv[1+2*index_n]: uv[1]=1, uv[3]=2, uv[5]=3, uv[7]=4
  // dot = conj(1)*1 + conj(2)*2 + conj(3)*3 + conj(4)*4 = 1+4+9+16 = 30
  // a[1+2*1] = a[3] += 30 => a[3]=30; d_vec[istep-1-1=0] += 30 => d_vec[0]=30
  const int l_rows=2, l_cols=2, matrixRows=2, n_stored_vecs=2;
  const int max_local_rows=2, max_local_cols=2, istep=2;
  double hvu[8]={0,1,0,2,0,3,0,4}; // vu[1]=1, vu[3]=2, vu[5]=3, vu[7]=4
  double huv[8]={0,1,0,2,0,3,0,4}; // uv[1]=1, uv[3]=2, uv[5]=3, uv[7]=4
  double ha[4]={0,0,0,0};
  double hd[3]={0,0,0};

  double *dvu = sycl::malloc_device<double>(8, q);
  double *duv = sycl::malloc_device<double>(8, q);
  double *da  = sycl::malloc_device<double>(4, q);
  double *dd  = sycl::malloc_device<double>(3, q);

  q.memcpy(dvu, hvu, 8*sizeof(double)).wait();
  q.memcpy(duv, huv, 8*sizeof(double)).wait();
  q.memcpy(da, ha, 4*sizeof(double)).wait();
  q.memcpy(dd, hd, 3*sizeof(double)).wait();

  q.submit([&](sycl::handler &cgh){
    local_buffer<double> cache(sycl::range<1>(TPB), cgh);
    cgh.parallel_for(sycl::nd_range<1>(TPB, TPB), [=](sycl::nd_item<1> it){
      gpu_update_matrix_element_add_kernel<double,double>(
          dvu, duv, da, dd,
          l_rows, l_cols, matrixRows, max_local_rows, max_local_cols,
          istep, n_stored_vecs, 0, it, cache);
    });
  }).wait();

  double ha_out[4], hd_out[3];
  dev_read_n(q, ha_out, da, 4);
  dev_read_n(q, hd_out, dd, 3);

  REPORT("update_matrix_element_add: a[3]==30", neq(ha_out[3], 30.0));
  REPORT("update_matrix_element_add: d_vec[0]==30", neq(hd_out[0], 30.0));

  sycl::free(dvu,q); sycl::free(duv,q);
  sycl::free(da,q); sycl::free(dd,q);
}

// ============================================================
// Kernel 5: gpu_transpose_reduceadd_vectors_copy_block (3D, double only)
// ============================================================

static void gpu_transpose_reduceadd_vectors_copy_block_kernel_impl(
    double *aux_transpose_dev, double *vmat_st_dev,
    int nvc, int nvr, int n_block, int nblks_skip, int nblks_tot,
    int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st,
    int direction, int isSkewsymmetric, int isReduceadd,
    sycl::nd_item<3> it)
{
  int lc0 = it.get_group(1);
  int i = nblks_skip + n_block + it.get_group(2) * lcm_s_t;
  if (i > nblks_tot - 1) return;
  int nl = std::min(nvr - i * nblk, nblk);
  int k = (i - nblks_skip - n_block) / lcm_s_t * nblk + lc0 * auxstride;
  int ns_plus_lc0_ld_st = (i / np_st) * nblk + lc0 * ld_st;
  int j = it.get_local_id(0) + it.get_group(0) * it.get_local_range(0);
  if (j >= nl) return;
  if (direction == 1) {
    aux_transpose_dev[k + j] = vmat_st_dev[ns_plus_lc0_ld_st + j];
  }
  if (direction == 2 && !isReduceadd) {
    double sign = isSkewsymmetric ? -1.0 : 1.0;
    vmat_st_dev[ns_plus_lc0_ld_st + j] = sign * aux_transpose_dev[k + j];
  }
  if (direction == 2 && isReduceadd) {
    vmat_st_dev[ns_plus_lc0_ld_st + j] = vmat_st_dev[ns_plus_lc0_ld_st + j] + aux_transpose_dev[k + j];
  }
}

static void test_transpose_reduceadd_vectors_copy_block(sycl::queue &q)
{
  printf("\ngpu_transpose_reduceadd_vectors_copy_block:\n");
  // nblk=2, nvc=1, nvr=4, n_block=0, nblks_skip=0, nblks_tot=2, lcm_s_t=1
  // np_st=1, ld_st=4, auxstride=4
  // i=0: nl=min(4-0,2)=2; k=(0-0-0)/1*2+0*4=0; ns+lc0*ld_st=(0/1)*2+0*4=0
  //       aux[0]=vmat[0], aux[1]=vmat[1]
  // i=1: nl=min(4-2,2)=2; k=(1-0-0)/1*2+0*4=2; ns+lc0*ld_st=(1/1)*2+0*4=2
  //       aux[2]=vmat[2], aux[3]=vmat[3]
  const int nblk=2, nvc=1, nvr=4, n_block=0, nblks_skip=0, nblks_tot=2;
  const int lcm_s_t=1, auxstride=4, np_st=1, ld_st=4;
  const int threads=32;

  double hvmat[8]={1,2,3,4,5,6,7,8};
  double zeros4[4]={0,0,0,0};

  double *dvmat = sycl::malloc_device<double>(8, q);
  double *daux  = sycl::malloc_device<double>(8, q);

  q.memcpy(dvmat, hvmat, 8*sizeof(double)).wait();
  q.memcpy(daux, zeros4, 4*sizeof(double)).wait();
  q.memcpy(daux+4, zeros4, 4*sizeof(double)).wait();

  int i0 = nblks_skip + n_block;
  int num_i = (nblks_tot - 1 - i0) / lcm_s_t + 1; // = 2
  sycl::range<3> blocks((nblk+threads-1)/threads, nvc, num_i);
  sycl::range<3> local(threads, 1, 1);

  // direction=1: vmat -> aux
  q.submit([&](sycl::handler &cgh){
    cgh.parallel_for(sycl::nd_range<3>(blocks*local, local),
        [=](sycl::nd_item<3> it){
      gpu_transpose_reduceadd_vectors_copy_block_kernel_impl(
          daux, dvmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
          lcm_s_t, nblk, auxstride, np_st, ld_st, 1, 0, 0, it);
    });
  }).wait();

  double haux[8];
  dev_read_n(q, haux, daux, 8);

  REPORT("transpose_copy_block dir1: aux[0]==1", neq(haux[0], 1.0));
  REPORT("transpose_copy_block dir1: aux[1]==2", neq(haux[1], 2.0));
  REPORT("transpose_copy_block dir1: aux[2]==3", neq(haux[2], 3.0));
  REPORT("transpose_copy_block dir1: aux[3]==4", neq(haux[3], 4.0));

  sycl::free(dvmat,q); sycl::free(daux,q);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for tridiag kernels (SYCL) ===\n");

  sycl::queue q(sycl::default_selector_v);

  test_hh_transform<double>(q, "double");
  test_hh_transform<float> (q, "float");
  test_dot_product(q);
  test_dot_product_and_assign(q);
  test_copy_and_set_zeros(q);
  test_set_e_vec_scale_set_one_store_v_row(q);
  test_store_u_v_in_uv_vu(q);
  test_update_matrix_element_add(q);
  test_transpose_reduceadd_vectors_copy_block(q);

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
