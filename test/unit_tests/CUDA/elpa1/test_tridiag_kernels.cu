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

// Unit tests for kernels from src/elpa1/GPU/tridiag_gpu.h (CUDA backend).
// Covers: gpu_hh_transform, gpu_dot_product, gpu_dot_product_and_assign.

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
#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"
#include "../../../../src/elpa1/GPU/tridiag_gpu.h"

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
    printf("  %-70s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool neq (double a, double b, double tol=1e-10) { return fabs (a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }
#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
static bool half_neq(__half a, float ref, float tol=0.02f) {
    return fabsf(__half2float(a) - ref) < tol * (fabsf(ref) + 1.0f);
}
#endif

template <typename T>
static T dev_read(T *d) {
  T h;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// ============================================================
// gpu_hh_transform
// Computes Householder reflection coefficients from alpha and ||x||^2.
//
// Case A: xnorm_sq==0, alpha>=0  -> tau=0,  xf=0,  alpha unchanged
// Case B: xnorm_sq==0, alpha< 0  -> tau=2,  xf=0,  alpha=-alpha
// Case C: xnorm_sq >0            -> general formula; verified analytically below
//
// For Case C with alpha=3, xnorm_sq=16 (real):
//   beta = sqrt(9+16) = 5   (sign of alpha = +)
//   alpha += beta  -> alpha = 8
//   beta > 0, so:
//     alpha_r_new = xnorm_sq/8 = 2
//     tau = 2/5 = 0.4
//     alpha_dev = -2  (negated)
//     xf = 1/(-2) = -0.5
//     alpha_dev = beta = 5  (output is beta)
// ============================================================
template <typename T, typename Treal>
static void test_hh_transform(const char *tname, Treal tol)
{
  printf("\ngpu_hh_transform<%s>:\n", tname);

  T *alpha, *xnorm, *xf, *tau;
  CUDA_CHECK(cudaMalloc(&alpha, sizeof(T)));
  CUDA_CHECK(cudaMalloc(&xnorm, sizeof(T)));
  CUDA_CHECK(cudaMalloc(&xf,    sizeof(T)));
  CUDA_CHECK(cudaMalloc(&tau,   sizeof(T)));

  auto upload = [](T *d, double re, double im=0.0) {
    T h = elpaHostNumberFromInt<T>((int)0);
    // Set by constructing the value host-side via make
    // We write directly by reinterpreting as real pair
    double *p = (double *)&h;
    float  *fp = (float  *)&h;
    (void)p; (void)fp;
    h = elpaHostNumberFromInt<T>((int)re);  // approx for simple integers
    (void)im;
    CUDA_CHECK(cudaMemcpy(d, &h, sizeof(T), cudaMemcpyHostToDevice));
  };
  (void)upload;

  // Upload helper: scalar host->device
  auto up_scalar = [](T *d, T val) {
    CUDA_CHECK(cudaMemcpy(d, &val, sizeof(T), cudaMemcpyHostToDevice));
  };

  // --- Case A: xnorm_sq=0, alpha=4 (positive) -> tau=0 ---
  {
    T h_alpha = elpaHostNumberFromInt<T>(4);
    T h_zero  = elpaHostNumberFromInt<T>(0);
    up_scalar(alpha, h_alpha);
    up_scalar(xnorm, h_zero);
    gpu_hh_transform<T>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    T h_tau = dev_read(tau); T h_xf = dev_read(xf); T h_alpha_out = dev_read(alpha);
    bool ok_tau   = neq((double)h_tau, 0.0);
    bool ok_xf    = neq((double)h_xf,  0.0);
    bool ok_alpha = neq((double)h_alpha_out, 4.0);
    char lbl[128]; snprintf(lbl,sizeof(lbl),"%s Case A (xnorm=0,alpha>0): tau=0", tname);
    REPORT(lbl, ok_tau && ok_xf && ok_alpha);
  }

  // --- Case B: xnorm_sq=0, alpha=-4 (negative) -> tau=2, alpha=4 ---
  {
    T h_alpha = elpaHostNumberFromInt<T>(-4);
    T h_zero  = elpaHostNumberFromInt<T>(0);
    up_scalar(alpha, h_alpha);
    up_scalar(xnorm, h_zero);
    gpu_hh_transform<T>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    T h_tau = dev_read(tau); T h_xf = dev_read(xf); T h_alpha_out = dev_read(alpha);
    bool ok_tau   = neq((double)h_tau, 2.0);
    bool ok_xf    = neq((double)h_xf,  0.0);
    bool ok_alpha = neq((double)h_alpha_out, 4.0);
    char lbl[128]; snprintf(lbl,sizeof(lbl),"%s Case B (xnorm=0,alpha<0): tau=2,alpha=4", tname);
    REPORT(lbl, ok_tau && ok_xf && ok_alpha);
  }

  // --- Case C: alpha=3, xnorm_sq=16 -> alpha_out=5, tau=0.4, xf=-0.5 ---
  {
    T h_alpha = elpaHostNumberFromInt<T>(3);
    T h_xnorm = elpaHostNumberFromInt<T>(16);
    up_scalar(alpha, h_alpha);
    up_scalar(xnorm, h_xnorm);
    gpu_hh_transform<T>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    T h_tau = dev_read(tau); T h_xf = dev_read(xf); T h_alpha_out = dev_read(alpha);
    char lbl1[128], lbl2[128], lbl3[128];
    snprintf(lbl1,sizeof(lbl1),"%s Case C: alpha_out=5.0", tname);
    snprintf(lbl2,sizeof(lbl2),"%s Case C: tau_out=0.4",   tname);
    snprintf(lbl3,sizeof(lbl3),"%s Case C: xf_out=-0.5",   tname);
    REPORT(lbl1, neq((double)h_alpha_out, 5.0, 1e-6));
    REPORT(lbl2, neq((double)h_tau,       0.4, 1e-6));
    REPORT(lbl3, neq((double)h_xf,       -0.5, 1e-6));
  }

  CUDA_CHECK(cudaFree(alpha)); CUDA_CHECK(cudaFree(xnorm));
  CUDA_CHECK(cudaFree(xf));    CUDA_CHECK(cudaFree(tau));
}

// ============================================================
// gpu_dot_product
// result = sum_i conj(x[i*incx]) * y[i*incy],  result pre-zeroed.
// ============================================================
static void test_dot_product()
{
  printf("\ngpu_dot_product:\n");
  int SM = 1;

  // --- double: x=[1,2,3], y=[4,5,6], result=32 ---
  {
    double hx[] = {1,2,3}, hy[] = {4,5,6};
    double *dx,*dy,*dr;
    CUDA_CHECK(cudaMalloc(&dx, 3*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dy, 3*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dr, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dx, hx, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dr, 0, sizeof(double)));
    gpu_dot_product<double>(3, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    REPORT("double: dot([1,2,3],[4,5,6])==32", neq(dev_read(dr), 32.0));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dr));
  }

  // --- float: same values ---
  {
    float hx[] = {1,2,3}, hy[] = {4,5,6};
    float *dx,*dy,*dr;
    CUDA_CHECK(cudaMalloc(&dx, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dr, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx, 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dr, 0, sizeof(float)));
    gpu_dot_product<float>(3, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    REPORT("float: dot([1,2,3],[4,5,6])==32", neqf(dev_read(dr), 32.0f));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dr));
  }

  // --- double_complex: conj([1+1i,2+2i]) . [3+3i,4+4i]
  //     = (1-1i)(3+3i) + (2-2i)(4+4i)
  //     = (3+3i-3i+3) + (8+8i-8i+8)
  //     = 6 + 16 = 22+0i ---
  {
    gpuDoubleComplex hx[] = {make_gpuDoubleComplex(1,1), make_gpuDoubleComplex(2,2)};
    gpuDoubleComplex hy[] = {make_gpuDoubleComplex(3,3), make_gpuDoubleComplex(4,4)};
    gpuDoubleComplex *dx,*dy,*dr;
    CUDA_CHECK(cudaMalloc(&dx, 2*sizeof(gpuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&dy, 2*sizeof(gpuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&dr, sizeof(gpuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(dx, hx, 2*sizeof(gpuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, 2*sizeof(gpuDoubleComplex), cudaMemcpyHostToDevice));
    gpuDoubleComplex zero = make_gpuDoubleComplex(0,0);
    CUDA_CHECK(cudaMemcpy(dr, &zero, sizeof(gpuDoubleComplex), cudaMemcpyHostToDevice));
    gpu_dot_product<gpuDoubleComplex>(2, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuDoubleComplex h = dev_read(dr);
    REPORT("double_complex: conj(x).y real==22", neq(h.x, 22.0));
    REPORT("double_complex: conj(x).y imag==0",  neq(h.y,  0.0));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dr));
  }

  // --- float_complex: same pattern ---
  {
    gpuFloatComplex hx[] = {make_gpuFloatComplex(1,1), make_gpuFloatComplex(2,2)};
    gpuFloatComplex hy[] = {make_gpuFloatComplex(3,3), make_gpuFloatComplex(4,4)};
    gpuFloatComplex *dx,*dy,*dr;
    CUDA_CHECK(cudaMalloc(&dx, 2*sizeof(gpuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&dy, 2*sizeof(gpuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&dr, sizeof(gpuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(dx, hx, 2*sizeof(gpuFloatComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, 2*sizeof(gpuFloatComplex), cudaMemcpyHostToDevice));
    gpuFloatComplex zerof = make_gpuFloatComplex(0,0);
    CUDA_CHECK(cudaMemcpy(dr, &zerof, sizeof(gpuFloatComplex), cudaMemcpyHostToDevice));
    gpu_dot_product<gpuFloatComplex>(2, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuFloatComplex h = dev_read(dr);
    REPORT("float_complex: conj(x).y real==22", neqf(h.x, 22.0f));
    REPORT("float_complex: conj(x).y imag==0",  neqf(h.y,  0.0f));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dr));
  }

  // --- stride test: incx=2, incy=2; x=[1,_,2,_,3], y=[4,_,5,_,6] ---
  {
    double hx[] = {1,0,2,0,3}, hy[] = {4,0,5,0,6};
    double *dx,*dy,*dr;
    CUDA_CHECK(cudaMalloc(&dx, 5*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dy, 5*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dr, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dx, hx, 5*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, 5*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dr, 0, sizeof(double)));
    gpu_dot_product<double>(3, dx, 2, dy, 2, dr, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    REPORT("double: dot with stride 2 == 32", neq(dev_read(dr), 32.0));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dr));
  }
}

// ============================================================
// gpu_dot_product_and_assign
// v_row = [v0, v1, ..., v_{n-2}, v_{n-1}]
// isOurProcessRow=1: aux1[0] = dot(v[0..n-2], v[0..n-2]),  aux1[1] = v[n-1]
// isOurProcessRow=0: aux1[0] = dot(v[0..n-1], v[0..n-1]),  aux1[1] = 0
// ============================================================
static void test_dot_product_and_assign()
{
  printf("\ngpu_dot_product_and_assign:\n");
  int SM = 1;

  // v = [1,2,3,4], l_rows=4
  double hv[] = {1,2,3,4};
  double *dv, *aux;
  CUDA_CHECK(cudaMalloc(&dv,  4*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&aux, 2*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dv, hv, 4*sizeof(double), cudaMemcpyHostToDevice));

  // isOurProcessRow=1: aux[0]=1+4+9=14, aux[1]=4
  {
    CUDA_CHECK(cudaMemset(aux, 0, 2*sizeof(double)));
    gpu_dot_product_and_assign<double>(dv, 4, 1, aux, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ha[2]; CUDA_CHECK(cudaMemcpy(ha, aux, 2*sizeof(double), cudaMemcpyDeviceToHost));
    REPORT("double isOurProcessRow=1: aux[0]==14 (dot of first 3)", neq(ha[0], 14.0));
    REPORT("double isOurProcessRow=1: aux[1]==4  (last element)",   neq(ha[1],  4.0));
  }

  // isOurProcessRow=0: aux[0]=1+4+9+16=30, aux[1]=0
  {
    CUDA_CHECK(cudaMemset(aux, 0, 2*sizeof(double)));
    gpu_dot_product_and_assign<double>(dv, 4, 0, aux, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    double ha[2]; CUDA_CHECK(cudaMemcpy(ha, aux, 2*sizeof(double), cudaMemcpyDeviceToHost));
    REPORT("double isOurProcessRow=0: aux[0]==30 (dot of all 4)",   neq(ha[0], 30.0));
    REPORT("double isOurProcessRow=0: aux[1]==0  (set to zero)",     neq(ha[1],  0.0));
  }

  CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(aux));

  // float version
  float hvf[] = {1,2,3,4};
  float *dvf, *auxf;
  CUDA_CHECK(cudaMalloc(&dvf,  4*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&auxf, 2*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dvf, hvf, 4*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(auxf, 0, 2*sizeof(float)));
  gpu_dot_product_and_assign<float>(dvf, 4, 1, auxf, 0, SM, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());
  float haf[2]; CUDA_CHECK(cudaMemcpy(haf, auxf, 2*sizeof(float), cudaMemcpyDeviceToHost));
  REPORT("float isOurProcessRow=1: aux[0]==14", neqf(haf[0], 14.0f));
  REPORT("float isOurProcessRow=1: aux[1]==4",  neqf(haf[1],  4.0f));
  CUDA_CHECK(cudaFree(dvf)); CUDA_CHECK(cudaFree(auxf));
}

// ============================================================
// gpu_copy_and_set_zeros
// ============================================================
static void test_copy_and_set_zeros()
{
  printf("\ngpu_copy_and_set_zeros:\n");
  int SM = 1;

  // double: l_rows=3, l_cols=4, matrixRows=4, istep=3
  // isOurProcessCol_prev=1 -> v_row = a_dev column l_cols
  //   a[i + 4*4] for i=0..2 = a[16..18] = 17,18,19
  // d_vec[istep-2] = d_vec[1] = a[(l_rows-1)+matrixRows*(l_cols-1)] = a[14] = 15
  // l_cols>l_rows -> u_col[3] zeroed; useCCL=1 -> vav zeroed; aux1 zeroed
  {
    int l_rows=3, l_cols=4, matrixRows=4, istep=3;
    int a_sz = matrixRows*(l_cols+1); // 20
    double h_a[20]; for (int i=0; i<20; i++) h_a[i]=(double)(i+1);

    double *d_a,*d_v,*d_u,*d_aux1,*d_vav,*d_dvec;
    CUDA_CHECK(cudaMalloc(&d_a,    a_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v,    l_rows*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u,    l_cols*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_aux1, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vav,  sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dvec, istep*sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_sz*sizeof(double), cudaMemcpyHostToDevice));
    double fill=99.0;
    for (int i=0; i<l_cols; i++) CUDA_CHECK(cudaMemcpy(d_u+i, &fill, sizeof(double), cudaMemcpyHostToDevice));
    double ai=5.0, vi=7.0;
    CUDA_CHECK(cudaMemcpy(d_aux1, &ai, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vav,  &vi, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dvec, 0, istep*sizeof(double)));

    gpu_copy_and_set_zeros<double,double>(
        d_v, d_u, d_a, d_aux1, d_vav, d_dvec,
        l_rows, l_cols, matrixRows, istep,
        1,1,1/*isOurProcessRow/Col/ColPrev*/, 0/*isSkewsymmetric*/, 1/*useCCL*/, 0,SM,(gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hv[3]; CUDA_CHECK(cudaMemcpy(hv,d_v,3*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: v_row[0]=17", neq(hv[0],17.0));
    REPORT("double copy_and_set_zeros: v_row[1]=18", neq(hv[1],18.0));
    REPORT("double copy_and_set_zeros: v_row[2]=19", neq(hv[2],19.0));

    double hu3; CUDA_CHECK(cudaMemcpy(&hu3,d_u+3,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: u_col[3]=0 (l_cols>l_rows)", neq(hu3,0.0));

    double haux1; CUDA_CHECK(cudaMemcpy(&haux1,d_aux1,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: aux1=0", neq(haux1,0.0));

    double hvav; CUDA_CHECK(cudaMemcpy(&hvav,d_vav,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: vav=0 (useCCL=1)", neq(hvav,0.0));

    double hd; CUDA_CHECK(cudaMemcpy(&hd,d_dvec+1,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: d_vec[1]=15", neq(hd,15.0));

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_aux1)); CUDA_CHECK(cudaFree(d_vav)); CUDA_CHECK(cudaFree(d_dvec));
  }

  // float: isSkewsymmetric=1 -> d_vec[0] forced to 0 (not read from diagonal)
  {
    int l_rows=2, l_cols=2, matrixRows=2, istep=2;
    float h_a[6]; for (int i=0; i<6; i++) h_a[i]=(float)(i+1);

    float *d_a,*d_v,*d_u,*d_aux1,*d_vav,*d_dvec;
    CUDA_CHECK(cudaMalloc(&d_a,    6*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,    l_rows*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u,    l_cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_aux1, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vav,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dvec, istep*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, 6*sizeof(float), cudaMemcpyHostToDevice));
    float dv[2]={9.0f,9.0f}; CUDA_CHECK(cudaMemcpy(d_dvec,dv,2*sizeof(float),cudaMemcpyHostToDevice));
    float ax=3.0f; CUDA_CHECK(cudaMemcpy(d_aux1,&ax,sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vav,0,sizeof(float)));

    gpu_copy_and_set_zeros<float,float>(
        d_v, d_u, d_a, d_aux1, d_vav, d_dvec,
        l_rows, l_cols, matrixRows, istep,
        1,1,1, 1/*isSkewsymmetric*/, 0/*useCCL*/, 0,SM,(gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float hd0; CUDA_CHECK(cudaMemcpy(&hd0,d_dvec,sizeof(float),cudaMemcpyDeviceToHost));
    REPORT("float copy_and_set_zeros: d_vec[0]=0 (isSkewsymmetric)", neqf(hd0,0.0f));

    float haux1; CUDA_CHECK(cudaMemcpy(&haux1,d_aux1,sizeof(float),cudaMemcpyDeviceToHost));
    REPORT("float copy_and_set_zeros: aux1=0", neqf(haux1,0.0f));

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_aux1)); CUDA_CHECK(cudaFree(d_vav)); CUDA_CHECK(cudaFree(d_dvec));
  }
}

// ============================================================
// gpu_set_e_vec_scale_set_one_store_v_row
// ============================================================
static void test_set_e_vec_scale_set_one_store_v_row()
{
  printf("\ngpu_set_e_vec_scale_set_one_store_v_row:\n");

  // Sub-test A: useCCL=0, isOurProcessRow=1
  // v_row=[2,4,6,8], xf=3.0 -> v_row=[6,12,18,1], copied to a_dev column l_cols
  {
    int l_rows=4, l_cols=1, matrixRows=4, istep=2;
    double hv[4]={2.0,4.0,6.0,8.0};
    double ha[8]={}; double xf_val=3.0;

    double *d_v,*d_a,*d_e,*d_vrl,*d_tau,*d_xf;
    CUDA_CHECK(cudaMalloc(&d_v,   l_rows*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a,   8*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_e,   istep*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vrl, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau, istep*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xf,  sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_v, hv, 4*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, ha, 8*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xf, &xf_val, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_e,   0, istep*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_tau, 0, istep*sizeof(double)));

    gpu_set_e_vec_scale_set_one_store_v_row<double,double>(
        d_e, d_vrl, d_a, d_v, d_tau, d_xf,
        l_rows, l_cols, matrixRows, istep,
        1/*isOurProcessRow*/, 0/*useCCL*/, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hvo[4]; CUDA_CHECK(cudaMemcpy(hvo,d_v,4*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double set_e_vec: v_row[0]=6  (2*3)", neq(hvo[0],6.0));
    REPORT("double set_e_vec: v_row[1]=12 (4*3)", neq(hvo[1],12.0));
    REPORT("double set_e_vec: v_row[2]=18 (6*3)", neq(hvo[2],18.0));
    REPORT("double set_e_vec: v_row[l_rows-1]=1 (isOurProcessRow)", neq(hvo[3],1.0));

    double hao[8]; CUDA_CHECK(cudaMemcpy(hao,d_a,8*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double set_e_vec: a_dev[i+matrixRows*l_cols] stores v_row[0]=6",  neq(hao[4],6.0));
    REPORT("double set_e_vec: a_dev[i+matrixRows*l_cols] stores v_row[3]=1",  neq(hao[7],1.0));

    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_e));
    CUDA_CHECK(cudaFree(d_vrl)); CUDA_CHECK(cudaFree(d_tau)); CUDA_CHECK(cudaFree(d_xf));
  }

  // Sub-test B: useCCL=1, isOurProcessRow=1
  // v_row[l_rows] <- tau[istep-1]; e_vec[istep-2] <- real(vrl)
  {
    int l_rows=3, l_cols=0, matrixRows=3, istep=2;
    double hv[4]={1.0,2.0,3.0,0.0}; // extra slot at index l_rows for tau broadcast
    double xf_val=1.0, vrl_val=100.0;
    double htau[2]={0.0,42.0};

    double *d_v,*d_a,*d_e,*d_vrl,*d_tau,*d_xf;
    CUDA_CHECK(cudaMalloc(&d_v,   (l_rows+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a,   3*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_e,   istep*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vrl, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau, istep*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xf,  sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_v,   hv,       (l_rows+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, htau,     2*sizeof(double),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vrl, &vrl_val, sizeof(double),             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xf,  &xf_val,  sizeof(double),             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_e, 0, istep*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_a, 0, 3*sizeof(double)));

    gpu_set_e_vec_scale_set_one_store_v_row<double,double>(
        d_e, d_vrl, d_a, d_v, d_tau, d_xf,
        l_rows, l_cols, matrixRows, istep,
        1/*isOurProcessRow*/, 1/*useCCL*/, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hvo[4]; CUDA_CHECK(cudaMemcpy(hvo,d_v,(l_rows+1)*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double set_e_vec: v_row[l_rows]=tau[istep-1]=42 (useCCL)", neq(hvo[3],42.0));
    REPORT("double set_e_vec: v_row[l_rows-1]=1 (isOurProcessRow)",     neq(hvo[2],1.0));

    double he0; CUDA_CHECK(cudaMemcpy(&he0,d_e,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double set_e_vec: e_vec[0]=100 (useCCL,isOurProcessRow)", neq(he0,100.0));

    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_e));
    CUDA_CHECK(cudaFree(d_vrl)); CUDA_CHECK(cudaFree(d_tau)); CUDA_CHECK(cudaFree(d_xf));
  }
}

// ============================================================
// gpu_store_u_v_in_uv_vu
// tau=4, vav=3; v_row=[1,2], u_row=[1,2], v_col=[5,6], u_col=[3,4]
// conjg_tau = 4 (real), nsv=0, mlr=mlc=2
// vu[i+2*0] = 4*v_row[i]:        vu[0]=4,  vu[1]=8
// vu[i+2*1] = 0.5*4*v_row[i]*3 - u_row[i]: vu[2]=5, vu[3]=10
// uv[i+2*1] = 4*v_col[i]:        uv[2]=20, uv[3]=24
// uv[i+2*0] = 0.5*4*v_col[i]*3 - u_col[i]: uv[0]=27, uv[1]=32
// ============================================================
static void test_store_u_v_in_uv_vu()
{
  printf("\ngpu_store_u_v_in_uv_vu:\n");

  int l_rows=2, l_cols=2, n_stored_vecs=0, mlr=2, mlc=2, istep=1;
  double hv_row[2]={1.0,2.0}, hu_row[2]={1.0,2.0};
  double hv_col[2]={5.0,6.0}, hu_col[2]={3.0,4.0};
  double tau_val=4.0, vav_val=3.0;

  double *d_vu,*d_uv,*d_v_row,*d_u_row,*d_v_col,*d_u_col;
  double *d_tau_dev,*d_aux,*d_vav,*d_tau_vop;

  CUDA_CHECK(cudaMalloc(&d_vu,      mlr*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_uv,      mlc*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_v_row,   l_rows*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_row,   l_rows*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_v_col,   l_cols*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_col,   l_cols*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_tau_dev, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_aux,     sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_vav,     sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_tau_vop, sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_v_row, hv_row, 2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u_row, hu_row, 2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_col, hv_col, 2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u_col, hu_col, 2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vav,     &vav_val, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tau_vop, &tau_val, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_vu, 0, mlr*2*sizeof(double)));
  CUDA_CHECK(cudaMemset(d_uv, 0, mlc*2*sizeof(double)));

  gpu_store_u_v_in_uv_vu<double>(
      d_vu, d_uv, d_v_row, d_u_row, d_v_col, d_u_col,
      d_tau_dev, d_aux, d_vav, d_tau_vop,
      l_rows, l_cols, n_stored_vecs, mlr, mlc,
      istep, 0/*useCCL*/, 0, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  double hvu[4]; CUDA_CHECK(cudaMemcpy(hvu,d_vu,4*sizeof(double),cudaMemcpyDeviceToHost));
  REPORT("double store_u_v_in_uv_vu: vu[0]=4  (conjg_tau*v_row[0])",      neq(hvu[0], 4.0));
  REPORT("double store_u_v_in_uv_vu: vu[1]=8  (conjg_tau*v_row[1])",      neq(hvu[1], 8.0));
  REPORT("double store_u_v_in_uv_vu: vu[2]=5  (0.5*tau*vrow[0]*vav-urow)", neq(hvu[2], 5.0));
  REPORT("double store_u_v_in_uv_vu: vu[3]=10 (0.5*tau*vrow[1]*vav-urow)", neq(hvu[3],10.0));

  double huv[4]; CUDA_CHECK(cudaMemcpy(huv,d_uv,4*sizeof(double),cudaMemcpyDeviceToHost));
  REPORT("double store_u_v_in_uv_vu: uv[2]=20 (conjg_tau*v_col[0])",       neq(huv[2],20.0));
  REPORT("double store_u_v_in_uv_vu: uv[3]=24 (conjg_tau*v_col[1])",       neq(huv[3],24.0));
  REPORT("double store_u_v_in_uv_vu: uv[0]=27 (0.5*tau*vcol[0]*vav-ucol)", neq(huv[0],27.0));
  REPORT("double store_u_v_in_uv_vu: uv[1]=32 (0.5*tau*vcol[1]*vav-ucol)", neq(huv[1],32.0));

  CUDA_CHECK(cudaFree(d_vu)); CUDA_CHECK(cudaFree(d_uv));
  CUDA_CHECK(cudaFree(d_v_row)); CUDA_CHECK(cudaFree(d_u_row));
  CUDA_CHECK(cudaFree(d_v_col)); CUDA_CHECK(cudaFree(d_u_col));
  CUDA_CHECK(cudaFree(d_tau_dev)); CUDA_CHECK(cudaFree(d_aux));
  CUDA_CHECK(cudaFree(d_vav)); CUDA_CHECK(cudaFree(d_tau_vop));
}

// ============================================================
// gpu_update_matrix_element_add
// ============================================================
static void test_update_matrix_element_add()
{
  printf("\ngpu_update_matrix_element_add:\n");
  int SM = 1;

  // double: n_stored_vecs=2, l_rows=3, l_cols=3, mlr=mlc=matrixRows=3, istep=3
  // vu[(l_rows-1)+mlr*n] = vu[2+3*n]: 1,2,3,4 for n=0..3
  // uv[(l_cols-1)+mlc*n] = uv[2+3*n]: 5,3,2,1 for n=0..3
  // dot = 1*5+2*3+3*2+4*1 = 21
  // a[8] initial=10 -> 31;  d_vec[1] initial=0 -> 21
  {
    int l_rows=3,l_cols=3,matrixRows=3,mlr=3,mlc=3,istep=3,nsv=2;
    double h_vu[12]={}, h_uv[12]={}, h_a[9]={}, h_d[3]={};
    h_vu[2]=1.0; h_vu[5]=2.0; h_vu[8]=3.0; h_vu[11]=4.0;
    h_uv[2]=5.0; h_uv[5]=3.0; h_uv[8]=2.0; h_uv[11]=1.0;
    h_a[8]=10.0;

    double *d_vu,*d_uv,*d_a,*d_d;
    CUDA_CHECK(cudaMalloc(&d_vu,12*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uv,12*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a,  9*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d,  3*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vu,h_vu,12*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv,h_uv,12*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a,  9*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, h_d,  3*sizeof(double),cudaMemcpyHostToDevice));

    gpu_update_matrix_element_add<double,double>(
        d_vu, d_uv, d_a, d_d,
        l_rows, l_cols, matrixRows, mlr, mlc, istep, nsv,
        0/*isSkewsymmetric*/, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double ha8; CUDA_CHECK(cudaMemcpy(&ha8,d_a+8,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double update_matrix_element: a[8]=31 (10+dot21)", neq(ha8,31.0));

    double hd1; CUDA_CHECK(cudaMemcpy(&hd1,d_d+1,sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double update_matrix_element: d_vec[1]=21 (dot product)", neq(hd1,21.0));

    CUDA_CHECK(cudaFree(d_vu)); CUDA_CHECK(cudaFree(d_uv));
    CUDA_CHECK(cudaFree(d_a));  CUDA_CHECK(cudaFree(d_d));
  }

  // float: isSkewsymmetric=1 -> d_vec unchanged
  // n_stored_vecs=1, l_rows=l_cols=mlr=mlc=matrixRows=2, istep=2
  // vu[1+2*n] n=0,1: 3,4; uv[1+2*n]: 2,1; dot=3*2+4*1=10
  // a[3] initial=1 -> 11;  d_vec[0] initial=5, isSkewsymmetric -> stays 5
  {
    int l_rows=2,l_cols=2,matrixRows=2,mlr=2,mlc=2,istep=2,nsv=1;
    float h_vu[4]={},h_uv[4]={},h_a[4]={},h_d[2]={5.0f,5.0f};
    h_vu[1]=3.0f; h_vu[3]=4.0f;
    h_uv[1]=2.0f; h_uv[3]=1.0f;
    h_a[3]=1.0f;

    float *d_vu,*d_uv,*d_a,*d_d;
    CUDA_CHECK(cudaMalloc(&d_vu,4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uv,4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, 4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d, 2*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vu,h_vu,4*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv,h_uv,4*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, 4*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, h_d, 2*sizeof(float),cudaMemcpyHostToDevice));

    gpu_update_matrix_element_add<float,float>(
        d_vu, d_uv, d_a, d_d,
        l_rows, l_cols, matrixRows, mlr, mlc, istep, nsv,
        1/*isSkewsymmetric*/, 0, SM, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ha3; CUDA_CHECK(cudaMemcpy(&ha3,d_a+3,sizeof(float),cudaMemcpyDeviceToHost));
    REPORT("float update_matrix_element: a[3]=11 (1+dot10)", neqf(ha3,11.0f));

    float hd0; CUDA_CHECK(cudaMemcpy(&hd0,d_d,sizeof(float),cudaMemcpyDeviceToHost));
    REPORT("float update_matrix_element: d_vec[0]=5 unchanged (isSkewsymmetric)", neqf(hd0,5.0f));

    CUDA_CHECK(cudaFree(d_vu)); CUDA_CHECK(cudaFree(d_uv));
    CUDA_CHECK(cudaFree(d_a));  CUDA_CHECK(cudaFree(d_d));
  }
}

// ============================================================
// gpu_transpose_reduceadd_vectors_copy_block
// nvc=1, nvr=6, nblks_tot=3, nblk=2, lcm_s_t=1, np_st=1, ld_st=6, auxstride=6
// num_i=3; for i=0,1,2: nl=2, k=0,2,4, ns=0,2,4  (k==ns in this config)
// Test A direction=1: vmat -> aux (copy)
// Test B direction=2, isSkewsymmetric=1: aux -> -vmat
// Test C direction=2, isReduceadd=1:     vmat += aux
// ============================================================
static void test_transpose_reduceadd_vectors_copy_block()
{
  printf("\ngpu_transpose_reduceadd_vectors_copy_block:\n");
  int SM = 1;
  int nvc=1, nvr=6, n_block=0, nblks_skip=0, nblks_tot=3;
  int lcm_s_t=1, nblk=2, auxstride=6, np_st=1, ld_st=6, N=6;

  // Test A: direction=1
  {
    double h_vmat[6]={10,20,30,40,50,60};
    double *d_aux, *d_vmat;
    CUDA_CHECK(cudaMalloc(&d_aux,  N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vmat, N*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_aux, 0, N*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vmat, h_vmat, N*sizeof(double), cudaMemcpyHostToDevice));

    gpu_transpose_reduceadd_vectors_copy_block<double>(
        d_aux, d_vmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
        lcm_s_t, nblk, auxstride, np_st, ld_st,
        1/*direction*/, 0,0, 0,SM,(gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double haux[6]; CUDA_CHECK(cudaMemcpy(haux,d_aux,6*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double transpose dir=1: aux[0]=10", neq(haux[0],10.0));
    REPORT("double transpose dir=1: aux[2]=30", neq(haux[2],30.0));
    REPORT("double transpose dir=1: aux[5]=60", neq(haux[5],60.0));

    CUDA_CHECK(cudaFree(d_aux)); CUDA_CHECK(cudaFree(d_vmat));
  }

  // Test B: direction=2, isSkewsymmetric=1 -> vmat = -aux
  {
    double h_aux[6]={1,2,3,4,5,6};
    double *d_aux, *d_vmat;
    CUDA_CHECK(cudaMalloc(&d_aux,  N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vmat, N*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux, N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vmat, 0, N*sizeof(double)));

    gpu_transpose_reduceadd_vectors_copy_block<double>(
        d_aux, d_vmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
        lcm_s_t, nblk, auxstride, np_st, ld_st,
        2/*direction*/, 1/*isSkewsymmetric*/, 0, 0,SM,(gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hv[6]; CUDA_CHECK(cudaMemcpy(hv,d_vmat,6*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double transpose dir=2 skew: vmat[0]=-1", neq(hv[0],-1.0));
    REPORT("double transpose dir=2 skew: vmat[3]=-4", neq(hv[3],-4.0));

    CUDA_CHECK(cudaFree(d_aux)); CUDA_CHECK(cudaFree(d_vmat));
  }

  // Test C: direction=2, isReduceadd=1 -> vmat += aux
  {
    double h_aux[6]={1,2,3,4,5,6}, h_vmat[6]={10,20,30,40,50,60};
    double *d_aux, *d_vmat;
    CUDA_CHECK(cudaMalloc(&d_aux,  N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vmat, N*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_aux,  h_aux,  N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vmat, h_vmat, N*sizeof(double), cudaMemcpyHostToDevice));

    gpu_transpose_reduceadd_vectors_copy_block<double>(
        d_aux, d_vmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
        lcm_s_t, nblk, auxstride, np_st, ld_st,
        2/*direction*/, 0, 1/*isReduceadd*/, 0,SM,(gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double hv[6]; CUDA_CHECK(cudaMemcpy(hv,d_vmat,6*sizeof(double),cudaMemcpyDeviceToHost));
    REPORT("double transpose dir=2 reduceadd: vmat[0]=11", neq(hv[0],11.0));
    REPORT("double transpose dir=2 reduceadd: vmat[5]=66", neq(hv[5],66.0));

    CUDA_CHECK(cudaFree(d_aux)); CUDA_CHECK(cudaFree(d_vmat));
  }
}

#ifdef WANT_HALF_PRECISION_REAL
// ============================================================
// gpu_hh_transform<__half>
// Same cases as the double/float template, but using __float2half for
// initialization and __half2float for comparison.
// ============================================================
static void test_hh_transform_half()
{
  printf("\ngpu_hh_transform<__half>:\n");

  __half *alpha, *xnorm, *xf, *tau;
  CUDA_CHECK(cudaMalloc(&alpha, sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&xnorm, sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&xf,    sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&tau,   sizeof(__half)));

  auto up = [](void *d, float v) {
    __half h = __float2half(v);
    CUDA_CHECK(cudaMemcpy(d, &h, sizeof(__half), cudaMemcpyHostToDevice));
  };

  // --- Case A: xnorm_sq=0, alpha=4 (positive) -> tau=0 ---
  {
    up(alpha, 4.0f); up(xnorm, 0.0f);
    gpu_hh_transform<__half>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    __half h_tau, h_xf, h_alpha_out;
    CUDA_CHECK(cudaMemcpy(&h_tau,       tau,   sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_xf,        xf,    sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_alpha_out, alpha, sizeof(__half), cudaMemcpyDeviceToHost));
    REPORT("__half Case A (xnorm=0,alpha>0): tau=0",
           half_neq(h_tau, 0.0f) && half_neq(h_xf, 0.0f) && half_neq(h_alpha_out, 4.0f));
  }

  // --- Case B: xnorm_sq=0, alpha=-4 (negative) -> tau=2, alpha=4 ---
  {
    up(alpha, -4.0f); up(xnorm, 0.0f);
    gpu_hh_transform<__half>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    __half h_tau, h_xf, h_alpha_out;
    CUDA_CHECK(cudaMemcpy(&h_tau,       tau,   sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_xf,        xf,    sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_alpha_out, alpha, sizeof(__half), cudaMemcpyDeviceToHost));
    REPORT("__half Case B (xnorm=0,alpha<0): tau=2,alpha=4",
           half_neq(h_tau, 2.0f) && half_neq(h_xf, 0.0f) && half_neq(h_alpha_out, 4.0f));
  }

  // --- Case C: alpha=3, xnorm_sq=16 -> alpha_out=5, tau=0.4, xf=-0.5 ---
  {
    up(alpha, 3.0f); up(xnorm, 16.0f);
    gpu_hh_transform<__half>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    __half h_tau, h_xf, h_alpha_out;
    CUDA_CHECK(cudaMemcpy(&h_tau,       tau,   sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_xf,        xf,    sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_alpha_out, alpha, sizeof(__half), cudaMemcpyDeviceToHost));
    REPORT("__half Case C: alpha_out=5.0", half_neq(h_alpha_out, 5.0f));
    REPORT("__half Case C: tau_out=0.4",   half_neq(h_tau,       0.4f));
    REPORT("__half Case C: xf_out=-0.5",   half_neq(h_xf,       -0.5f));
  }

  CUDA_CHECK(cudaFree(alpha)); CUDA_CHECK(cudaFree(xnorm));
  CUDA_CHECK(cudaFree(xf));    CUDA_CHECK(cudaFree(tau));
}

// ============================================================
// gpu_dot_product<__half>: x=[1,2,3], y=[4,5,6], result=32
// ============================================================
static void test_dot_product_half()
{
  printf("\ngpu_dot_product<__half>:\n");
  int SM = 1;

  __half hx[3] = {__float2half(1.0f), __float2half(2.0f), __float2half(3.0f)};
  __half hy[3] = {__float2half(4.0f), __float2half(5.0f), __float2half(6.0f)};
  __half zero  = __float2half(0.0f);

  __half *dx, *dy, *dr;
  CUDA_CHECK(cudaMalloc(&dx, 3*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dy, 3*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dr, sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(dx, hx, 3*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, hy, 3*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dr, &zero, sizeof(__half), cudaMemcpyHostToDevice));

  gpu_dot_product<__half>(3, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  __half h_result;
  CUDA_CHECK(cudaMemcpy(&h_result, dr, sizeof(__half), cudaMemcpyDeviceToHost));
  REPORT("__half: dot([1,2,3],[4,5,6])==32", half_neq(h_result, 32.0f));

  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dr));
}

// ============================================================
// gpu_dot_product_and_assign<__half>: v=[1,2,3,4]
// isOurProcessRow=1 -> aux[0]=14, aux[1]=4
// ============================================================
static void test_dot_product_and_assign_half()
{
  printf("\ngpu_dot_product_and_assign<__half>:\n");
  int SM = 1;

  __half hv[4] = {__float2half(1.0f), __float2half(2.0f),
                  __float2half(3.0f), __float2half(4.0f)};
  __half zero  = __float2half(0.0f);

  __half *dv, *aux;
  CUDA_CHECK(cudaMalloc(&dv,  4*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&aux, 2*sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(dv, hv, 4*sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(aux,   &zero, sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(aux+1, &zero, sizeof(__half), cudaMemcpyHostToDevice));

  gpu_dot_product_and_assign<__half>(dv, 4, 1, aux, 0, SM, (gpuStream_t)0);
  CUDA_CHECK(cudaDeviceSynchronize());

  __half ha[2];
  CUDA_CHECK(cudaMemcpy(ha, aux, 2*sizeof(__half), cudaMemcpyDeviceToHost));
  REPORT("__half dot_product_and_assign: aux[0]==14 (dot first 3)", half_neq(ha[0], 14.0f));
  REPORT("__half dot_product_and_assign: aux[1]==4  (last element)", half_neq(ha[1],  4.0f));

  CUDA_CHECK(cudaFree(dv)); CUDA_CHECK(cudaFree(aux));
}
#endif /* WANT_HALF_PRECISION_REAL */

// ============================================================
int main(void)
{
  printf("=== Unit tests for tridiag_gpu.h kernels (CUDA) ===\n");

  test_hh_transform<double,double>("double", 1e-10);
  test_hh_transform<float,float>  ("float",  1e-5f);
  test_dot_product();
  test_dot_product_and_assign();
  test_copy_and_set_zeros();
  test_set_e_vec_scale_set_one_store_v_row();
  test_store_u_v_in_uv_vu();
  test_update_matrix_element_add();
  test_transpose_reduceadd_vectors_copy_block();

#ifdef WANT_HALF_PRECISION_REAL
  test_hh_transform_half();
  test_dot_product_half();
  test_dot_product_and_assign_half();
#endif /* WANT_HALF_PRECISION_REAL */

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
