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

// Unit tests for kernels from src/elpa1/GPU/tridiag_gpu.h.
// AMD ROCm/HIP port of test/unit_tests/CUDA/test_tridiag_kernels.cu.
// Covers: gpu_hh_transform, gpu_dot_product, gpu_dot_product_and_assign.

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
#include "../../../../src/elpa1/GPU/tridiag_gpu.h"

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
    printf("  %-70s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool neq (double a, double b, double tol=1e-10) { return fabs(a-b) < tol; }
static bool neqf(float  a, float  b, float  tol=1e-5f) { return fabsf(a-b) < tol; }

template <typename T>
static T dev_read(T *d) {
  T h;
  HIP_CHECK(hipMemcpy(&h, d, sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

// ============================================================
// gpu_hh_transform
// ============================================================
template <typename T, typename Treal>
static void test_hh_transform(const char *tname, Treal tol)
{
  printf("\ngpu_hh_transform<%s>:\n", tname);

  T *alpha, *xnorm, *xf, *tau;
  HIP_CHECK(hipMalloc(&alpha, sizeof(T)));
  HIP_CHECK(hipMalloc(&xnorm, sizeof(T)));
  HIP_CHECK(hipMalloc(&xf,    sizeof(T)));
  HIP_CHECK(hipMalloc(&tau,   sizeof(T)));

  auto up_scalar = [](T *d, T val) {
    HIP_CHECK(hipMemcpy(d, &val, sizeof(T), hipMemcpyHostToDevice));
  };

  // --- Case A: xnorm_sq=0, alpha=4 (positive) -> tau=0 ---
  {
    T h_alpha = elpaHostNumberFromInt<T>(4);
    T h_zero  = elpaHostNumberFromInt<T>(0);
    up_scalar(alpha, h_alpha);
    up_scalar(xnorm, h_zero);
    gpu_hh_transform<T>(alpha, xnorm, xf, tau, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
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
    HIP_CHECK(hipDeviceSynchronize());
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
    HIP_CHECK(hipDeviceSynchronize());
    T h_tau = dev_read(tau); T h_xf = dev_read(xf); T h_alpha_out = dev_read(alpha);
    char lbl1[128], lbl2[128], lbl3[128];
    snprintf(lbl1,sizeof(lbl1),"%s Case C: alpha_out=5.0", tname);
    snprintf(lbl2,sizeof(lbl2),"%s Case C: tau_out=0.4",   tname);
    snprintf(lbl3,sizeof(lbl3),"%s Case C: xf_out=-0.5",   tname);
    REPORT(lbl1, neq((double)h_alpha_out, 5.0, 1e-6));
    REPORT(lbl2, neq((double)h_tau,       0.4, 1e-6));
    REPORT(lbl3, neq((double)h_xf,       -0.5, 1e-6));
  }

  HIP_CHECK(hipFree(alpha)); HIP_CHECK(hipFree(xnorm));
  HIP_CHECK(hipFree(xf));    HIP_CHECK(hipFree(tau));
}

// ============================================================
// gpu_dot_product
// ============================================================
static void test_dot_product()
{
  printf("\ngpu_dot_product:\n");
  int SM = 1;

  // double: x=[1,2,3], y=[4,5,6], result=32
  {
    double hx[] = {1,2,3}, hy[] = {4,5,6};
    double *dx,*dy,*dr;
    HIP_CHECK(hipMalloc(&dx, 3*sizeof(double)));
    HIP_CHECK(hipMalloc(&dy, 3*sizeof(double)));
    HIP_CHECK(hipMalloc(&dr, sizeof(double)));
    HIP_CHECK(hipMemcpy(dx, hx, 3*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, 3*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dr, 0, sizeof(double)));
    gpu_dot_product<double>(3, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    REPORT("double: dot([1,2,3],[4,5,6])==32", neq(dev_read(dr), 32.0));
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dr));
  }

  // float: same values
  {
    float hx[] = {1,2,3}, hy[] = {4,5,6};
    float *dx,*dy,*dr;
    HIP_CHECK(hipMalloc(&dx, 3*sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, 3*sizeof(float)));
    HIP_CHECK(hipMalloc(&dr, sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, hx, 3*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, 3*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dr, 0, sizeof(float)));
    gpu_dot_product<float>(3, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    REPORT("float: dot([1,2,3],[4,5,6])==32", neqf(dev_read(dr), 32.0f));
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dr));
  }

  // double_complex: conj([1+1i,2+2i]) . [3+3i,4+4i] = 22+0i
  {
    gpuDoubleComplex hx[] = {make_gpuDoubleComplex(1,1), make_gpuDoubleComplex(2,2)};
    gpuDoubleComplex hy[] = {make_gpuDoubleComplex(3,3), make_gpuDoubleComplex(4,4)};
    gpuDoubleComplex *dx,*dy,*dr;
    HIP_CHECK(hipMalloc(&dx, 2*sizeof(gpuDoubleComplex)));
    HIP_CHECK(hipMalloc(&dy, 2*sizeof(gpuDoubleComplex)));
    HIP_CHECK(hipMalloc(&dr, sizeof(gpuDoubleComplex)));
    HIP_CHECK(hipMemcpy(dx, hx, 2*sizeof(gpuDoubleComplex), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, 2*sizeof(gpuDoubleComplex), hipMemcpyHostToDevice));
    gpuDoubleComplex zero = make_gpuDoubleComplex(0,0);
    HIP_CHECK(hipMemcpy(dr, &zero, sizeof(gpuDoubleComplex), hipMemcpyHostToDevice));
    gpu_dot_product<gpuDoubleComplex>(2, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    gpuDoubleComplex h = dev_read(dr);
    REPORT("double_complex: conj(x).y real==22", neq(h.x, 22.0));
    REPORT("double_complex: conj(x).y imag==0",  neq(h.y,  0.0));
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dr));
  }

  // float_complex: same pattern
  {
    gpuFloatComplex hx[] = {make_gpuFloatComplex(1,1), make_gpuFloatComplex(2,2)};
    gpuFloatComplex hy[] = {make_gpuFloatComplex(3,3), make_gpuFloatComplex(4,4)};
    gpuFloatComplex *dx,*dy,*dr;
    HIP_CHECK(hipMalloc(&dx, 2*sizeof(gpuFloatComplex)));
    HIP_CHECK(hipMalloc(&dy, 2*sizeof(gpuFloatComplex)));
    HIP_CHECK(hipMalloc(&dr, sizeof(gpuFloatComplex)));
    HIP_CHECK(hipMemcpy(dx, hx, 2*sizeof(gpuFloatComplex), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, 2*sizeof(gpuFloatComplex), hipMemcpyHostToDevice));
    gpuFloatComplex zerof = make_gpuFloatComplex(0,0);
    HIP_CHECK(hipMemcpy(dr, &zerof, sizeof(gpuFloatComplex), hipMemcpyHostToDevice));
    gpu_dot_product<gpuFloatComplex>(2, dx, 1, dy, 1, dr, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    gpuFloatComplex h = dev_read(dr);
    REPORT("float_complex: conj(x).y real==22", neqf(h.x, 22.0f));
    REPORT("float_complex: conj(x).y imag==0",  neqf(h.y,  0.0f));
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dr));
  }

  // stride test: incx=2, incy=2
  {
    double hx[] = {1,0,2,0,3}, hy[] = {4,0,5,0,6};
    double *dx,*dy,*dr;
    HIP_CHECK(hipMalloc(&dx, 5*sizeof(double)));
    HIP_CHECK(hipMalloc(&dy, 5*sizeof(double)));
    HIP_CHECK(hipMalloc(&dr, sizeof(double)));
    HIP_CHECK(hipMemcpy(dx, hx, 5*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, 5*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dr, 0, sizeof(double)));
    gpu_dot_product<double>(3, dx, 2, dy, 2, dr, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    REPORT("double: dot with stride 2 == 32", neq(dev_read(dr), 32.0));
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dr));
  }
}

// ============================================================
// gpu_dot_product_and_assign
// ============================================================
static void test_dot_product_and_assign()
{
  printf("\ngpu_dot_product_and_assign:\n");
  int SM = 1;

  double hv[] = {1,2,3,4};
  double *dv, *aux;
  HIP_CHECK(hipMalloc(&dv,  4*sizeof(double)));
  HIP_CHECK(hipMalloc(&aux, 2*sizeof(double)));
  HIP_CHECK(hipMemcpy(dv, hv, 4*sizeof(double), hipMemcpyHostToDevice));

  // isOurProcessRow=1: aux[0]=1+4+9=14, aux[1]=4
  {
    HIP_CHECK(hipMemset(aux, 0, 2*sizeof(double)));
    gpu_dot_product_and_assign<double>(dv, 4, 1, aux, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    double ha[2]; HIP_CHECK(hipMemcpy(ha, aux, 2*sizeof(double), hipMemcpyDeviceToHost));
    REPORT("double isOurProcessRow=1: aux[0]==14 (dot of first 3)", neq(ha[0], 14.0));
    REPORT("double isOurProcessRow=1: aux[1]==4  (last element)",   neq(ha[1],  4.0));
  }

  // isOurProcessRow=0: aux[0]=1+4+9+16=30, aux[1]=0
  {
    HIP_CHECK(hipMemset(aux, 0, 2*sizeof(double)));
    gpu_dot_product_and_assign<double>(dv, 4, 0, aux, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());
    double ha[2]; HIP_CHECK(hipMemcpy(ha, aux, 2*sizeof(double), hipMemcpyDeviceToHost));
    REPORT("double isOurProcessRow=0: aux[0]==30 (dot of all 4)",   neq(ha[0], 30.0));
    REPORT("double isOurProcessRow=0: aux[1]==0  (set to zero)",     neq(ha[1],  0.0));
  }

  HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(aux));

  float hvf[] = {1,2,3,4};
  float *dvf, *auxf;
  HIP_CHECK(hipMalloc(&dvf,  4*sizeof(float)));
  HIP_CHECK(hipMalloc(&auxf, 2*sizeof(float)));
  HIP_CHECK(hipMemcpy(dvf, hvf, 4*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(auxf, 0, 2*sizeof(float)));
  gpu_dot_product_and_assign<float>(dvf, 4, 1, auxf, 0, SM, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());
  float haf[2]; HIP_CHECK(hipMemcpy(haf, auxf, 2*sizeof(float), hipMemcpyDeviceToHost));
  REPORT("float isOurProcessRow=1: aux[0]==14", neqf(haf[0], 14.0f));
  REPORT("float isOurProcessRow=1: aux[1]==4",  neqf(haf[1],  4.0f));
  HIP_CHECK(hipFree(dvf)); HIP_CHECK(hipFree(auxf));
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
    HIP_CHECK(hipMalloc(&d_a,    a_sz*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_v,    l_rows*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_u,    l_cols*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_aux1, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vav,  sizeof(double)));
    HIP_CHECK(hipMalloc(&d_dvec, istep*sizeof(double)));

    HIP_CHECK(hipMemcpy(d_a, h_a, a_sz*sizeof(double), hipMemcpyHostToDevice));
    double fill=99.0;
    for (int i=0; i<l_cols; i++) HIP_CHECK(hipMemcpy(d_u+i, &fill, sizeof(double), hipMemcpyHostToDevice));
    double ai=5.0, vi=7.0;
    HIP_CHECK(hipMemcpy(d_aux1, &ai, sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vav,  &vi, sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dvec, 0, istep*sizeof(double)));

    gpu_copy_and_set_zeros<double,double>(
        d_v, d_u, d_a, d_aux1, d_vav, d_dvec,
        l_rows, l_cols, matrixRows, istep,
        1,1,1/*isOurProcessRow/Col/ColPrev*/, 0/*isSkewsymmetric*/, 1/*useCCL*/, 0,SM,(gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double hv[3]; HIP_CHECK(hipMemcpy(hv,d_v,3*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: v_row[0]=17", neq(hv[0],17.0));
    REPORT("double copy_and_set_zeros: v_row[1]=18", neq(hv[1],18.0));
    REPORT("double copy_and_set_zeros: v_row[2]=19", neq(hv[2],19.0));

    double hu3; HIP_CHECK(hipMemcpy(&hu3,d_u+3,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: u_col[3]=0 (l_cols>l_rows)", neq(hu3,0.0));

    double haux1; HIP_CHECK(hipMemcpy(&haux1,d_aux1,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: aux1=0", neq(haux1,0.0));

    double hvav; HIP_CHECK(hipMemcpy(&hvav,d_vav,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: vav=0 (useCCL=1)", neq(hvav,0.0));

    double hd; HIP_CHECK(hipMemcpy(&hd,d_dvec+1,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double copy_and_set_zeros: d_vec[1]=15", neq(hd,15.0));

    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_v)); HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_aux1)); HIP_CHECK(hipFree(d_vav)); HIP_CHECK(hipFree(d_dvec));
  }

  // float: isSkewsymmetric=1 -> d_vec[0] forced to 0 (not read from diagonal)
  {
    int l_rows=2, l_cols=2, matrixRows=2, istep=2;
    float h_a[6]; for (int i=0; i<6; i++) h_a[i]=(float)(i+1);

    float *d_a,*d_v,*d_u,*d_aux1,*d_vav,*d_dvec;
    HIP_CHECK(hipMalloc(&d_a,    6*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v,    l_rows*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_u,    l_cols*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_aux1, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_vav,  sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dvec, istep*sizeof(float)));

    HIP_CHECK(hipMemcpy(d_a, h_a, 6*sizeof(float), hipMemcpyHostToDevice));
    float dv[2]={9.0f,9.0f}; HIP_CHECK(hipMemcpy(d_dvec,dv,2*sizeof(float),hipMemcpyHostToDevice));
    float ax=3.0f; HIP_CHECK(hipMemcpy(d_aux1,&ax,sizeof(float),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_vav,0,sizeof(float)));

    gpu_copy_and_set_zeros<float,float>(
        d_v, d_u, d_a, d_aux1, d_vav, d_dvec,
        l_rows, l_cols, matrixRows, istep,
        1,1,1, 1/*isSkewsymmetric*/, 0/*useCCL*/, 0,SM,(gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    float hd0; HIP_CHECK(hipMemcpy(&hd0,d_dvec,sizeof(float),hipMemcpyDeviceToHost));
    REPORT("float copy_and_set_zeros: d_vec[0]=0 (isSkewsymmetric)", neqf(hd0,0.0f));

    float haux1; HIP_CHECK(hipMemcpy(&haux1,d_aux1,sizeof(float),hipMemcpyDeviceToHost));
    REPORT("float copy_and_set_zeros: aux1=0", neqf(haux1,0.0f));

    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_v)); HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_aux1)); HIP_CHECK(hipFree(d_vav)); HIP_CHECK(hipFree(d_dvec));
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
    HIP_CHECK(hipMalloc(&d_v,   l_rows*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_a,   8*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_e,   istep*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vrl, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau, istep*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_xf,  sizeof(double)));

    HIP_CHECK(hipMemcpy(d_v, hv, 4*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a, ha, 8*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xf, &xf_val, sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_e,   0, istep*sizeof(double)));
    HIP_CHECK(hipMemset(d_tau, 0, istep*sizeof(double)));

    gpu_set_e_vec_scale_set_one_store_v_row<double,double>(
        d_e, d_vrl, d_a, d_v, d_tau, d_xf,
        l_rows, l_cols, matrixRows, istep,
        1/*isOurProcessRow*/, 0/*useCCL*/, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double hvo[4]; HIP_CHECK(hipMemcpy(hvo,d_v,4*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double set_e_vec: v_row[0]=6  (2*3)", neq(hvo[0],6.0));
    REPORT("double set_e_vec: v_row[1]=12 (4*3)", neq(hvo[1],12.0));
    REPORT("double set_e_vec: v_row[2]=18 (6*3)", neq(hvo[2],18.0));
    REPORT("double set_e_vec: v_row[l_rows-1]=1 (isOurProcessRow)", neq(hvo[3],1.0));

    double hao[8]; HIP_CHECK(hipMemcpy(hao,d_a,8*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double set_e_vec: a_dev[i+matrixRows*l_cols] stores v_row[0]=6",  neq(hao[4],6.0));
    REPORT("double set_e_vec: a_dev[i+matrixRows*l_cols] stores v_row[3]=1",  neq(hao[7],1.0));

    HIP_CHECK(hipFree(d_v)); HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_e));
    HIP_CHECK(hipFree(d_vrl)); HIP_CHECK(hipFree(d_tau)); HIP_CHECK(hipFree(d_xf));
  }

  // Sub-test B: useCCL=1, isOurProcessRow=1
  // v_row[l_rows] <- tau[istep-1]; e_vec[istep-2] <- real(vrl)
  {
    int l_rows=3, l_cols=0, matrixRows=3, istep=2;
    double hv[4]={1.0,2.0,3.0,0.0}; // extra slot at index l_rows for tau broadcast
    double xf_val=1.0, vrl_val=100.0;
    double htau[2]={0.0,42.0};

    double *d_v,*d_a,*d_e,*d_vrl,*d_tau,*d_xf;
    HIP_CHECK(hipMalloc(&d_v,   (l_rows+1)*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_a,   3*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_e,   istep*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vrl, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau, istep*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_xf,  sizeof(double)));

    HIP_CHECK(hipMemcpy(d_v,   hv,       (l_rows+1)*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tau, htau,     2*sizeof(double),           hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vrl, &vrl_val, sizeof(double),             hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xf,  &xf_val,  sizeof(double),             hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_e, 0, istep*sizeof(double)));
    HIP_CHECK(hipMemset(d_a, 0, 3*sizeof(double)));

    gpu_set_e_vec_scale_set_one_store_v_row<double,double>(
        d_e, d_vrl, d_a, d_v, d_tau, d_xf,
        l_rows, l_cols, matrixRows, istep,
        1/*isOurProcessRow*/, 1/*useCCL*/, 0, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double hvo[4]; HIP_CHECK(hipMemcpy(hvo,d_v,(l_rows+1)*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double set_e_vec: v_row[l_rows]=tau[istep-1]=42 (useCCL)", neq(hvo[3],42.0));
    REPORT("double set_e_vec: v_row[l_rows-1]=1 (isOurProcessRow)",     neq(hvo[2],1.0));

    double he0; HIP_CHECK(hipMemcpy(&he0,d_e,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double set_e_vec: e_vec[0]=100 (useCCL,isOurProcessRow)", neq(he0,100.0));

    HIP_CHECK(hipFree(d_v)); HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_e));
    HIP_CHECK(hipFree(d_vrl)); HIP_CHECK(hipFree(d_tau)); HIP_CHECK(hipFree(d_xf));
  }
}

// ============================================================
// gpu_store_u_v_in_uv_vu
// tau=4, vav=3; v_row=[1,2], u_row=[1,2], v_col=[5,6], u_col=[3,4]
// conjg_tau=4 (real), nsv=0, mlr=mlc=2
// vu[i+2*0]=4*v_row[i]:           vu[0]=4,  vu[1]=8
// vu[i+2*1]=0.5*4*v_row[i]*3-u_row[i]: vu[2]=5, vu[3]=10
// uv[i+2*1]=4*v_col[i]:           uv[2]=20, uv[3]=24
// uv[i+2*0]=0.5*4*v_col[i]*3-u_col[i]: uv[0]=27, uv[1]=32
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

  HIP_CHECK(hipMalloc(&d_vu,      mlr*2*sizeof(double)));
  HIP_CHECK(hipMalloc(&d_uv,      mlc*2*sizeof(double)));
  HIP_CHECK(hipMalloc(&d_v_row,   l_rows*sizeof(double)));
  HIP_CHECK(hipMalloc(&d_u_row,   l_rows*sizeof(double)));
  HIP_CHECK(hipMalloc(&d_v_col,   l_cols*sizeof(double)));
  HIP_CHECK(hipMalloc(&d_u_col,   l_cols*sizeof(double)));
  HIP_CHECK(hipMalloc(&d_tau_dev, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_aux,     sizeof(double)));
  HIP_CHECK(hipMalloc(&d_vav,     sizeof(double)));
  HIP_CHECK(hipMalloc(&d_tau_vop, sizeof(double)));

  HIP_CHECK(hipMemcpy(d_v_row, hv_row, 2*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_u_row, hu_row, 2*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_col, hv_col, 2*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_u_col, hu_col, 2*sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_vav,     &vav_val, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_tau_vop, &tau_val, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_vu, 0, mlr*2*sizeof(double)));
  HIP_CHECK(hipMemset(d_uv, 0, mlc*2*sizeof(double)));

  gpu_store_u_v_in_uv_vu<double>(
      d_vu, d_uv, d_v_row, d_u_row, d_v_col, d_u_col,
      d_tau_dev, d_aux, d_vav, d_tau_vop,
      l_rows, l_cols, n_stored_vecs, mlr, mlc,
      istep, 0/*useCCL*/, 0, (gpuStream_t)0);
  HIP_CHECK(hipDeviceSynchronize());

  double hvu[4]; HIP_CHECK(hipMemcpy(hvu,d_vu,4*sizeof(double),hipMemcpyDeviceToHost));
  REPORT("double store_u_v_in_uv_vu: vu[0]=4  (conjg_tau*v_row[0])",      neq(hvu[0], 4.0));
  REPORT("double store_u_v_in_uv_vu: vu[1]=8  (conjg_tau*v_row[1])",      neq(hvu[1], 8.0));
  REPORT("double store_u_v_in_uv_vu: vu[2]=5  (0.5*tau*vrow[0]*vav-urow)", neq(hvu[2], 5.0));
  REPORT("double store_u_v_in_uv_vu: vu[3]=10 (0.5*tau*vrow[1]*vav-urow)", neq(hvu[3],10.0));

  double huv[4]; HIP_CHECK(hipMemcpy(huv,d_uv,4*sizeof(double),hipMemcpyDeviceToHost));
  REPORT("double store_u_v_in_uv_vu: uv[2]=20 (conjg_tau*v_col[0])",       neq(huv[2],20.0));
  REPORT("double store_u_v_in_uv_vu: uv[3]=24 (conjg_tau*v_col[1])",       neq(huv[3],24.0));
  REPORT("double store_u_v_in_uv_vu: uv[0]=27 (0.5*tau*vcol[0]*vav-ucol)", neq(huv[0],27.0));
  REPORT("double store_u_v_in_uv_vu: uv[1]=32 (0.5*tau*vcol[1]*vav-ucol)", neq(huv[1],32.0));

  HIP_CHECK(hipFree(d_vu)); HIP_CHECK(hipFree(d_uv));
  HIP_CHECK(hipFree(d_v_row)); HIP_CHECK(hipFree(d_u_row));
  HIP_CHECK(hipFree(d_v_col)); HIP_CHECK(hipFree(d_u_col));
  HIP_CHECK(hipFree(d_tau_dev)); HIP_CHECK(hipFree(d_aux));
  HIP_CHECK(hipFree(d_vav)); HIP_CHECK(hipFree(d_tau_vop));
}

// ============================================================
// gpu_update_matrix_element_add
// ============================================================
static void test_update_matrix_element_add()
{
  printf("\ngpu_update_matrix_element_add:\n");
  int SM = 1;

  // double: n_stored_vecs=2, l_rows=3, l_cols=3, mlr=mlc=matrixRows=3, istep=3
  // dot = 1*5+2*3+3*2+4*1 = 21; a[8]=10 -> 31; d_vec[1]=0 -> 21
  {
    int l_rows=3,l_cols=3,matrixRows=3,mlr=3,mlc=3,istep=3,nsv=2;
    double h_vu[12]={}, h_uv[12]={}, h_a[9]={}, h_d[3]={};
    h_vu[2]=1.0; h_vu[5]=2.0; h_vu[8]=3.0; h_vu[11]=4.0;
    h_uv[2]=5.0; h_uv[5]=3.0; h_uv[8]=2.0; h_uv[11]=1.0;
    h_a[8]=10.0;

    double *d_vu,*d_uv,*d_a,*d_d;
    HIP_CHECK(hipMalloc(&d_vu,12*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_uv,12*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_a,  9*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_d,  3*sizeof(double)));
    HIP_CHECK(hipMemcpy(d_vu,h_vu,12*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_uv,h_uv,12*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a, h_a,  9*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, h_d,  3*sizeof(double),hipMemcpyHostToDevice));

    gpu_update_matrix_element_add<double,double>(
        d_vu, d_uv, d_a, d_d,
        l_rows, l_cols, matrixRows, mlr, mlc, istep, nsv,
        0/*isSkewsymmetric*/, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double ha8; HIP_CHECK(hipMemcpy(&ha8,d_a+8,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double update_matrix_element: a[8]=31 (10+dot21)", neq(ha8,31.0));

    double hd1; HIP_CHECK(hipMemcpy(&hd1,d_d+1,sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double update_matrix_element: d_vec[1]=21 (dot product)", neq(hd1,21.0));

    HIP_CHECK(hipFree(d_vu)); HIP_CHECK(hipFree(d_uv));
    HIP_CHECK(hipFree(d_a));  HIP_CHECK(hipFree(d_d));
  }

  // float: isSkewsymmetric=1 -> d_vec unchanged
  {
    int l_rows=2,l_cols=2,matrixRows=2,mlr=2,mlc=2,istep=2,nsv=1;
    float h_vu[4]={},h_uv[4]={},h_a[4]={},h_d[2]={5.0f,5.0f};
    h_vu[1]=3.0f; h_vu[3]=4.0f;
    h_uv[1]=2.0f; h_uv[3]=1.0f;
    h_a[3]=1.0f;

    float *d_vu,*d_uv,*d_a,*d_d;
    HIP_CHECK(hipMalloc(&d_vu,4*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_uv,4*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_a, 4*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_d, 2*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_vu,h_vu,4*sizeof(float),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_uv,h_uv,4*sizeof(float),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a, h_a, 4*sizeof(float),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, h_d, 2*sizeof(float),hipMemcpyHostToDevice));

    gpu_update_matrix_element_add<float,float>(
        d_vu, d_uv, d_a, d_d,
        l_rows, l_cols, matrixRows, mlr, mlc, istep, nsv,
        1/*isSkewsymmetric*/, 0, SM, (gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    float ha3; HIP_CHECK(hipMemcpy(&ha3,d_a+3,sizeof(float),hipMemcpyDeviceToHost));
    REPORT("float update_matrix_element: a[3]=11 (1+dot10)", neqf(ha3,11.0f));

    float hd0; HIP_CHECK(hipMemcpy(&hd0,d_d,sizeof(float),hipMemcpyDeviceToHost));
    REPORT("float update_matrix_element: d_vec[0]=5 unchanged (isSkewsymmetric)", neqf(hd0,5.0f));

    HIP_CHECK(hipFree(d_vu)); HIP_CHECK(hipFree(d_uv));
    HIP_CHECK(hipFree(d_a));  HIP_CHECK(hipFree(d_d));
  }
}

// ============================================================
// gpu_transpose_reduceadd_vectors_copy_block
// ============================================================
static void test_transpose_reduceadd_vectors_copy_block()
{
  printf("\ngpu_transpose_reduceadd_vectors_copy_block:\n");
  int SM = 1;
  int nvc=1, nvr=6, n_block=0, nblks_skip=0, nblks_tot=3;
  int lcm_s_t=1, nblk=2, auxstride=6, np_st=1, ld_st=6, N=6;

  // Test A: direction=1 -> vmat -> aux (copy)
  {
    double h_vmat[6]={10,20,30,40,50,60};
    double *d_aux, *d_vmat;
    HIP_CHECK(hipMalloc(&d_aux,  N*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vmat, N*sizeof(double)));
    HIP_CHECK(hipMemset(d_aux, 0, N*sizeof(double)));
    HIP_CHECK(hipMemcpy(d_vmat, h_vmat, N*sizeof(double), hipMemcpyHostToDevice));

    gpu_transpose_reduceadd_vectors_copy_block<double>(
        d_aux, d_vmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
        lcm_s_t, nblk, auxstride, np_st, ld_st,
        1/*direction*/, 0,0, 0,SM,(gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double haux[6]; HIP_CHECK(hipMemcpy(haux,d_aux,6*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double transpose dir=1: aux[0]=10", neq(haux[0],10.0));
    REPORT("double transpose dir=1: aux[2]=30", neq(haux[2],30.0));
    REPORT("double transpose dir=1: aux[5]=60", neq(haux[5],60.0));

    HIP_CHECK(hipFree(d_aux)); HIP_CHECK(hipFree(d_vmat));
  }

  // Test B: direction=2, isSkewsymmetric=1 -> vmat = -aux
  {
    double h_aux[6]={1,2,3,4,5,6};
    double *d_aux, *d_vmat;
    HIP_CHECK(hipMalloc(&d_aux,  N*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vmat, N*sizeof(double)));
    HIP_CHECK(hipMemcpy(d_aux, h_aux, N*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_vmat, 0, N*sizeof(double)));

    gpu_transpose_reduceadd_vectors_copy_block<double>(
        d_aux, d_vmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
        lcm_s_t, nblk, auxstride, np_st, ld_st,
        2/*direction*/, 1/*isSkewsymmetric*/, 0, 0,SM,(gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double hv[6]; HIP_CHECK(hipMemcpy(hv,d_vmat,6*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double transpose dir=2 skew: vmat[0]=-1", neq(hv[0],-1.0));
    REPORT("double transpose dir=2 skew: vmat[3]=-4", neq(hv[3],-4.0));

    HIP_CHECK(hipFree(d_aux)); HIP_CHECK(hipFree(d_vmat));
  }

  // Test C: direction=2, isReduceadd=1 -> vmat += aux
  {
    double h_aux[6]={1,2,3,4,5,6}, h_vmat[6]={10,20,30,40,50,60};
    double *d_aux, *d_vmat;
    HIP_CHECK(hipMalloc(&d_aux,  N*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vmat, N*sizeof(double)));
    HIP_CHECK(hipMemcpy(d_aux,  h_aux,  N*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vmat, h_vmat, N*sizeof(double), hipMemcpyHostToDevice));

    gpu_transpose_reduceadd_vectors_copy_block<double>(
        d_aux, d_vmat, nvc, nvr, n_block, nblks_skip, nblks_tot,
        lcm_s_t, nblk, auxstride, np_st, ld_st,
        2/*direction*/, 0, 1/*isReduceadd*/, 0,SM,(gpuStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    double hv[6]; HIP_CHECK(hipMemcpy(hv,d_vmat,6*sizeof(double),hipMemcpyDeviceToHost));
    REPORT("double transpose dir=2 reduceadd: vmat[0]=11", neq(hv[0],11.0));
    REPORT("double transpose dir=2 reduceadd: vmat[5]=66", neq(hv[5],66.0));

    HIP_CHECK(hipFree(d_aux)); HIP_CHECK(hipFree(d_vmat));
  }
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for tridiag_gpu.h kernels (ROCm/HIP) ===\n");

  test_hh_transform<double,double>("double", 1e-10);
  test_hh_transform<float,float>  ("float",  1e-5f);
  test_dot_product();
  test_dot_product_and_assign();
  test_copy_and_set_zeros();
  test_set_e_vec_scale_set_one_store_v_row();
  test_store_u_v_in_uv_vu();
  test_update_matrix_element_add();
  test_transpose_reduceadd_vectors_copy_block();

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
