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

// Unit tests for all functions/types in src/GPU/common_device_functions.h (AMD ROCm/HIP backend).
// ROCm/HIP port of test/unit_tests/CUDA/test_common_device_functions.cu.
// Covers: elpaHostNumberFromInt, elpaDeviceSign, elpaDeviceNumber,
//         elpaDeviceNumberFromRealImag, elpaDeviceAdd, elpaDeviceSubtract,
//         elpaDeviceMultiply, elpaDeviceDivide, elpaDeviceSqrt,
//         elpaDeviceComplexConjugate, elpaDeviceRealPart, elpaDeviceImagPart,
//         elpaDeviceEqual, elpaDeviceEqualBool, atomicAdd (complex),
//         pcol, prow, local_index.
// All four types tested where applicable: double, float, double_complex (hipDoubleComplex),
// float_complex (hipFloatComplex).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_AMD_GPU_VERSION

#include <stdint.h>
#include <cstring>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <type_traits>
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

#include "../../../../src/GPU/common_device_functions.h"

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

static bool neq (double a, double b) { return fabs (a - b) < 1e-12; }
static bool neqf(float  a, float  b) { return fabsf(a - b) < 1e-5f; }

// ---- Device kernels ----

template <typename T>
__global__ void k_sign(T a, T b, T *out) { *out = elpaDeviceSign(a, b); }

template <typename T>
__global__ void k_number(double v, T *out) { *out = elpaDeviceNumber<T>(v); }

template <typename T, typename R>
__global__ void k_fromRI(R re, R im, T *out) { *out = elpaDeviceNumberFromRealImag<T,R>(re, im); }

template <typename T>
__global__ void k_add(T a, T b, T *out) { *out = elpaDeviceAdd(a, b); }

template <typename T>
__global__ void k_sub(T a, T b, T *out) { *out = elpaDeviceSubtract(a, b); }

template <typename T>
__global__ void k_mul(T a, T b, T *out) { *out = elpaDeviceMultiply(a, b); }

template <typename T>
__global__ void k_div(T a, T b, T *out) { *out = elpaDeviceDivide(a, b); }

template <typename T>
__global__ void k_sqrt(T v, T *out) { *out = elpaDeviceSqrt(v); }

template <typename T>
__global__ void k_conj(T v, T *out) { *out = elpaDeviceComplexConjugate(v); }

// elpaDeviceRealPart / elpaDeviceImagPart return real scalars; use explicit kernels
// to avoid ambiguous template deduction on the output pointer type.
__global__ void k_realPart_d (double         v, double *o) { *o = elpaDeviceRealPart(v); }
__global__ void k_realPart_f (float          v, float  *o) { *o = elpaDeviceRealPart(v); }
__global__ void k_realPart_dc(double_complex v, double *o) { *o = elpaDeviceRealPart(v); }
__global__ void k_realPart_fc(float_complex  v, float  *o) { *o = elpaDeviceRealPart(v); }

__global__ void k_imagPart_d (double         v, double *o) { *o = elpaDeviceImagPart(v); }
__global__ void k_imagPart_f (float          v, float  *o) { *o = (float)elpaDeviceImagPart(v); }
__global__ void k_imagPart_dc(double_complex v, double *o) { *o = elpaDeviceImagPart(v); }
__global__ void k_imagPart_fc(float_complex  v, float  *o) { *o = elpaDeviceImagPart(v); }

template <typename T>
__global__ void k_equal(T a, T b, T *out) { *out = elpaDeviceEqual(a, b); }

template <typename T>
__global__ void k_equalBool(T a, T b, bool *out) { *out = elpaDeviceEqualBool(a, b); }

template <typename T>
__global__ void k_atomicAdd(T *addr, T val) { atomicAdd(addr, val); }

__global__ void k_pcol(int col, int nblk, int np_cols, int *out) {
  *out = pcol(col, nblk, np_cols);
}
__global__ void k_prow(int row, int nblk, int np_rows, int *out) {
  *out = prow(row, nblk, np_rows);
}
__global__ void k_localIndex(int idx, int my_proc, int num_procs, int nblk, int iflag, int *out) {
  *out = local_index(idx, my_proc, num_procs, nblk, iflag);
}

// ---- Device read helper ----
template <typename T>
static T dev_read(T *d) {
  T h;
  HIP_CHECK(hipMemcpy(&h, d, sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

// ============================================================
// elpaHostNumberFromInt  (host function — no kernel needed)
// ============================================================
static void test_elpaHostNumberFromInt()
{
  printf("\nelpaHostNumberFromInt:\n");

  REPORT("double(7)==7.0",  neq (elpaHostNumberFromInt<double>(7), 7.0));
  REPORT("float(7)==7.0f",  neqf(elpaHostNumberFromInt<float>(7),  7.0f));

  double_complex dc = elpaHostNumberFromInt<double_complex>(7);
  REPORT("double_complex(7).real==7.0", neq(dc.x, 7.0));
  REPORT("double_complex(7).imag==0.0", neq(dc.y, 0.0));

  float_complex fc = elpaHostNumberFromInt<float_complex>(7);
  REPORT("float_complex(7).real==7.0f", neqf(fc.x, 7.0f));
  REPORT("float_complex(7).imag==0.0f", neqf(fc.y, 0.0f));
}

// ============================================================
// elpaDeviceSign  (double and float only — no complex specialization)
// ============================================================
static void test_elpaDeviceSign()
{
  printf("\nelpaDeviceSign:\n");

  double *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float  *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));

  k_sign<<<1,1>>>(3.0, 1.0, dd);   HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: sign( 3, 1)== 3", neq(dev_read(dd),  3.0));
  k_sign<<<1,1>>>(3.0, 0.0, dd);   HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: sign( 3, 0)== 3 (b==0 treated as >=0)", neq(dev_read(dd),  3.0));
  k_sign<<<1,1>>>(3.0, -1.0, dd);  HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: sign( 3,-1)==-3", neq(dev_read(dd), -3.0));
  k_sign<<<1,1>>>(-5.0, 2.0, dd);  HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: sign(-5, 2)== 5", neq(dev_read(dd),  5.0));

  k_sign<<<1,1>>>(3.0f, 1.0f, df);  HIP_CHECK(hipDeviceSynchronize());
  REPORT("float:  sign( 3, 1)== 3", neqf(dev_read(df),  3.0f));
  k_sign<<<1,1>>>(3.0f, -1.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float:  sign( 3,-1)==-3", neqf(dev_read(df), -3.0f));

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
}

// ============================================================
// elpaDeviceNumber
// ============================================================
static void test_elpaDeviceNumber()
{
  printf("\nelpaDeviceNumber:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_number<double>        <<<1,1>>>(5.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double(5.0)==5.0",           neq (dev_read(dd), 5.0));

  k_number<float>         <<<1,1>>>(5.0, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float(5.0)==5.0f",           neqf(dev_read(df), 5.0f));

  k_number<double_complex><<<1,1>>>(5.0, dc); HIP_CHECK(hipDeviceSynchronize());
  { double_complex h = dev_read(dc);
    REPORT("double_complex(5.0).real==5.0", neq(h.x, 5.0));
    REPORT("double_complex(5.0).imag==0.0", neq(h.y, 0.0)); }

  k_number<float_complex> <<<1,1>>>(5.0, fc); HIP_CHECK(hipDeviceSynchronize());
  { float_complex h = dev_read(fc);
    REPORT("float_complex(5.0).real==5.0f", neqf(h.x, 5.0f));
    REPORT("float_complex(5.0).imag==0.0f", neqf(h.y, 0.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceNumberFromRealImag
// ============================================================
static void test_elpaDeviceNumberFromRealImag()
{
  printf("\nelpaDeviceNumberFromRealImag:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_fromRI<double,double>         <<<1,1>>>(3.0, 4.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: fromRI(3,4)==3.0 (real part only)", neq(dev_read(dd), 3.0));

  k_fromRI<float,float>           <<<1,1>>>(3.0f, 4.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: fromRI(3,4)==3.0f (real part only)", neqf(dev_read(df), 3.0f));

  k_fromRI<double_complex,double> <<<1,1>>>(3.0, 4.0, dc); HIP_CHECK(hipDeviceSynchronize());
  { double_complex h = dev_read(dc);
    REPORT("double_complex: fromRI(3,4).real==3.0", neq(h.x, 3.0));
    REPORT("double_complex: fromRI(3,4).imag==4.0", neq(h.y, 4.0)); }

  k_fromRI<float_complex,float>   <<<1,1>>>(3.0f, 4.0f, fc); HIP_CHECK(hipDeviceSynchronize());
  { float_complex h = dev_read(fc);
    REPORT("float_complex: fromRI(3,4).real==3.0f", neqf(h.x, 3.0f));
    REPORT("float_complex: fromRI(3,4).imag==4.0f", neqf(h.y, 4.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceAdd
// ============================================================
static void test_elpaDeviceAdd()
{
  printf("\nelpaDeviceAdd:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_add<<<1,1>>>(1.0, 2.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: 1+2==3", neq(dev_read(dd), 3.0));

  k_add<<<1,1>>>(1.0f, 2.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: 1+2==3",  neqf(dev_read(df), 3.0f));

  { double_complex a = make_hipDoubleComplex(1.0,2.0), b = make_hipDoubleComplex(3.0,4.0);
    k_add<<<1,1>>>(a, b, dc); HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: (1+2i)+(3+4i).real==4", neq(h.x, 4.0));
    REPORT("double_complex: (1+2i)+(3+4i).imag==6", neq(h.y, 6.0)); }

  { float_complex a = make_hipFloatComplex(1.0f,2.0f), b = make_hipFloatComplex(3.0f,4.0f);
    k_add<<<1,1>>>(a, b, fc); HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: (1+2i)+(3+4i).real==4", neqf(h.x, 4.0f));
    REPORT("float_complex: (1+2i)+(3+4i).imag==6", neqf(h.y, 6.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceSubtract
// ============================================================
static void test_elpaDeviceSubtract()
{
  printf("\nelpaDeviceSubtract:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_sub<<<1,1>>>(5.0, 2.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: 5-2==3", neq(dev_read(dd), 3.0));

  k_sub<<<1,1>>>(5.0f, 2.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: 5-2==3",  neqf(dev_read(df), 3.0f));

  { double_complex a = make_hipDoubleComplex(5.0,6.0), b = make_hipDoubleComplex(1.0,2.0);
    k_sub<<<1,1>>>(a, b, dc); HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: (5+6i)-(1+2i).real==4", neq(h.x, 4.0));
    REPORT("double_complex: (5+6i)-(1+2i).imag==4", neq(h.y, 4.0)); }

  { float_complex a = make_hipFloatComplex(5.0f,6.0f), b = make_hipFloatComplex(1.0f,2.0f);
    k_sub<<<1,1>>>(a, b, fc); HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: (5+6i)-(1+2i).real==4", neqf(h.x, 4.0f));
    REPORT("float_complex: (5+6i)-(1+2i).imag==4", neqf(h.y, 4.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceMultiply
// ============================================================
static void test_elpaDeviceMultiply()
{
  printf("\nelpaDeviceMultiply:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_mul<<<1,1>>>(3.0, 4.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: 3*4==12", neq(dev_read(dd), 12.0));

  k_mul<<<1,1>>>(3.0f, 4.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: 3*4==12",  neqf(dev_read(df), 12.0f));

  { // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5+10i
    double_complex a = make_hipDoubleComplex(1.0,2.0), b = make_hipDoubleComplex(3.0,4.0);
    k_mul<<<1,1>>>(a, b, dc); HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: (1+2i)*(3+4i).real==-5", neq(h.x, -5.0));
    REPORT("double_complex: (1+2i)*(3+4i).imag==10", neq(h.y, 10.0)); }

  { float_complex a = make_hipFloatComplex(1.0f,2.0f), b = make_hipFloatComplex(3.0f,4.0f);
    k_mul<<<1,1>>>(a, b, fc); HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: (1+2i)*(3+4i).real==-5", neqf(h.x, -5.0f));
    REPORT("float_complex: (1+2i)*(3+4i).imag==10", neqf(h.y, 10.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceDivide
// ============================================================
static void test_elpaDeviceDivide()
{
  printf("\nelpaDeviceDivide:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_div<<<1,1>>>(10.0, 4.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: 10/4==2.5", neq(dev_read(dd), 2.5));

  k_div<<<1,1>>>(10.0f, 4.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: 10/4==2.5",  neqf(dev_read(df), 2.5f));

  { // (4+2i)/(3+1i) = (4+2i)(3-1i)/10 = (12-4i+6i+2)/10 = (14+2i)/10 = 1.4+0.2i
    double_complex a = make_hipDoubleComplex(4.0,2.0), b = make_hipDoubleComplex(3.0,1.0);
    k_div<<<1,1>>>(a, b, dc); HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: (4+2i)/(3+1i).real==1.4", neq(h.x, 1.4));
    REPORT("double_complex: (4+2i)/(3+1i).imag==0.2", neq(h.y, 0.2)); }

  { float_complex a = make_hipFloatComplex(4.0f,2.0f), b = make_hipFloatComplex(3.0f,1.0f);
    k_div<<<1,1>>>(a, b, fc); HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: (4+2i)/(3+1i).real==1.4", neqf(h.x, 1.4f));
    REPORT("float_complex: (4+2i)/(3+1i).imag==0.2", neqf(h.y, 0.2f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceSqrt  (double and float only — no complex specialization)
// ============================================================
static void test_elpaDeviceSqrt()
{
  printf("\nelpaDeviceSqrt:\n");

  double *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float  *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));

  k_sqrt<<<1,1>>>(9.0, dd);  HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: sqrt(9)==3",          neq (dev_read(dd), 3.0));
  k_sqrt<<<1,1>>>(2.0, dd);  HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: sqrt(2)==1.41421356", neq (dev_read(dd), sqrt(2.0)));

  k_sqrt<<<1,1>>>(9.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: sqrt(9)==3",           neqf(dev_read(df), 3.0f));
  k_sqrt<<<1,1>>>(2.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: sqrt(2)==1.41421f",    neqf(dev_read(df), sqrtf(2.0f)));

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
}

// ============================================================
// elpaDeviceComplexConjugate
// ============================================================
static void test_elpaDeviceComplexConjugate()
{
  printf("\nelpaDeviceComplexConjugate:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_conj<<<1,1>>>(5.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: conj(5)==5", neq(dev_read(dd), 5.0));

  k_conj<<<1,1>>>(5.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: conj(5)==5",  neqf(dev_read(df), 5.0f));

  { double_complex a = make_hipDoubleComplex(3.0, 4.0);
    k_conj<<<1,1>>>(a, dc); HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: conj(3+4i).real== 3", neq(h.x,  3.0));
    REPORT("double_complex: conj(3+4i).imag==-4", neq(h.y, -4.0)); }

  { float_complex a = make_hipFloatComplex(3.0f, 4.0f);
    k_conj<<<1,1>>>(a, fc); HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: conj(3+4i).real== 3", neqf(h.x,  3.0f));
    REPORT("float_complex: conj(3+4i).imag==-4", neqf(h.y, -4.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceRealPart
// ============================================================
static void test_elpaDeviceRealPart()
{
  printf("\nelpaDeviceRealPart:\n");

  double *od; HIP_CHECK(hipMalloc(&od, sizeof(double)));
  float  *of; HIP_CHECK(hipMalloc(&of, sizeof(float)));

  k_realPart_d<<<1,1>>>(7.0, od); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: realPart(7.0)==7.0", neq(dev_read(od), 7.0));

  k_realPart_f<<<1,1>>>(7.0f, of); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: realPart(7.0f)==7.0f", neqf(dev_read(of), 7.0f));

  { double_complex a = make_hipDoubleComplex(3.0, 4.0);
    k_realPart_dc<<<1,1>>>(a, od); HIP_CHECK(hipDeviceSynchronize());
    REPORT("double_complex: realPart(3+4i)==3.0", neq(dev_read(od), 3.0)); }

  { float_complex a = make_hipFloatComplex(3.0f, 4.0f);
    k_realPart_fc<<<1,1>>>(a, of); HIP_CHECK(hipDeviceSynchronize());
    REPORT("float_complex: realPart(3+4i)==3.0f", neqf(dev_read(of), 3.0f)); }

  HIP_CHECK(hipFree(od)); HIP_CHECK(hipFree(of));
}

// ============================================================
// elpaDeviceImagPart
// ============================================================
static void test_elpaDeviceImagPart()
{
  printf("\nelpaDeviceImagPart:\n");

  double *od; HIP_CHECK(hipMalloc(&od, sizeof(double)));
  float  *of; HIP_CHECK(hipMalloc(&of, sizeof(float)));

  k_imagPart_d<<<1,1>>>(7.0, od); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: imagPart(7.0)==0.0", neq(dev_read(od), 0.0));

  k_imagPart_f<<<1,1>>>(7.0f, of); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: imagPart(7.0f)==0.0f", neqf(dev_read(of), 0.0f));

  { double_complex a = make_hipDoubleComplex(3.0, 4.0);
    k_imagPart_dc<<<1,1>>>(a, od); HIP_CHECK(hipDeviceSynchronize());
    REPORT("double_complex: imagPart(3+4i)==4.0", neq(dev_read(od), 4.0)); }

  { float_complex a = make_hipFloatComplex(3.0f, 4.0f);
    k_imagPart_fc<<<1,1>>>(a, of); HIP_CHECK(hipDeviceSynchronize());
    REPORT("float_complex: imagPart(3+4i)==4.0f", neqf(dev_read(of), 4.0f)); }

  HIP_CHECK(hipFree(od)); HIP_CHECK(hipFree(of));
}

// ============================================================
// elpaDeviceEqual
// ============================================================
static void test_elpaDeviceEqual()
{
  printf("\nelpaDeviceEqual:\n");

  double         *dd; HIP_CHECK(hipMalloc(&dd, sizeof(double)));
  float          *df; HIP_CHECK(hipMalloc(&df, sizeof(float)));
  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  k_equal<<<1,1>>>(3.0, 3.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: equal(3,3)==1.0", neq(dev_read(dd), 1.0));
  k_equal<<<1,1>>>(3.0, 4.0, dd); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: equal(3,4)==0.0", neq(dev_read(dd), 0.0));

  k_equal<<<1,1>>>(3.0f, 3.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: equal(3,3)==1.0f", neqf(dev_read(df), 1.0f));
  k_equal<<<1,1>>>(3.0f, 4.0f, df); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: equal(3,4)==0.0f", neqf(dev_read(df), 0.0f));

  { double_complex a = make_hipDoubleComplex(1.0,2.0);
    k_equal<<<1,1>>>(a, a, dc); HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: equal(z,z).real==1.0", neq(h.x, 1.0));
    REPORT("double_complex: equal(z,z).imag==0.0", neq(h.y, 0.0));
    double_complex b = make_hipDoubleComplex(2.0,1.0);
    k_equal<<<1,1>>>(a, b, dc); HIP_CHECK(hipDeviceSynchronize());
    h = dev_read(dc);
    REPORT("double_complex: equal(z1,z2).real==0.0", neq(h.x, 0.0)); }

  { float_complex a = make_hipFloatComplex(1.0f,2.0f);
    k_equal<<<1,1>>>(a, a, fc); HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: equal(z,z).real==1.0f", neqf(h.x, 1.0f));
    REPORT("float_complex: equal(z,z).imag==0.0f", neqf(h.y, 0.0f));
    float_complex b = make_hipFloatComplex(2.0f,1.0f);
    k_equal<<<1,1>>>(a, b, fc); HIP_CHECK(hipDeviceSynchronize());
    h = dev_read(fc);
    REPORT("float_complex: equal(z1,z2).real==0.0f", neqf(h.x, 0.0f)); }

  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(df));
  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// elpaDeviceEqualBool
// ============================================================
static void test_elpaDeviceEqualBool()
{
  printf("\nelpaDeviceEqualBool:\n");

  bool *db; HIP_CHECK(hipMalloc(&db, sizeof(bool)));

  k_equalBool<<<1,1>>>(3.0, 3.0, db); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: equalBool(3,3)==true",  dev_read(db) == true);
  k_equalBool<<<1,1>>>(3.0, 4.0, db); HIP_CHECK(hipDeviceSynchronize());
  REPORT("double: equalBool(3,4)==false", dev_read(db) == false);

  k_equalBool<<<1,1>>>(3.0f, 3.0f, db); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: equalBool(3,3)==true",   dev_read(db) == true);
  k_equalBool<<<1,1>>>(3.0f, 4.0f, db); HIP_CHECK(hipDeviceSynchronize());
  REPORT("float: equalBool(3,4)==false",  dev_read(db) == false);

  { double_complex a = make_hipDoubleComplex(1.0,2.0), b = make_hipDoubleComplex(2.0,1.0);
    k_equalBool<<<1,1>>>(a, a, db); HIP_CHECK(hipDeviceSynchronize());
    REPORT("double_complex: equalBool(z,z)==true",    dev_read(db) == true);
    k_equalBool<<<1,1>>>(a, b, db); HIP_CHECK(hipDeviceSynchronize());
    REPORT("double_complex: equalBool(z1,z2)==false", dev_read(db) == false); }

  { float_complex a = make_hipFloatComplex(1.0f,2.0f), b = make_hipFloatComplex(2.0f,1.0f);
    k_equalBool<<<1,1>>>(a, a, db); HIP_CHECK(hipDeviceSynchronize());
    REPORT("float_complex: equalBool(z,z)==true",    dev_read(db) == true);
    k_equalBool<<<1,1>>>(a, b, db); HIP_CHECK(hipDeviceSynchronize());
    REPORT("float_complex: equalBool(z1,z2)==false", dev_read(db) == false); }

  HIP_CHECK(hipFree(db));
}

// ============================================================
// atomicAdd for double_complex and float_complex
// ============================================================
static void test_atomicAdd()
{
  printf("\natomicAdd (complex types):\n");

  double_complex *dc; HIP_CHECK(hipMalloc(&dc, sizeof(double_complex)));
  float_complex  *fc; HIP_CHECK(hipMalloc(&fc, sizeof(float_complex)));

  { double_complex zero = make_hipDoubleComplex(0.0,0.0);
    HIP_CHECK(hipMemcpy(dc, &zero, sizeof(double_complex), hipMemcpyHostToDevice));
    double_complex inc = make_hipDoubleComplex(1.5, 2.5);
    k_atomicAdd<<<1,1>>>(dc, inc);
    k_atomicAdd<<<1,1>>>(dc, inc);
    HIP_CHECK(hipDeviceSynchronize());
    double_complex h = dev_read(dc);
    REPORT("double_complex: atomicAdd x2 with (1.5+2.5i).real==3.0", neq(h.x, 3.0));
    REPORT("double_complex: atomicAdd x2 with (1.5+2.5i).imag==5.0", neq(h.y, 5.0)); }

  { float_complex zero = make_hipFloatComplex(0.0f,0.0f);
    HIP_CHECK(hipMemcpy(fc, &zero, sizeof(float_complex), hipMemcpyHostToDevice));
    float_complex inc = make_hipFloatComplex(1.5f, 2.5f);
    k_atomicAdd<<<1,1>>>(fc, inc);
    k_atomicAdd<<<1,1>>>(fc, inc);
    HIP_CHECK(hipDeviceSynchronize());
    float_complex h = dev_read(fc);
    REPORT("float_complex: atomicAdd x2 with (1.5+2.5i).real==3.0f", neqf(h.x, 3.0f));
    REPORT("float_complex: atomicAdd x2 with (1.5+2.5i).imag==5.0f", neqf(h.y, 5.0f)); }

  HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(fc));
}

// ============================================================
// pcol / prow
// Block-cyclic layout: process owns global index idx iff
//   ((idx-1)/nblk) % np_procs == my_proc   (1-based Fortran indexing)
// ============================================================
static void test_pcol_prow()
{
  printf("\npcol / prow:\n");

  int *di; HIP_CHECK(hipMalloc(&di, sizeof(int)));

  // With nblk=4, np=4: block 0->proc0 (cols 1-4), block 1->proc1 (cols 5-8), ...
  k_pcol<<<1,1>>>( 1, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("pcol( 1, nblk=4, np=4)==0", dev_read(di) == 0);
  k_pcol<<<1,1>>>( 5, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("pcol( 5, nblk=4, np=4)==1", dev_read(di) == 1);
  k_pcol<<<1,1>>>( 9, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("pcol( 9, nblk=4, np=4)==2", dev_read(di) == 2);
  k_pcol<<<1,1>>>(13, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("pcol(13, nblk=4, np=4)==3", dev_read(di) == 3);
  k_pcol<<<1,1>>>(17, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("pcol(17, nblk=4, np=4)==0 (wraps around)", dev_read(di) == 0);

  // prow uses the same formula
  k_prow<<<1,1>>>( 1, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("prow( 1, nblk=4, np=4)==0", dev_read(di) == 0);
  k_prow<<<1,1>>>( 8, 4, 4, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("prow( 8, nblk=4, np=4)==1", dev_read(di) == 1);

  HIP_CHECK(hipFree(di));
}

// ============================================================
// local_index
// nblk=4, num_procs=4:
//   proc 0 owns global rows 1-4, 17-20, ...  (local 1-4, 5-8, ...)
//   proc 1 owns global rows 5-8,  21-24, ...
// ============================================================
static void test_localIndex()
{
  printf("\nlocal_index:\n");

  int *di; HIP_CHECK(hipMalloc(&di, sizeof(int)));

  // Local index for indices that live on the queried process
  k_localIndex<<<1,1>>>(1,  0, 4, 4, 0, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=1,  proc=0, np=4, nblk=4, iflag=0)==1", dev_read(di) == 1);
  k_localIndex<<<1,1>>>(4,  0, 4, 4, 0, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=4,  proc=0, np=4, nblk=4, iflag=0)==4", dev_read(di) == 4);
  k_localIndex<<<1,1>>>(5,  1, 4, 4, 0, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=5,  proc=1, np=4, nblk=4, iflag=0)==1", dev_read(di) == 1);
  k_localIndex<<<1,1>>>(17, 0, 4, 4, 0, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=17, proc=0, np=4, nblk=4, iflag=0)==5", dev_read(di) == 5);

  // iflag==0: non-local index returns 0
  k_localIndex<<<1,1>>>(6, 0, 4, 4, 0, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=6,  proc=0, np=4, nblk=4, iflag=0)==0 (non-local)", dev_read(di) == 0);

  // iflag<0: last local index before the given global index
  // idx=6 is in block 1 (proc 1); last local on proc 0 before that is local 4 (global 4)
  k_localIndex<<<1,1>>>(6, 0, 4, 4, -1, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=6,  proc=0, np=4, nblk=4, iflag=-1)==4", dev_read(di) == 4);

  // iflag>0: next local index after the given global index
  // idx=6 is in block 1 (proc 1); next local on proc 0 after that is local 5 (global 17)
  k_localIndex<<<1,1>>>(6, 0, 4, 4, 1, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=6,  proc=0, np=4, nblk=4, iflag=1)==5", dev_read(di) == 5);

  // Edge: idx before proc's first owned block (proc=3, idx=1 is block 0 -> proc 0)
  // iflag<0: no owned block before idx=1 on proc 3 -> 0
  k_localIndex<<<1,1>>>(1, 3, 4, 4, -1, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=1,  proc=3, np=4, nblk=4, iflag=-1)==0", dev_read(di) == 0);
  // iflag>0: next local on proc 3 after idx=1 is local 1 (global 13)
  k_localIndex<<<1,1>>>(1, 3, 4, 4,  1, di); HIP_CHECK(hipDeviceSynchronize());
  REPORT("local_index(idx=1,  proc=3, np=4, nblk=4, iflag=1)==1", dev_read(di) == 1);

  HIP_CHECK(hipFree(di));
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for common_device_functions.h (ROCm/HIP) ===\n");

  test_elpaHostNumberFromInt();
  test_elpaDeviceSign();
  test_elpaDeviceNumber();
  test_elpaDeviceNumberFromRealImag();
  test_elpaDeviceAdd();
  test_elpaDeviceSubtract();
  test_elpaDeviceMultiply();
  test_elpaDeviceDivide();
  test_elpaDeviceSqrt();
  test_elpaDeviceComplexConjugate();
  test_elpaDeviceRealPart();
  test_elpaDeviceImagPart();
  test_elpaDeviceEqual();
  test_elpaDeviceEqualBool();
  test_atomicAdd();
  test_pcol_prow();
  test_localIndex();

  printf("\n=== Summary: %d failure(s) ===\n", g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else  /* !WITH_AMD_GPU_VERSION */

int main(void)
{
  fprintf(stderr, "Error: this test requires WITH_AMD_GPU_VERSION\n");
  abort();
}

#endif  /* WITH_AMD_GPU_VERSION */
#endif  /* WITH_UNIT_TESTS */
