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

#include <sycl/sycl.hpp>
#include <complex>
#include <cstring>
#include <type_traits>
#include <cstdint>

#include "src/GPU/common_device_functions.h"
#include "src/GPU/gpu_to_cuda_and_hip_interface.h"

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-70s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

static bool neq (double a, double b) { return fabs (a - b) < 1e-12; }
static bool neqf(float  a, float  b) { return fabsf(a - b) < 1e-5f; }

template <typename T>
static T dev_read(sycl::queue &q, T *d) {
  T h;
  q.memcpy(&h, d, sizeof(T)).wait();
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
  REPORT("double_complex(7).real==7.0", neq(dc.real(), 7.0));
  REPORT("double_complex(7).imag==0.0", neq(dc.imag(), 0.0));

  float_complex fc = elpaHostNumberFromInt<float_complex>(7);
  REPORT("float_complex(7).real==7.0f", neqf(fc.real(), 7.0f));
  REPORT("float_complex(7).imag==0.0f", neqf(fc.imag(), 0.0f));
}

// ============================================================
// elpaDeviceSign  (double and float only — no complex specialization)
// ============================================================
static void test_elpaDeviceSign(sycl::queue &q)
{
  printf("\nelpaDeviceSign:\n");

  double *dd = sycl::malloc_device<double>(1, q);
  float  *df = sycl::malloc_device<float> (1, q);

  { double a=3.0,  b=1.0;  q.single_task([=](){ *dd=elpaDeviceSign(a,b); }).wait(); }
  REPORT("double: sign( 3, 1)== 3", neq(dev_read(q,dd),  3.0));
  { double a=3.0,  b=0.0;  q.single_task([=](){ *dd=elpaDeviceSign(a,b); }).wait(); }
  REPORT("double: sign( 3, 0)== 3 (b==0 treated as >=0)", neq(dev_read(q,dd),  3.0));
  { double a=3.0,  b=-1.0; q.single_task([=](){ *dd=elpaDeviceSign(a,b); }).wait(); }
  REPORT("double: sign( 3,-1)==-3", neq(dev_read(q,dd), -3.0));
  { double a=-5.0, b=2.0;  q.single_task([=](){ *dd=elpaDeviceSign(a,b); }).wait(); }
  REPORT("double: sign(-5, 2)== 5", neq(dev_read(q,dd),  5.0));

  { float a=3.0f,  b=1.0f;  q.single_task([=](){ *df=elpaDeviceSign(a,b); }).wait(); }
  REPORT("float:  sign( 3, 1)== 3", neqf(dev_read(q,df),  3.0f));
  { float a=3.0f,  b=-1.0f; q.single_task([=](){ *df=elpaDeviceSign(a,b); }).wait(); }
  REPORT("float:  sign( 3,-1)==-3", neqf(dev_read(q,df), -3.0f));

  sycl::free(dd, q); sycl::free(df, q);
}

// ============================================================
// elpaDeviceNumber
// ============================================================
static void test_elpaDeviceNumber(sycl::queue &q)
{
  printf("\nelpaDeviceNumber:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceNumber<double>(5.0); }).wait();
  REPORT("double(5.0)==5.0", neq(dev_read(q,dd), 5.0));

  q.single_task([=](){ *df = elpaDeviceNumber<float>(5.0); }).wait();
  REPORT("float(5.0)==5.0f", neqf(dev_read(q,df), 5.0f));

  q.single_task([=](){ *dc = elpaDeviceNumber<double_complex>(5.0); }).wait();
  { double_complex h = dev_read(q,dc);
    REPORT("double_complex(5.0).real==5.0", neq(h.real(), 5.0));
    REPORT("double_complex(5.0).imag==0.0", neq(h.imag(), 0.0)); }

  q.single_task([=](){ *fc = elpaDeviceNumber<float_complex>(5.0); }).wait();
  { float_complex h = dev_read(q,fc);
    REPORT("float_complex(5.0).real==5.0f", neqf(h.real(), 5.0f));
    REPORT("float_complex(5.0).imag==0.0f", neqf(h.imag(), 0.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceNumberFromRealImag
// ============================================================
static void test_elpaDeviceNumberFromRealImag(sycl::queue &q)
{
  printf("\nelpaDeviceNumberFromRealImag:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceNumberFromRealImag<double>(3.0, 4.0); }).wait();
  REPORT("double: fromRI(3,4)==3.0 (real part only)", neq(dev_read(q,dd), 3.0));

  q.single_task([=](){ *df = elpaDeviceNumberFromRealImag<float>(3.0f, 4.0f); }).wait();
  REPORT("float: fromRI(3,4)==3.0f (real part only)", neqf(dev_read(q,df), 3.0f));

  q.single_task([=](){ *dc = elpaDeviceNumberFromRealImag<double_complex>(3.0, 4.0); }).wait();
  { double_complex h = dev_read(q,dc);
    REPORT("double_complex: fromRI(3,4).real==3.0", neq(h.real(), 3.0));
    REPORT("double_complex: fromRI(3,4).imag==4.0", neq(h.imag(), 4.0)); }

  q.single_task([=](){ *fc = elpaDeviceNumberFromRealImag<float_complex>(3.0f, 4.0f); }).wait();
  { float_complex h = dev_read(q,fc);
    REPORT("float_complex: fromRI(3,4).real==3.0f", neqf(h.real(), 3.0f));
    REPORT("float_complex: fromRI(3,4).imag==4.0f", neqf(h.imag(), 4.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceAdd
// ============================================================
static void test_elpaDeviceAdd(sycl::queue &q)
{
  printf("\nelpaDeviceAdd:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceAdd(1.0, 2.0); }).wait();
  REPORT("double: 1+2==3", neq(dev_read(q,dd), 3.0));

  q.single_task([=](){ *df = elpaDeviceAdd(1.0f, 2.0f); }).wait();
  REPORT("float: 1+2==3",  neqf(dev_read(q,df), 3.0f));

  { double_complex a(1.0,2.0), b(3.0,4.0);
    q.single_task([=](){ *dc = elpaDeviceAdd(a, b); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: (1+2i)+(3+4i).real==4", neq(h.real(), 4.0));
    REPORT("double_complex: (1+2i)+(3+4i).imag==6", neq(h.imag(), 6.0)); }

  { float_complex a(1.0f,2.0f), b(3.0f,4.0f);
    q.single_task([=](){ *fc = elpaDeviceAdd(a, b); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: (1+2i)+(3+4i).real==4", neqf(h.real(), 4.0f));
    REPORT("float_complex: (1+2i)+(3+4i).imag==6", neqf(h.imag(), 6.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceSubtract
// ============================================================
static void test_elpaDeviceSubtract(sycl::queue &q)
{
  printf("\nelpaDeviceSubtract:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceSubtract(5.0, 2.0); }).wait();
  REPORT("double: 5-2==3", neq(dev_read(q,dd), 3.0));

  q.single_task([=](){ *df = elpaDeviceSubtract(5.0f, 2.0f); }).wait();
  REPORT("float: 5-2==3",  neqf(dev_read(q,df), 3.0f));

  { double_complex a(5.0,6.0), b(1.0,2.0);
    q.single_task([=](){ *dc = elpaDeviceSubtract(a, b); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: (5+6i)-(1+2i).real==4", neq(h.real(), 4.0));
    REPORT("double_complex: (5+6i)-(1+2i).imag==4", neq(h.imag(), 4.0)); }

  { float_complex a(5.0f,6.0f), b(1.0f,2.0f);
    q.single_task([=](){ *fc = elpaDeviceSubtract(a, b); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: (5+6i)-(1+2i).real==4", neqf(h.real(), 4.0f));
    REPORT("float_complex: (5+6i)-(1+2i).imag==4", neqf(h.imag(), 4.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceMultiply
// ============================================================
static void test_elpaDeviceMultiply(sycl::queue &q)
{
  printf("\nelpaDeviceMultiply:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceMultiply(3.0, 4.0); }).wait();
  REPORT("double: 3*4==12", neq(dev_read(q,dd), 12.0));

  q.single_task([=](){ *df = elpaDeviceMultiply(3.0f, 4.0f); }).wait();
  REPORT("float: 3*4==12",  neqf(dev_read(q,df), 12.0f));

  { // (1+2i)*(3+4i) = -5+10i
    double_complex a(1.0,2.0), b(3.0,4.0);
    q.single_task([=](){ *dc = elpaDeviceMultiply(a, b); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: (1+2i)*(3+4i).real==-5", neq(h.real(), -5.0));
    REPORT("double_complex: (1+2i)*(3+4i).imag==10", neq(h.imag(), 10.0)); }

  { float_complex a(1.0f,2.0f), b(3.0f,4.0f);
    q.single_task([=](){ *fc = elpaDeviceMultiply(a, b); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: (1+2i)*(3+4i).real==-5", neqf(h.real(), -5.0f));
    REPORT("float_complex: (1+2i)*(3+4i).imag==10", neqf(h.imag(), 10.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceDivide
// ============================================================
static void test_elpaDeviceDivide(sycl::queue &q)
{
  printf("\nelpaDeviceDivide:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceDivide(10.0, 4.0); }).wait();
  REPORT("double: 10/4==2.5", neq(dev_read(q,dd), 2.5));

  q.single_task([=](){ *df = elpaDeviceDivide(10.0f, 4.0f); }).wait();
  REPORT("float: 10/4==2.5",  neqf(dev_read(q,df), 2.5f));

  { // (4+2i)/(3+1i) = 1.4+0.2i
    double_complex a(4.0,2.0), b(3.0,1.0);
    q.single_task([=](){ *dc = elpaDeviceDivide(a, b); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: (4+2i)/(3+1i).real==1.4", neq(h.real(), 1.4));
    REPORT("double_complex: (4+2i)/(3+1i).imag==0.2", neq(h.imag(), 0.2)); }

  { float_complex a(4.0f,2.0f), b(3.0f,1.0f);
    q.single_task([=](){ *fc = elpaDeviceDivide(a, b); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: (4+2i)/(3+1i).real==1.4", neqf(h.real(), 1.4f));
    REPORT("float_complex: (4+2i)/(3+1i).imag==0.2", neqf(h.imag(), 0.2f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceSqrt  (double and float only)
// ============================================================
static void test_elpaDeviceSqrt(sycl::queue &q)
{
  printf("\nelpaDeviceSqrt:\n");

  double *dd = sycl::malloc_device<double>(1, q);
  float  *df = sycl::malloc_device<float> (1, q);

  q.single_task([=](){ *dd = elpaDeviceSqrt(9.0); }).wait();
  REPORT("double: sqrt(9)==3",          neq (dev_read(q,dd), 3.0));
  q.single_task([=](){ *dd = elpaDeviceSqrt(2.0); }).wait();
  REPORT("double: sqrt(2)==1.41421356", neq (dev_read(q,dd), sqrt(2.0)));

  q.single_task([=](){ *df = elpaDeviceSqrt(9.0f); }).wait();
  REPORT("float: sqrt(9)==3",           neqf(dev_read(q,df), 3.0f));
  q.single_task([=](){ *df = elpaDeviceSqrt(2.0f); }).wait();
  REPORT("float: sqrt(2)==1.41421f",    neqf(dev_read(q,df), sqrtf(2.0f)));

  sycl::free(dd, q); sycl::free(df, q);
}

// ============================================================
// elpaDeviceComplexConjugate
// ============================================================
static void test_elpaDeviceComplexConjugate(sycl::queue &q)
{
  printf("\nelpaDeviceComplexConjugate:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceComplexConjugate(5.0); }).wait();
  REPORT("double: conj(5)==5", neq(dev_read(q,dd), 5.0));

  q.single_task([=](){ *df = elpaDeviceComplexConjugate(5.0f); }).wait();
  REPORT("float: conj(5)==5",  neqf(dev_read(q,df), 5.0f));

  { double_complex a(3.0, 4.0);
    q.single_task([=](){ *dc = elpaDeviceComplexConjugate(a); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: conj(3+4i).real== 3", neq(h.real(),  3.0));
    REPORT("double_complex: conj(3+4i).imag==-4", neq(h.imag(), -4.0)); }

  { float_complex a(3.0f, 4.0f);
    q.single_task([=](){ *fc = elpaDeviceComplexConjugate(a); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: conj(3+4i).real== 3", neqf(h.real(),  3.0f));
    REPORT("float_complex: conj(3+4i).imag==-4", neqf(h.imag(), -4.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceRealPart
// ============================================================
static void test_elpaDeviceRealPart(sycl::queue &q)
{
  printf("\nelpaDeviceRealPart:\n");

  double *od = sycl::malloc_device<double>(1, q);
  float  *of = sycl::malloc_device<float> (1, q);

  q.single_task([=](){ *od = elpaDeviceRealPart(7.0); }).wait();
  REPORT("double: realPart(7.0)==7.0", neq(dev_read(q,od), 7.0));

  q.single_task([=](){ *of = elpaDeviceRealPart(7.0f); }).wait();
  REPORT("float: realPart(7.0f)==7.0f", neqf(dev_read(q,of), 7.0f));

  { double_complex a(3.0, 4.0);
    q.single_task([=](){ *od = elpaDeviceRealPart(a); }).wait();
    REPORT("double_complex: realPart(3+4i)==3.0", neq(dev_read(q,od), 3.0)); }

  { float_complex a(3.0f, 4.0f);
    q.single_task([=](){ *of = elpaDeviceRealPart(a); }).wait();
    REPORT("float_complex: realPart(3+4i)==3.0f", neqf(dev_read(q,of), 3.0f)); }

  sycl::free(od, q); sycl::free(of, q);
}

// ============================================================
// elpaDeviceImagPart
// ============================================================
static void test_elpaDeviceImagPart(sycl::queue &q)
{
  printf("\nelpaDeviceImagPart:\n");

  double *od = sycl::malloc_device<double>(1, q);
  float  *of = sycl::malloc_device<float> (1, q);

  q.single_task([=](){ *od = elpaDeviceImagPart(7.0); }).wait();
  REPORT("double: imagPart(7.0)==0.0", neq(dev_read(q,od), 0.0));

  q.single_task([=](){ *of = (float)elpaDeviceImagPart(7.0f); }).wait();
  REPORT("float: imagPart(7.0f)==0.0f", neqf(dev_read(q,of), 0.0f));

  { double_complex a(3.0, 4.0);
    q.single_task([=](){ *od = elpaDeviceImagPart(a); }).wait();
    REPORT("double_complex: imagPart(3+4i)==4.0", neq(dev_read(q,od), 4.0)); }

  { float_complex a(3.0f, 4.0f);
    q.single_task([=](){ *of = elpaDeviceImagPart(a); }).wait();
    REPORT("float_complex: imagPart(3+4i)==4.0f", neqf(dev_read(q,of), 4.0f)); }

  sycl::free(od, q); sycl::free(of, q);
}

// ============================================================
// elpaDeviceEqual
// ============================================================
static void test_elpaDeviceEqual(sycl::queue &q)
{
  printf("\nelpaDeviceEqual:\n");

  double         *dd = sycl::malloc_device<double>        (1, q);
  float          *df = sycl::malloc_device<float>         (1, q);
  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  q.single_task([=](){ *dd = elpaDeviceEqual(3.0, 3.0); }).wait();
  REPORT("double: equal(3,3)==1.0", neq(dev_read(q,dd), 1.0));
  q.single_task([=](){ *dd = elpaDeviceEqual(3.0, 4.0); }).wait();
  REPORT("double: equal(3,4)==0.0", neq(dev_read(q,dd), 0.0));

  q.single_task([=](){ *df = elpaDeviceEqual(3.0f, 3.0f); }).wait();
  REPORT("float: equal(3,3)==1.0f", neqf(dev_read(q,df), 1.0f));
  q.single_task([=](){ *df = elpaDeviceEqual(3.0f, 4.0f); }).wait();
  REPORT("float: equal(3,4)==0.0f", neqf(dev_read(q,df), 0.0f));

  { double_complex a(1.0,2.0);
    q.single_task([=](){ *dc = elpaDeviceEqual(a, a); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: equal(z,z).real==1.0", neq(h.real(), 1.0));
    REPORT("double_complex: equal(z,z).imag==0.0", neq(h.imag(), 0.0));
    double_complex b(2.0,1.0);
    q.single_task([=](){ *dc = elpaDeviceEqual(a, b); }).wait();
    h = dev_read(q,dc);
    REPORT("double_complex: equal(z1,z2).real==0.0", neq(h.real(), 0.0)); }

  { float_complex a(1.0f,2.0f);
    q.single_task([=](){ *fc = elpaDeviceEqual(a, a); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: equal(z,z).real==1.0f", neqf(h.real(), 1.0f));
    REPORT("float_complex: equal(z,z).imag==0.0f", neqf(h.imag(), 0.0f));
    float_complex b(2.0f,1.0f);
    q.single_task([=](){ *fc = elpaDeviceEqual(a, b); }).wait();
    h = dev_read(q,fc);
    REPORT("float_complex: equal(z1,z2).real==0.0f", neqf(h.real(), 0.0f)); }

  sycl::free(dd,q); sycl::free(df,q); sycl::free(dc,q); sycl::free(fc,q);
}

// ============================================================
// elpaDeviceEqualBool
// ============================================================
static void test_elpaDeviceEqualBool(sycl::queue &q)
{
  printf("\nelpaDeviceEqualBool:\n");

  bool *db = sycl::malloc_device<bool>(1, q);

  q.single_task([=](){ *db = elpaDeviceEqualBool(3.0, 3.0); }).wait();
  REPORT("double: equalBool(3,3)==true",  dev_read(q,db) == true);
  q.single_task([=](){ *db = elpaDeviceEqualBool(3.0, 4.0); }).wait();
  REPORT("double: equalBool(3,4)==false", dev_read(q,db) == false);

  q.single_task([=](){ *db = elpaDeviceEqualBool(3.0f, 3.0f); }).wait();
  REPORT("float: equalBool(3,3)==true",   dev_read(q,db) == true);
  q.single_task([=](){ *db = elpaDeviceEqualBool(3.0f, 4.0f); }).wait();
  REPORT("float: equalBool(3,4)==false",  dev_read(q,db) == false);

  { double_complex a(1.0,2.0), b(2.0,1.0);
    q.single_task([=](){ *db = elpaDeviceEqualBool(a, a); }).wait();
    REPORT("double_complex: equalBool(z,z)==true",    dev_read(q,db) == true);
    q.single_task([=](){ *db = elpaDeviceEqualBool(a, b); }).wait();
    REPORT("double_complex: equalBool(z1,z2)==false", dev_read(q,db) == false); }

  { float_complex a(1.0f,2.0f), b(2.0f,1.0f);
    q.single_task([=](){ *db = elpaDeviceEqualBool(a, a); }).wait();
    REPORT("float_complex: equalBool(z,z)==true",     dev_read(q,db) == true);
    q.single_task([=](){ *db = elpaDeviceEqualBool(a, b); }).wait();
    REPORT("float_complex: equalBool(z1,z2)==false",  dev_read(q,db) == false); }

  sycl::free(db, q);
}

// ============================================================
// atomicAdd for double_complex and float_complex
// Two sequential single_task calls accumulate the same value twice.
// ============================================================
static void test_atomicAdd(sycl::queue &q)
{
  printf("\natomicAdd (complex types):\n");

  double_complex *dc = sycl::malloc_device<double_complex>(1, q);
  float_complex  *fc = sycl::malloc_device<float_complex> (1, q);

  { double_complex zero(0.0,0.0), inc(1.5,2.5);
    q.memcpy(dc, &zero, sizeof(double_complex)).wait();
    q.single_task([=](){ atomicAdd(dc, inc); }).wait();
    q.single_task([=](){ atomicAdd(dc, inc); }).wait();
    double_complex h = dev_read(q,dc);
    REPORT("double_complex: atomicAdd x2 with (1.5+2.5i).real==3.0", neq(h.real(), 3.0));
    REPORT("double_complex: atomicAdd x2 with (1.5+2.5i).imag==5.0", neq(h.imag(), 5.0)); }

  { float_complex zero(0.0f,0.0f), inc(1.5f,2.5f);
    q.memcpy(fc, &zero, sizeof(float_complex)).wait();
    q.single_task([=](){ atomicAdd(fc, inc); }).wait();
    q.single_task([=](){ atomicAdd(fc, inc); }).wait();
    float_complex h = dev_read(q,fc);
    REPORT("float_complex: atomicAdd x2 with (1.5+2.5i).real==3.0f", neqf(h.real(), 3.0f));
    REPORT("float_complex: atomicAdd x2 with (1.5+2.5i).imag==5.0f", neqf(h.imag(), 5.0f)); }

  sycl::free(dc, q); sycl::free(fc, q);
}

// ============================================================
// pcol / prow
// ============================================================
static void test_pcol_prow(sycl::queue &q)
{
  printf("\npcol / prow:\n");

  int *di = sycl::malloc_device<int>(1, q);

  q.single_task([=](){ *di = pcol( 1, 4, 4); }).wait();
  REPORT("pcol( 1, nblk=4, np=4)==0", dev_read(q,di) == 0);
  q.single_task([=](){ *di = pcol( 5, 4, 4); }).wait();
  REPORT("pcol( 5, nblk=4, np=4)==1", dev_read(q,di) == 1);
  q.single_task([=](){ *di = pcol( 9, 4, 4); }).wait();
  REPORT("pcol( 9, nblk=4, np=4)==2", dev_read(q,di) == 2);
  q.single_task([=](){ *di = pcol(13, 4, 4); }).wait();
  REPORT("pcol(13, nblk=4, np=4)==3", dev_read(q,di) == 3);
  q.single_task([=](){ *di = pcol(17, 4, 4); }).wait();
  REPORT("pcol(17, nblk=4, np=4)==0 (wraps around)", dev_read(q,di) == 0);

  q.single_task([=](){ *di = prow( 1, 4, 4); }).wait();
  REPORT("prow( 1, nblk=4, np=4)==0", dev_read(q,di) == 0);
  q.single_task([=](){ *di = prow( 8, 4, 4); }).wait();
  REPORT("prow( 8, nblk=4, np=4)==1", dev_read(q,di) == 1);

  sycl::free(di, q);
}

// ============================================================
// local_index
// ============================================================
static void test_localIndex(sycl::queue &q)
{
  printf("\nlocal_index:\n");

  int *di = sycl::malloc_device<int>(1, q);

  q.single_task([=](){ *di = local_index(1,  0, 4, 4, 0); }).wait();
  REPORT("local_index(idx=1,  proc=0, np=4, nblk=4, iflag=0)==1", dev_read(q,di) == 1);
  q.single_task([=](){ *di = local_index(4,  0, 4, 4, 0); }).wait();
  REPORT("local_index(idx=4,  proc=0, np=4, nblk=4, iflag=0)==4", dev_read(q,di) == 4);
  q.single_task([=](){ *di = local_index(5,  1, 4, 4, 0); }).wait();
  REPORT("local_index(idx=5,  proc=1, np=4, nblk=4, iflag=0)==1", dev_read(q,di) == 1);
  q.single_task([=](){ *di = local_index(17, 0, 4, 4, 0); }).wait();
  REPORT("local_index(idx=17, proc=0, np=4, nblk=4, iflag=0)==5", dev_read(q,di) == 5);

  q.single_task([=](){ *di = local_index(6, 0, 4, 4, 0); }).wait();
  REPORT("local_index(idx=6,  proc=0, np=4, nblk=4, iflag=0)==0 (non-local)", dev_read(q,di) == 0);

  q.single_task([=](){ *di = local_index(6, 0, 4, 4, -1); }).wait();
  REPORT("local_index(idx=6,  proc=0, np=4, nblk=4, iflag=-1)==4", dev_read(q,di) == 4);

  q.single_task([=](){ *di = local_index(6, 0, 4, 4, 1); }).wait();
  REPORT("local_index(idx=6,  proc=0, np=4, nblk=4, iflag=1)==5", dev_read(q,di) == 5);

  q.single_task([=](){ *di = local_index(1, 3, 4, 4, -1); }).wait();
  REPORT("local_index(idx=1,  proc=3, np=4, nblk=4, iflag=-1)==0", dev_read(q,di) == 0);
  q.single_task([=](){ *di = local_index(1, 3, 4, 4,  1); }).wait();
  REPORT("local_index(idx=1,  proc=3, np=4, nblk=4, iflag=1)==1",  dev_read(q,di) == 1);

  sycl::free(di, q);
}

// ============================================================
int main(void)
{
  printf("=== Unit tests for common_device_functions.h (SYCL) ===\n");

  sycl::queue q(sycl::default_selector_v);

  test_elpaHostNumberFromInt();
  test_elpaDeviceSign(q);
  test_elpaDeviceNumber(q);
  test_elpaDeviceNumberFromRealImag(q);
  test_elpaDeviceAdd(q);
  test_elpaDeviceSubtract(q);
  test_elpaDeviceMultiply(q);
  test_elpaDeviceDivide(q);
  test_elpaDeviceSqrt(q);
  test_elpaDeviceComplexConjugate(q);
  test_elpaDeviceRealPart(q);
  test_elpaDeviceImagPart(q);
  test_elpaDeviceEqual(q);
  test_elpaDeviceEqualBool(q);
  test_atomicAdd(q);
  test_pcol_prow(q);
  test_localIndex(q);

  printf("\n=== Summary: %d failure(s) ===\n", g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else  /* !WITH_SYCL_GPU_VERSION */

int main(void)
{
  fprintf(stderr, "Error: this test requires WITH_SYCL_GPU_VERSION\n");
  abort();
}

#endif  /* WITH_SYCL_GPU_VERSION */
#endif  /* WITH_UNIT_TESTS */
