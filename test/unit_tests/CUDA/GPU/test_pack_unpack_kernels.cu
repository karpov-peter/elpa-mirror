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

// Unit tests for the kernels in src/GPU/CUDA/cuUtils_template.cu:
//   my_pack_c_cuda_kernel_*
//   my_unpack_c_cuda_kernel_*
//   extract_hh_tau_c_cuda_kernel_*
// All four data types are covered: double, float, cuDoubleComplex, cuFloatComplex.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <type_traits>
#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

// Pull in all type variants — single include suffices (C++ templates handle all types).
#include "../../../../src/GPU/CUDA/cuUtils_template.cu"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---- type helpers ----

template <typename T>
static T make_val(int i)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>)
        return make_cuDoubleComplex((double)i, (double)(i * 2));
    else if constexpr (std::is_same_v<T, cuFloatComplex>)
        return make_cuFloatComplex((float)i, (float)(i * 2));
    else
        return (T)i;
}

#ifdef WANT_HALF_PRECISION_REAL
template <> __half make_val<__half>(int i) { return __float2half((float)i); }
#endif
#ifdef WANT_HALF_PRECISION_COMPLEX
template <> __half2 make_val<__half2>(int i) { return make_half2(__float2half((float)i), __float2half((float)(i * 2))); }
#endif

template <typename T> static T one_val()
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>) return make_cuDoubleComplex(1.0, 0.0);
    else if constexpr (std::is_same_v<T, cuFloatComplex>) return make_cuFloatComplex(1.0f, 0.0f);
    else return (T)1;
}

#ifdef WANT_HALF_PRECISION_REAL
template <> __half one_val<__half>() { return __float2half(1.0f); }
#endif
#ifdef WANT_HALF_PRECISION_COMPLEX
template <> __half2 one_val<__half2>() { return make_half2(__float2half(1.0f), __float2half(0.0f)); }
#endif

template <typename T> static T zero_val()
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>) return make_cuDoubleComplex(0.0, 0.0);
    else if constexpr (std::is_same_v<T, cuFloatComplex>) return make_cuFloatComplex(0.0f, 0.0f);
    else return (T)0;
}

#ifdef WANT_HALF_PRECISION_REAL
template <> __half zero_val<__half>() { return __float2half(0.0f); }
#endif
#ifdef WANT_HALF_PRECISION_COMPLEX
template <> __half2 zero_val<__half2>() { return make_half2(__float2half(0.0f), __float2half(0.0f)); }
#endif

template <typename T>
static bool vals_equal(T a, T b)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

#ifdef WANT_HALF_PRECISION_REAL
template <> bool vals_equal<__half>(__half a, __half b) {
    return fabsf(__half2float(a) - __half2float(b)) < 0.02f * (fabsf(__half2float(b)) + 1.0f);
}
#endif
#ifdef WANT_HALF_PRECISION_COMPLEX
template <> bool vals_equal<__half2>(__half2 a, __half2 b) {
    return fabsf(__half2float(a.x) - __half2float(b.x)) < 0.02f * (fabsf(__half2float(b.x)) + 1.0f) &&
           fabsf(__half2float(a.y) - __half2float(b.y)) < 0.02f * (fabsf(__half2float(b.y)) + 1.0f);
}
#endif

// ---- launcher dispatch overloads ----

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      double *a, double *rg, cudaStream_t s)
{ launch_my_pack_c_cuda_kernel_real_double(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      float *a, float *rg, cudaStream_t s)
{ launch_my_pack_c_cuda_kernel_real_single(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      cuDoubleComplex *a, cuDoubleComplex *rg, cudaStream_t s)
{ launch_my_pack_c_cuda_kernel_complex_double(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      cuFloatComplex *a, cuFloatComplex *rg, cudaStream_t s)
{ launch_my_pack_c_cuda_kernel_complex_single(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        double *rg, double *a, cudaStream_t s)
{ launch_my_unpack_c_cuda_kernel_real_double(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        float *rg, float *a, cudaStream_t s)
{ launch_my_unpack_c_cuda_kernel_real_single(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        cuDoubleComplex *rg, cuDoubleComplex *a, cudaStream_t s)
{ launch_my_unpack_c_cuda_kernel_complex_double(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        cuFloatComplex *rg, cuFloatComplex *a, cudaStream_t s)
{ launch_my_unpack_c_cuda_kernel_complex_single(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_extract(double *hh, double *tau, int nbw, int n, int val, cudaStream_t s)
{ launch_extract_hh_tau_c_cuda_kernel_real_double(hh, tau, nbw, n, val, s); }

static void call_extract(float *hh, float *tau, int nbw, int n, int val, cudaStream_t s)
{ launch_extract_hh_tau_c_cuda_kernel_real_single(hh, tau, nbw, n, val, s); }

static void call_extract(cuDoubleComplex *hh, cuDoubleComplex *tau, int nbw, int n, int val, cudaStream_t s)
{ launch_extract_hh_tau_c_cuda_kernel_complex_double(hh, tau, nbw, n, val, s); }

static void call_extract(cuFloatComplex *hh, cuFloatComplex *tau, int nbw, int n, int val, cudaStream_t s)
{ launch_extract_hh_tau_c_cuda_kernel_complex_single(hh, tau, nbw, n, val, s); }

#ifdef WANT_HALF_PRECISION_REAL
static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      __half *a, __half *rg, cudaStream_t s)
{ launch_my_pack_c_cuda_kernel_real_half(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        __half *rg, __half *a, cudaStream_t s)
{ launch_my_unpack_c_cuda_kernel_real_half(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_extract(__half *hh, __half *tau, int nbw, int n, int val, cudaStream_t s)
{ launch_extract_hh_tau_c_cuda_kernel_real_half(hh, tau, nbw, n, val, s); }
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      __half2 *a, __half2 *rg, cudaStream_t s)
{ launch_my_pack_c_cuda_kernel_complex_half(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        __half2 *rg, __half2 *a, cudaStream_t s)
{ launch_my_unpack_c_cuda_kernel_complex_half(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_extract(__half2 *hh, __half2 *tau, int nbw, int n, int val, cudaStream_t s)
{ launch_extract_hh_tau_c_cuda_kernel_complex_half(hh, tau, nbw, n, val, s); }
#endif

// ============================================================
// Test: my_pack_c_cuda_kernel
//
// Parameters: stripe_width=2, a_dim2=2, stripe_count=2,
//             row_count=2, l_nev=4, n_offset=0, max_idx=4
//
// src layout [sw=2, ad2=2, sc=2]:  src[t + 2*row + 4*b]
// dst layout:                      dst[2*b + t + 4*row]
//
// With src[k] = make_val(k+1), k=0..7:
//   (row=0,b=0): dst[0..1] = src[0..1] = {v(1),v(2)}
//   (row=0,b=1): dst[2..3] = src[4..5] = {v(5),v(6)}
//   (row=1,b=0): dst[4..5] = src[2..3] = {v(3),v(4)}
//   (row=1,b=1): dst[6..7] = src[6..7] = {v(7),v(8)}
// ============================================================
template <typename T>
static int run_pack_test(const char *type_name)
{
    const int sw = 2, ad2 = 2, sc = 2, rc = 2, ln = 4, n_off = 0, max_idx = 4;
    const int src_n = sw * ad2 * sc;  /* 8 */
    const int dst_n = ln * rc;        /* 8 */

    T h_src[8], h_dst[8], expected[8];
    for (int i = 0; i < src_n; i++) h_src[i] = make_val<T>(i + 1);

    expected[0] = make_val<T>(1); expected[1] = make_val<T>(2);
    expected[2] = make_val<T>(5); expected[3] = make_val<T>(6);
    expected[4] = make_val<T>(3); expected[5] = make_val<T>(4);
    expected[6] = make_val<T>(7); expected[7] = make_val<T>(8);

    T *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, src_n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, src_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, dst_n * sizeof(T)));

    call_pack(rc, n_off, max_idx, sw, ad2, sc, ln, d_src, d_dst, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, dst_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    int failures = 0;
    for (int i = 0; i < dst_n; i++)
        if (!vals_equal(h_dst[i], expected[i])) { failures++; }

    printf("  my_pack_c_cuda_kernel   <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: my_unpack_c_cuda_kernel  (inverse of pack)
//
// Start from the permuted layout produced by pack and recover
// the original sequential src values.
// ============================================================
template <typename T>
static int run_unpack_test(const char *type_name)
{
    const int sw = 2, ad2 = 2, sc = 2, rc = 2, ln = 4, n_off = 0, max_idx = 4;
    const int rg_n = ln * rc;        /* row_group = 8 */
    const int a_n  = sw * ad2 * sc;  /* a = 8 */

    /* Packed layout (output of pack test) */
    T h_rg[8];
    h_rg[0] = make_val<T>(1); h_rg[1] = make_val<T>(2);
    h_rg[2] = make_val<T>(5); h_rg[3] = make_val<T>(6);
    h_rg[4] = make_val<T>(3); h_rg[5] = make_val<T>(4);
    h_rg[6] = make_val<T>(7); h_rg[7] = make_val<T>(8);

    T *d_rg, *d_a;
    CUDA_CHECK(cudaMalloc(&d_rg, rg_n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_a,  a_n  * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_rg, h_rg, rg_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_a, 0, a_n * sizeof(T)));

    call_unpack(rc, n_off, max_idx, sw, ad2, sc, ln, d_rg, d_a, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_a[8];
    CUDA_CHECK(cudaMemcpy(h_a, d_a, a_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rg));
    CUDA_CHECK(cudaFree(d_a));

    int failures = 0;
    for (int i = 0; i < a_n; i++)
        if (!vals_equal(h_a[i], make_val<T>(i + 1))) { failures++; }

    printf("  my_unpack_c_cuda_kernel <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: extract_hh_tau_c_cuda_kernel
//
// n=4 reflectors, nbw=2.  hh[col*nbw] is the first element of
// each reflector column; hh[col*nbw+1] is padding (untouched).
//
//   hh_tau[col]    ← hh[col*nbw]          (copy)
//   hh[col*nbw]    ← one_val  if is_zero==0
//                  ← zero_val if is_zero!=0
// ============================================================
template <typename T>
static int run_extract_hh_tau_one(const char *type_name, int is_zero_val)
{
    const int n = 4, nbw = 2;
    const int hh_n = n * nbw;  /* 8 */

    T h_hh[8], h_tau[4];
    for (int col = 0; col < n; col++)
    {
        h_hh[col * nbw]     = make_val<T>(col + 1);
        h_hh[col * nbw + 1] = make_val<T>(9);   /* padding, should be unchanged */
    }

    T *d_hh, *d_tau;
    CUDA_CHECK(cudaMalloc(&d_hh,  hh_n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tau, n    * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_hh, h_hh, hh_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tau, 0, n * sizeof(T)));

    call_extract(d_hh, d_tau, nbw, n, is_zero_val, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hh,  d_hh,  hh_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tau, d_tau, n    * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hh));
    CUDA_CHECK(cudaFree(d_tau));

    int failures = 0;
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_tau[col], make_val<T>(col + 1))) { failures++; }

    T expected_first = (is_zero_val == 0) ? one_val<T>() : zero_val<T>();
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_hh[col * nbw], expected_first)) { failures++; }

    /* Padding elements must be untouched */
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_hh[col * nbw + 1], make_val<T>(9))) { failures++; }

    printf("  extract_hh_tau_c_cuda_kernel<%s>(is_zero=%d): %s\n",
           type_name, is_zero_val, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

template <typename T>
static int run_extract_hh_tau_test(const char *type_name)
{
    int failures = 0;
    failures += run_extract_hh_tau_one<T>(type_name, 0);
    failures += run_extract_hh_tau_one<T>(type_name, 1);
    return failures;
}

// ============================================================
int main(void)
{
    int failures = 0;

    printf("=== Unit tests for pack/unpack/extract_hh_tau CUDA kernels ===\n\n");
    printf("Testing my_pack_c_cuda_kernel:\n");
    failures += run_pack_test<double>         ("double");
    failures += run_pack_test<float>          ("float");
    failures += run_pack_test<cuDoubleComplex>("cuDoubleComplex");
    failures += run_pack_test<cuFloatComplex> ("cuFloatComplex");

    printf("\nTesting my_unpack_c_cuda_kernel:\n");
    failures += run_unpack_test<double>         ("double");
    failures += run_unpack_test<float>          ("float");
    failures += run_unpack_test<cuDoubleComplex>("cuDoubleComplex");
    failures += run_unpack_test<cuFloatComplex> ("cuFloatComplex");

    printf("\nTesting extract_hh_tau_c_cuda_kernel:\n");
    failures += run_extract_hh_tau_test<double>         ("double");
    failures += run_extract_hh_tau_test<float>          ("float");
    failures += run_extract_hh_tau_test<cuDoubleComplex>("cuDoubleComplex");
    failures += run_extract_hh_tau_test<cuFloatComplex> ("cuFloatComplex");

#ifdef WANT_HALF_PRECISION_REAL
    printf("\nTesting my_pack_c_cuda_kernel (half real):\n");
    failures += run_pack_test<__half>("__half");
    printf("\nTesting my_unpack_c_cuda_kernel (half real):\n");
    failures += run_unpack_test<__half>("__half");
    printf("\nTesting extract_hh_tau_c_cuda_kernel (half real):\n");
    failures += run_extract_hh_tau_test<__half>("__half");
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
    printf("\nTesting my_pack_c_cuda_kernel (half complex):\n");
    failures += run_pack_test<__half2>("__half2");
    printf("\nTesting my_unpack_c_cuda_kernel (half complex):\n");
    failures += run_unpack_test<__half2>("__half2");
    printf("\nTesting extract_hh_tau_c_cuda_kernel (half complex):\n");
    failures += run_extract_hh_tau_test<__half2>("__half2");
#endif

    printf("\n=== Summary: %d failure(s) ===\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
    fprintf(stderr, "Error: this test should only be called when the ELPA build is with NVIDIA GPU version enabled\n");
    abort();
}

#endif /* WITH_NVIDIA_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
