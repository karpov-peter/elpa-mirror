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

// Unit tests for the kernels in src/GPU/SYCL/syclUtils.cpp (SYCL backend):
//   launch_my_pack_c_sycl_kernel    (double, float, gpuDoubleComplex, gpuFloatComplex)
//   launch_my_unpack_c_sycl_kernel  (double, float, gpuDoubleComplex, gpuFloatComplex)
//   launch_extract_hh_tau_c_sycl_kernel (double, float, gpuDoubleComplex, gpuFloatComplex)
// Self-contained: kernel bodies copied here to avoid syclCommon.hpp dependency.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <type_traits>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_SYCL_GPU_VERSION

#include <sycl/sycl.hpp>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-72s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

// ============================================================
// Type helpers
// Under SYCL: gpuDoubleComplex = std::complex<double>
//             gpuFloatComplex  = std::complex<float>
//             make_gpuDoubleComplex(r,i) = std::complex<double>(r,i)
//             make_gpuFloatComplex(r,i)  = std::complex<float>(r,i)
// ============================================================

template <typename T>
static T make_val(int i)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>)
        return gpuDoubleComplex{(double)i, (double)(i * 2)};
    else if constexpr (std::is_same_v<T, gpuFloatComplex>)
        return gpuFloatComplex{(float)i, (float)(i * 2)};
    else
        return (T)i;
}

template <typename T>
static T one_val()
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>)
        return gpuDoubleComplex{1.0, 0.0};
    else if constexpr (std::is_same_v<T, gpuFloatComplex>)
        return gpuFloatComplex{1.0f, 0.0f};
    else
        return (T)1;
}

template <typename T>
static T zero_val()
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>)
        return gpuDoubleComplex{0.0, 0.0};
    else if constexpr (std::is_same_v<T, gpuFloatComplex>)
        return gpuFloatComplex{0.0f, 0.0f};
    else
        return (T)0;
}

template <typename T>
static bool vals_equal(T a, T b)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>)
        return a.real() == b.real() && a.imag() == b.imag();
    else
        return a == b;
}

// ============================================================
// Pack kernel body (from syclUtils.cpp)
// ============================================================
template <typename T>
static void pack_kernel_body(T *a, T *row_group,
                              int b_id, int t_id,
                              int stripe_width, int a_dim2, int l_nev,
                              int n_offset, int max_idx,
                              const sycl::nd_item<2> &it)
{
    int dst_ind = b_id * stripe_width + t_id;
    if (dst_ind < max_idx) {
        row_group[dst_ind + (l_nev * it.get_group(1))] =
            a[t_id + (stripe_width * (n_offset + it.get_group(1))) + (b_id * stripe_width * a_dim2)];
    }
}

// ============================================================
// Unpack kernel body (from syclUtils.cpp)
// ============================================================
template <typename T>
static void unpack_kernel_body(T *row_group, T *a,
                                int b_id, int t_id,
                                int stripe_width, int a_dim2, int l_nev,
                                int n_offset, int max_idx,
                                const sycl::nd_item<2> &it)
{
    int src_ind = b_id * stripe_width + t_id;
    if (src_ind < max_idx) {
        a[(t_id + ((n_offset + it.get_group(1)) * stripe_width) + (b_id * stripe_width * a_dim2))] =
            row_group[src_ind + it.get_group(1) * l_nev];
    }
}

// ============================================================
// Test: pack kernel
//
// Parameters: sw=2, ad2=2, sc=2, rc=2, ln=4, n_off=0, max_idx=4
// src[k] = make_val(k+1), k=0..7
// Expected packed layout:
//   expected[0]=v(1), expected[1]=v(2), expected[2]=v(5), expected[3]=v(6),
//   expected[4]=v(3), expected[5]=v(4), expected[6]=v(7), expected[7]=v(8)
// ============================================================
template <typename T>
static void test_pack(sycl::queue &q, const char *tname)
{
    const int sw = 2, ad2 = 2, sc = 2, rc = 2, ln = 4, n_off = 0, max_idx = 4;
    const int src_n = sw * ad2 * sc;  // 8
    const int dst_n = ln * rc;        // 8

    T h_src[8], expected[8];
    for (int i = 0; i < src_n; i++) h_src[i] = make_val<T>(i + 1);

    expected[0] = make_val<T>(1); expected[1] = make_val<T>(2);
    expected[2] = make_val<T>(5); expected[3] = make_val<T>(6);
    expected[4] = make_val<T>(3); expected[5] = make_val<T>(4);
    expected[6] = make_val<T>(7); expected[7] = make_val<T>(8);

    T *d_src = sycl::malloc_device<T>(src_n, q);
    T *d_dst = sycl::malloc_device<T>(dst_n, q);
    q.memcpy(d_src, h_src, src_n * sizeof(T)).wait();
    q.memset(d_dst, 0, dst_n * sizeof(T)).wait();

    // Replicate syclUtils.cpp launch: wgSize = sw (since sw <= maxWgSize for small sw)
    int wgSize = sw;
    sycl::range<2> rLocal  = sycl::range<2>(1, wgSize);
    sycl::range<2> rGlobal = sycl::range<2>(sc, rc * wgSize);

    // Only one i_off iteration since stripe_width/wgSize = sw/sw = 1
    for (int i_off = 0; i_off < sw / wgSize; i_off++) {
        int i_off_cap = i_off;
        q.parallel_for(sycl::nd_range<2>(rGlobal, rLocal),
                       [=](sycl::nd_item<2> it) {
            int b_id = it.get_group(0);
            int t_id = it.get_local_id(1) + i_off_cap * wgSize;
            pack_kernel_body(d_src, d_dst, b_id, t_id, sw, ad2, ln, n_off, max_idx, it);
        });
        q.wait_and_throw();
    }

    T h_dst[8];
    q.memcpy(h_dst, d_dst, dst_n * sizeof(T)).wait();
    sycl::free(d_src, q);
    sycl::free(d_dst, q);

    bool ok = true;
    for (int i = 0; i < dst_n; i++)
        if (!vals_equal(h_dst[i], expected[i])) ok = false;

    char lbl[128];
    snprintf(lbl, sizeof(lbl), "pack kernel <%s>: output matches expected permutation", tname);
    REPORT(lbl, ok);
}

// ============================================================
// Test: unpack kernel (inverse of pack)
//
// Start from the packed layout produced by pack and recover
// the original sequential src values.
// ============================================================
template <typename T>
static void test_unpack(sycl::queue &q, const char *tname)
{
    const int sw = 2, ad2 = 2, sc = 2, rc = 2, ln = 4, n_off = 0, max_idx = 4;
    const int rg_n = ln * rc;        // 8 (row_group)
    const int a_n  = sw * ad2 * sc;  // 8 (a)

    // Packed layout (output of pack test)
    T h_rg[8];
    h_rg[0] = make_val<T>(1); h_rg[1] = make_val<T>(2);
    h_rg[2] = make_val<T>(5); h_rg[3] = make_val<T>(6);
    h_rg[4] = make_val<T>(3); h_rg[5] = make_val<T>(4);
    h_rg[6] = make_val<T>(7); h_rg[7] = make_val<T>(8);

    T *d_rg = sycl::malloc_device<T>(rg_n, q);
    T *d_a  = sycl::malloc_device<T>(a_n,  q);
    q.memcpy(d_rg, h_rg, rg_n * sizeof(T)).wait();
    q.memset(d_a, 0, a_n * sizeof(T)).wait();

    int wgSize = sw;
    sycl::range<2> rLocal  = sycl::range<2>(1, wgSize);
    sycl::range<2> rGlobal = sycl::range<2>(sc, rc * wgSize);

    for (int i_off = 0; i_off < sw / wgSize; i_off++) {
        int i_off_cap = i_off;
        q.parallel_for(sycl::nd_range<2>(rGlobal, rLocal),
                       [=](sycl::nd_item<2> it) {
            int b_id = it.get_group(0);
            int t_id = it.get_local_id(1) + i_off_cap * (int)it.get_local_range(1);
            unpack_kernel_body(d_rg, d_a, b_id, t_id, sw, ad2, ln, n_off, max_idx, it);
        });
    }
    q.wait_and_throw();

    T h_a[8];
    q.memcpy(h_a, d_a, a_n * sizeof(T)).wait();
    sycl::free(d_rg, q);
    sycl::free(d_a, q);

    bool ok = true;
    for (int i = 0; i < a_n; i++)
        if (!vals_equal(h_a[i], make_val<T>(i + 1))) ok = false;

    char lbl[128];
    snprintf(lbl, sizeof(lbl), "unpack kernel <%s>: recovers original sequential values", tname);
    REPORT(lbl, ok);
}

// ============================================================
// Test: extract_hh_tau kernel
//
// n=4 reflectors, nbw=2.
// hh[col*nbw] is the first element of each reflector column.
// hh[col*nbw+1] is padding (must remain unchanged).
//
//   hh_tau[col]  <- hh[col*nbw]            (copy)
//   hh[col*nbw]  <- one_val  if is_zero==0
//                <- zero_val if is_zero!=0
// ============================================================
template <typename T>
static void test_extract_hh_tau_one(sycl::queue &q, const char *tname, int is_zero_val)
{
    const int n = 4, nbw = 2;
    const int hh_n = n * nbw;  // 8

    T h_hh[8], h_tau[4];
    for (int col = 0; col < n; col++) {
        h_hh[col * nbw]     = make_val<T>(col + 1);
        h_hh[col * nbw + 1] = make_val<T>(9);  // padding, must be unchanged
    }

    T *d_hh  = sycl::malloc_device<T>(hh_n, q);
    T *d_tau = sycl::malloc_device<T>(n,    q);
    q.memcpy(d_hh, h_hh, hh_n * sizeof(T)).wait();
    q.memset(d_tau, 0, n * sizeof(T)).wait();

    // Launch as in syclUtils.cpp (use fixed maxWgSize=32 for test)
    const int maxWgSize = 32;
    int numWorkItems = (1 + (n - 1) / maxWgSize) * maxWgSize;

    int is_zero_cap = is_zero_val;
    q.parallel_for(sycl::nd_range<1>(numWorkItems, maxWgSize),
                   [=](sycl::nd_item<1> it) {
        int h_idx = it.get_global_id(0);
        if (h_idx < n) {
            d_tau[h_idx] = d_hh[h_idx * nbw];
            if (is_zero_cap == 0) {
                d_hh[h_idx * nbw] = (T)1.0;
            } else {
                d_hh[h_idx * nbw] = (T)0.0;
            }
        }
    });
    q.wait_and_throw();

    q.memcpy(h_hh,  d_hh,  hh_n * sizeof(T)).wait();
    q.memcpy(h_tau, d_tau, n    * sizeof(T)).wait();
    sycl::free(d_hh,  q);
    sycl::free(d_tau, q);

    // Check hh_tau: must equal original hh[col*nbw] = make_val(col+1)
    bool ok_tau = true;
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_tau[col], make_val<T>(col + 1))) ok_tau = false;

    // Check hh first elements replaced by one_val or zero_val
    T expected_first = (is_zero_val == 0) ? one_val<T>() : zero_val<T>();
    bool ok_hh_first = true;
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_hh[col * nbw], expected_first)) ok_hh_first = false;

    // Padding elements must be untouched
    bool ok_padding = true;
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_hh[col * nbw + 1], make_val<T>(9))) ok_padding = false;

    char lbl_tau[128], lbl_hh[128], lbl_pad[128];
    snprintf(lbl_tau, sizeof(lbl_tau),
             "extract_hh_tau<%s>(is_zero=%d): hh_tau copied correctly", tname, is_zero_val);
    snprintf(lbl_hh, sizeof(lbl_hh),
             "extract_hh_tau<%s>(is_zero=%d): hh first element replaced", tname, is_zero_val);
    snprintf(lbl_pad, sizeof(lbl_pad),
             "extract_hh_tau<%s>(is_zero=%d): padding unchanged", tname, is_zero_val);
    REPORT(lbl_tau, ok_tau);
    REPORT(lbl_hh,  ok_hh_first);
    REPORT(lbl_pad, ok_padding);
}

template <typename T>
static void test_extract_hh_tau(sycl::queue &q, const char *tname)
{
    test_extract_hh_tau_one<T>(q, tname, 0);
    test_extract_hh_tau_one<T>(q, tname, 1);
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for pack/unpack kernels (SYCL) ===\n");

    sycl::queue q(sycl::default_selector_v);

    printf("\nTesting pack kernel:\n");
    test_pack<double>         (q, "double");
    test_pack<float>          (q, "float");
    test_pack<gpuDoubleComplex>(q, "gpuDoubleComplex");
    test_pack<gpuFloatComplex> (q, "gpuFloatComplex");

    printf("\nTesting unpack kernel:\n");
    test_unpack<double>         (q, "double");
    test_unpack<float>          (q, "float");
    test_unpack<gpuDoubleComplex>(q, "gpuDoubleComplex");
    test_unpack<gpuFloatComplex> (q, "gpuFloatComplex");

    printf("\nTesting extract_hh_tau kernel:\n");
    test_extract_hh_tau<double>         (q, "double");
    test_extract_hh_tau<float>          (q, "float");
    test_extract_hh_tau<gpuDoubleComplex>(q, "gpuDoubleComplex");
    test_extract_hh_tau<gpuFloatComplex> (q, "gpuFloatComplex");

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
