// Unit tests for the kernels in src/GPU/ROCm/hipUtils_template.h:
//   my_pack_c_hip_kernel_*
//   my_unpack_c_hip_kernel_*
//   extract_hh_tau_c_hip_kernel_*
// All four data types are covered: double, float, gpuDoubleComplex, gpuFloatComplex.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_AMD_GPU_VERSION

#include <stdint.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <type_traits>
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

// Pull in all four type variants using the same macro sequence as the ROCm build.
#define REALCASE 1
#undef COMPLEXCASE
#define DOUBLE_PRECISION_REAL 1
#include "../../../../src/GPU/ROCm/hipUtils_template.h"
#undef DOUBLE_PRECISION_REAL

#undef DOUBLE_PRECISION_REAL
#include "../../../../src/GPU/ROCm/hipUtils_template.h"

#define COMPLEXCASE 1
#undef REALCASE
#define DOUBLE_PRECISION_COMPLEX 1
#include "../../../../src/GPU/ROCm/hipUtils_template.h"
#undef DOUBLE_PRECISION_COMPLEX

#undef DOUBLE_PRECISION_COMPLEX
#include "../../../../src/GPU/ROCm/hipUtils_template.h"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _err = (call);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s\n",                             \
              __FILE__, __LINE__, hipGetErrorString(_err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---- type helpers ----

template <typename T>
static T make_val(int i)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>)
        return make_gpuDoubleComplex((double)i, (double)(i * 2));
    else if constexpr (std::is_same_v<T, gpuFloatComplex>)
        return make_gpuFloatComplex((float)i, (float)(i * 2));
    else
        return (T)i;
}

template <typename T> static T one_val()
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>) return make_gpuDoubleComplex(1.0, 0.0);
    else if constexpr (std::is_same_v<T, gpuFloatComplex>) return make_gpuFloatComplex(1.0f, 0.0f);
    else return (T)1;
}

template <typename T> static T zero_val()
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>) return make_gpuDoubleComplex(0.0, 0.0);
    else if constexpr (std::is_same_v<T, gpuFloatComplex>) return make_gpuFloatComplex(0.0f, 0.0f);
    else return (T)0;
}

template <typename T>
static bool vals_equal(T a, T b)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

// ---- launcher dispatch overloads ----

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      double *a, double *rg, hipStream_t s)
{ launch_my_pack_c_hip_kernel_real_double(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      float *a, float *rg, hipStream_t s)
{ launch_my_pack_c_hip_kernel_real_single(rc, n_off, max_idx, sw, ad2, sc, ln, a, rg, s); }

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      gpuDoubleComplex *a, gpuDoubleComplex *rg, hipStream_t s)
{ launch_my_pack_c_hip_kernel_complex_double(rc, n_off, max_idx, sw, ad2, sc, ln,
                                              (hipDoubleComplex *)a, (hipDoubleComplex *)rg, s); }

static void call_pack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                      gpuFloatComplex *a, gpuFloatComplex *rg, hipStream_t s)
{ launch_my_pack_c_hip_kernel_complex_single(rc, n_off, max_idx, sw, ad2, sc, ln,
                                              (hipFloatComplex *)a, (hipFloatComplex *)rg, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        double *rg, double *a, hipStream_t s)
{ launch_my_unpack_c_hip_kernel_real_double(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        float *rg, float *a, hipStream_t s)
{ launch_my_unpack_c_hip_kernel_real_single(rc, n_off, max_idx, sw, ad2, sc, ln, rg, a, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        gpuDoubleComplex *rg, gpuDoubleComplex *a, hipStream_t s)
{ launch_my_unpack_c_hip_kernel_complex_double(rc, n_off, max_idx, sw, ad2, sc, ln,
                                                (hipDoubleComplex *)rg, (hipDoubleComplex *)a, s); }

static void call_unpack(int rc, int n_off, int max_idx, int sw, int ad2, int sc, int ln,
                        gpuFloatComplex *rg, gpuFloatComplex *a, hipStream_t s)
{ launch_my_unpack_c_hip_kernel_complex_single(rc, n_off, max_idx, sw, ad2, sc, ln,
                                                (hipFloatComplex *)rg, (hipFloatComplex *)a, s); }

static void call_extract(double *hh, double *tau, int nbw, int n, int val, hipStream_t s)
{ launch_extract_hh_tau_c_hip_kernel_real_double(hh, tau, nbw, n, val, s); }

static void call_extract(float *hh, float *tau, int nbw, int n, int val, hipStream_t s)
{ launch_extract_hh_tau_c_hip_kernel_real_single(hh, tau, nbw, n, val, s); }

static void call_extract(gpuDoubleComplex *hh, gpuDoubleComplex *tau, int nbw, int n, int val, hipStream_t s)
{ launch_extract_hh_tau_c_hip_kernel_complex_double(
      (hipDoubleComplex *)hh, (hipDoubleComplex *)tau, nbw, n, val, s); }

static void call_extract(gpuFloatComplex *hh, gpuFloatComplex *tau, int nbw, int n, int val, hipStream_t s)
{ launch_extract_hh_tau_c_hip_kernel_complex_single(
      (hipFloatComplex *)hh, (hipFloatComplex *)tau, nbw, n, val, s); }

// ============================================================
// Test: my_pack_c_hip_kernel
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
    HIP_CHECK(hipMalloc(&d_src, src_n * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_dst, dst_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_src, h_src, src_n * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dst, 0, dst_n * sizeof(T)));

    call_pack(rc, n_off, max_idx, sw, ad2, sc, ln, d_src, d_dst, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dst, d_dst, dst_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    int failures = 0;
    for (int i = 0; i < dst_n; i++)
        if (!vals_equal(h_dst[i], expected[i])) { failures++; }

    printf("  my_pack_c_hip_kernel   <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: my_unpack_c_hip_kernel  (inverse of pack)
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
    HIP_CHECK(hipMalloc(&d_rg, rg_n * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_a,  a_n  * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_rg, h_rg, rg_n * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_a, 0, a_n * sizeof(T)));

    call_unpack(rc, n_off, max_idx, sw, ad2, sc, ln, d_rg, d_a, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_a[8];
    HIP_CHECK(hipMemcpy(h_a, d_a, a_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_rg));
    HIP_CHECK(hipFree(d_a));

    int failures = 0;
    for (int i = 0; i < a_n; i++)
        if (!vals_equal(h_a[i], make_val<T>(i + 1))) { failures++; }

    printf("  my_unpack_c_hip_kernel <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: extract_hh_tau_c_hip_kernel
//
// n=4 reflectors, nbw=2.  hh[col*nbw] is the first element of
// each reflector column; hh[col*nbw+1] is padding (untouched).
//
//   hh_tau[col]    <- hh[col*nbw]          (copy)
//   hh[col*nbw]    <- one_val  if is_zero==0
//                  <- zero_val if is_zero!=0
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
    HIP_CHECK(hipMalloc(&d_hh,  hh_n * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tau, n    * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_hh, h_hh, hh_n * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_tau, 0, n * sizeof(T)));

    call_extract(d_hh, d_tau, nbw, n, is_zero_val, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_hh,  d_hh,  hh_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_tau, d_tau, n    * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_hh));
    HIP_CHECK(hipFree(d_tau));

    int failures = 0;
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_tau[col], make_val<T>(col + 1))) { failures++; }

    T expected_first = (is_zero_val == 0) ? one_val<T>() : zero_val<T>();
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_hh[col * nbw], expected_first)) { failures++; }

    /* Padding elements must be untouched */
    for (int col = 0; col < n; col++)
        if (!vals_equal(h_hh[col * nbw + 1], make_val<T>(9))) { failures++; }

    printf("  extract_hh_tau_c_hip_kernel<%s>(is_zero=%d): %s\n",
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

    printf("Testing my_pack_c_hip_kernel:\n");
    failures += run_pack_test<double>         ("double");
    failures += run_pack_test<float>          ("float");
    failures += run_pack_test<gpuDoubleComplex>("gpuDoubleComplex");
    failures += run_pack_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nTesting my_unpack_c_hip_kernel:\n");
    failures += run_unpack_test<double>         ("double");
    failures += run_unpack_test<float>          ("float");
    failures += run_unpack_test<gpuDoubleComplex>("gpuDoubleComplex");
    failures += run_unpack_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nTesting extract_hh_tau_c_hip_kernel:\n");
    failures += run_extract_hh_tau_test<double>         ("double");
    failures += run_extract_hh_tau_test<float>          ("float");
    failures += run_extract_hh_tau_test<gpuDoubleComplex>("gpuDoubleComplex");
    failures += run_extract_hh_tau_test<gpuFloatComplex> ("gpuFloatComplex");

    if (failures == 0)
        printf("\nAll tests passed.\n");
    else
        printf("\n%d test(s) FAILED.\n", failures);

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
    fprintf(stderr, "Error: this test requires WITH_AMD_GPU_VERSION\n");
    abort();
}

#endif /* WITH_AMD_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
