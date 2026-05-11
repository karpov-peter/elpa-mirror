// Unit tests for kernels in src/cholesky/GPU/ROCm/elpa_cholesky_hip.hip
// (ROCm/HIP port of CUDA test):
//   hip_check_device_info_kernel
//   hip_accumulate_device_info_kernel
//   hip_copy_a_tmatc_kernel<T>        (all four data types)
//   hip_set_a_lower_to_zero_kernel<T> (all four data types)

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

#include "../../../../src/cholesky/GPU/ROCm/elpa_cholesky_hip.hip"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _err = (call);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s\n",                             \
              __FILE__, __LINE__, hipGetErrorString(_err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---- host-side type helpers ----

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

template <typename T>
static T zero_val()
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>) return make_gpuDoubleComplex(0.0, 0.0);
    else if constexpr (std::is_same_v<T, gpuFloatComplex>) return make_gpuFloatComplex(0.0f, 0.0f);
    else return (T)0;
}

template <typename T>
static T conj_val(T v)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex>)
        return make_gpuDoubleComplex(v.x, -v.y);
    else if constexpr (std::is_same_v<T, gpuFloatComplex>)
        return make_gpuFloatComplex(v.x, -v.y);
    else
        return v;
}

template <typename T>
static bool vals_equal(T a, T b)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

// ---- per-type char tag for hip_set_a_lower_to_zero_FromC ----

template <typename T> static char type_char();
template <> char type_char<double>()          { return 'D'; }
template <> char type_char<float>()           { return 'S'; }
template <> char type_char<gpuDoubleComplex>() { return 'Z'; }
template <> char type_char<gpuFloatComplex>()  { return 'C'; }

// ---- dispatcher overloads for hip_copy_a_tmatc ----

static void call_copy_a_tmatc(double *a, double *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               hipStream_t s)
{ hip_copy_double_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

static void call_copy_a_tmatc(float *a, float *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               hipStream_t s)
{ hip_copy_float_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

// The HIP cholesky complex functions take hipDoubleComplex* (= gpuDoubleComplex*)
static void call_copy_a_tmatc(gpuDoubleComplex *a, gpuDoubleComplex *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               hipStream_t s)
{ hip_copy_double_complex_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

static void call_copy_a_tmatc(gpuFloatComplex *a, gpuFloatComplex *tmatc,
                               int *nblk, int *mRows, int *lcols, int *lcolx, int *lrow1,
                               hipStream_t s)
{ hip_copy_float_complex_a_tmatc_FromC(a, tmatc, nblk, mRows, lcols, lcolx, lrow1, s); }

// ============================================================
// Test: hip_check_device_info_kernel
// ============================================================
static int run_check_device_info_test()
{
    int h_info = 0;
    int *d_info;
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_info, &h_info, sizeof(int), hipMemcpyHostToDevice));

    hip_check_device_info_FromC(d_info, nullptr);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipFree(d_info));
    printf("  hip_check_device_info_kernel   (info=0): PASS\n");
    return 0;
}

// ============================================================
// Test: hip_accumulate_device_info_kernel
// ============================================================
static int run_accumulate_device_info_test()
{
    int h_abs = 5;
    int h_new;
    int *d_abs, *d_new;
    HIP_CHECK(hipMalloc(&d_abs, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_new, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_abs, &h_abs, sizeof(int), hipMemcpyHostToDevice));

    int deltas[] = { -3, 0, 7 };
    for (int i = 0; i < 3; i++) {
        h_new = deltas[i];
        HIP_CHECK(hipMemcpy(d_new, &h_new, sizeof(int), hipMemcpyHostToDevice));
        hip_accumulate_device_info_FromC(d_abs, d_new, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
    }

    HIP_CHECK(hipMemcpy(&h_abs, d_abs, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_abs));
    HIP_CHECK(hipFree(d_new));

    int failures = (h_abs != 15) ? 1 : 0;
    printf("  hip_accumulate_device_info_kernel (expected 15, got %d): %s\n",
           h_abs, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: hip_copy_a_tmatc_kernel<T>
// ============================================================
template <typename T>
static int run_copy_a_tmatc_test(const char *type_name)
{
    int nblk = 2, matrixRows = 3, l_cols = 3, l_colx = 1, l_row1 = 1;
    const int a_n     = matrixRows * l_cols;
    const int tmatc_n = l_cols * nblk;

    T h_a[9];
    for (int i = 0; i < a_n; i++) h_a[i] = make_val<T>(i + 1);

    T *d_a, *d_tmatc;
    HIP_CHECK(hipMalloc(&d_a,     a_n     * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tmatc, tmatc_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a, h_a, a_n * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_tmatc, 0, tmatc_n * sizeof(T)));

    call_copy_a_tmatc(d_a, d_tmatc, &nblk, &matrixRows, &l_cols, &l_colx, &l_row1, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_tmatc[6];
    HIP_CHECK(hipMemcpy(h_tmatc, d_tmatc, tmatc_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_tmatc));

    // tmatc[(jj-1) + (ii-1)*l_cols] = conj(a[(ii-1) + (jj-1)*matrixRows])
    T expected[6];
    for (int ii = 1; ii <= nblk; ii++)
        for (int jj = 1; jj <= l_cols; jj++)
            expected[(jj-1) + (ii-1)*l_cols] = conj_val(h_a[(ii-1) + (jj-1)*matrixRows]);

    int failures = 0;
    for (int i = 0; i < tmatc_n; i++)
        if (!vals_equal(h_tmatc[i], expected[i])) failures++;

    printf("  hip_copy_a_tmatc_kernel        <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: hip_set_a_lower_to_zero_kernel<T>
// ============================================================
template <typename T>
static int run_set_a_lower_to_zero_test(const char *type_name)
{
    int na = 3, matrixRows = 3, my_pcol = 0, np_cols = 1;
    int my_prow = 0, np_rows = 1, nblk = 1, wantDebug = 1;
    const int a_n = matrixRows * na;

    T h_a[9];
    for (int i = 0; i < a_n; i++) h_a[i] = make_val<T>(i + 1);

    T *d_a;
    HIP_CHECK(hipMalloc(&d_a, a_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a, h_a, a_n * sizeof(T), hipMemcpyHostToDevice));

    hip_set_a_lower_to_zero_FromC(type_char<T>(), (intptr_t)d_a,
                                  &na, &matrixRows, &my_pcol, &np_cols,
                                  &my_prow, &np_rows, &nblk, &wantDebug, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_result[9];
    HIP_CHECK(hipMemcpy(h_result, d_a, a_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));

    T expected[9];
    for (int i = 0; i < a_n; i++) expected[i] = h_a[i];
    expected[1] = zero_val<T>();
    expected[2] = zero_val<T>();
    expected[5] = zero_val<T>();

    int failures = 0;
    for (int i = 0; i < a_n; i++)
        if (!vals_equal(h_result[i], expected[i])) failures++;

    printf("  hip_set_a_lower_to_zero_kernel <%s>: %s\n", type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
int main(void)
{
    int failures = 0;

    printf("Testing hip_check_device_info_kernel:\n");
    failures += run_check_device_info_test();

    printf("\nTesting hip_accumulate_device_info_kernel:\n");
    failures += run_accumulate_device_info_test();

    printf("\nTesting hip_copy_a_tmatc_kernel:\n");
    failures += run_copy_a_tmatc_test<double>          ("double");
    failures += run_copy_a_tmatc_test<float>           ("float");
    failures += run_copy_a_tmatc_test<gpuDoubleComplex>("gpuDoubleComplex");
    failures += run_copy_a_tmatc_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nTesting hip_set_a_lower_to_zero_kernel:\n");
    failures += run_set_a_lower_to_zero_test<double>          ("double");
    failures += run_set_a_lower_to_zero_test<float>           ("float");
    failures += run_set_a_lower_to_zero_test<gpuDoubleComplex>("gpuDoubleComplex");
    failures += run_set_a_lower_to_zero_test<gpuFloatComplex> ("gpuFloatComplex");

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
