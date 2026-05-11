// Unit tests for kernels in src/elpa1/GPU/ROCm/elpa1_hip.hip
// (ROCm/HIP port of CUDA test):
//   hip_copy_real_part_to_q_*_complex_kernel  (double_complex, float_complex)
//   hip_zero_skewsymmetric_q_*_kernel          (double, float)
//   hip_copy_skewsymmetric_second_half_q_*     (double, float; plus and minus)
//   hip_copy_skewsymmetric_first_half_q_*      (double, float)
//   hip_get_skewsymmetric_second_half_q_*      (double, float)
//   hip_put_skewsymmetric_second_half_q_*      (double, float)

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

#include "../../../../src/elpa1/GPU/ROCm/elpa1_hip.hip"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _err = (call);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s\n",                             \
              __FILE__, __LINE__, hipGetErrorString(_err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Real type deduction from complex type
template <typename T> struct hip_real_type;
template <> struct hip_real_type<gpuDoubleComplex> { using type = double; };
template <> struct hip_real_type<gpuFloatComplex>  { using type = float;  };

// ---- dispatcher overloads for hip_zero_skewsymmetric_q ----

static void call_zero_skewsymmetric_q(double *q, int *mR, int *mC, hipStream_t s)
{ hip_zero_skewsymmetric_q_double_FromC(q, mR, mC, s); }

static void call_zero_skewsymmetric_q(float *q, int *mR, int *mC, hipStream_t s)
{ hip_zero_skewsymmetric_q_float_FromC(q, mR, mC, s); }

// ---- dispatcher overloads for hip_copy_skewsymmetric_second_half_q ----

static void call_copy_second_half_q(double *q, int *i, int *mR, int *mC, int *sign, hipStream_t s)
{ hip_copy_skewsymmetric_second_half_q_double_FromC(q, i, mR, mC, sign, s); }

static void call_copy_second_half_q(float *q, int *i, int *mR, int *mC, int *sign, hipStream_t s)
{ hip_copy_skewsymmetric_second_half_q_float_FromC(q, i, mR, mC, sign, s); }

// ---- dispatcher overloads for hip_copy_skewsymmetric_first_half_q ----

static void call_copy_first_half_q(double *q, int *i, int *mR, int *mC, int *sign, hipStream_t s)
{ hip_copy_skewsymmetric_first_half_q_double_FromC(q, i, mR, mC, sign, s); }

static void call_copy_first_half_q(float *q, int *i, int *mR, int *mC, int *sign, hipStream_t s)
{ hip_copy_skewsymmetric_first_half_q_float_FromC(q, i, mR, mC, sign, s); }

// ---- dispatcher overloads for hip_get_skewsymmetric_second_half_q ----

static void call_get_second_half_q(double *q, double *q2, int *mR, int *mC, hipStream_t s)
{ hip_get_skewsymmetric_second_half_q_double_FromC(q, q2, mR, mC, s); }

static void call_get_second_half_q(float *q, float *q2, int *mR, int *mC, hipStream_t s)
{ hip_get_skewsymmetric_second_half_q_float_FromC(q, q2, mR, mC, s); }

// ---- dispatcher overloads for hip_put_skewsymmetric_second_half_q ----

static void call_put_second_half_q(double *q, double *q2, int *mR, int *mC, hipStream_t s)
{ hip_put_skewsymmetric_second_half_q_double_FromC(q, q2, mR, mC, s); }

static void call_put_second_half_q(float *q, float *q2, int *mR, int *mC, hipStream_t s)
{ hip_put_skewsymmetric_second_half_q_float_FromC(q, q2, mR, mC, s); }

// ============================================================
// Test: hip_copy_real_part_to_q_*_complex_kernel
// ============================================================
template <typename TC>
static int run_copy_real_part_to_q_test(const char *type_name)
{
    using TR = typename hip_real_type<TC>::type;

    int matrixRows = 4, l_rows = 3, l_cols_nev = 2;
    const int n = matrixRows * l_cols_nev;

    TR  h_qreal[8];
    TC  h_q[8];
    for (int k = 0; k < n; k++) {
        h_qreal[k] = (TR)(k + 1);
        h_q[k].x   = (TR)99;
        h_q[k].y   = (TR)99;
    }

    TR *d_qreal;  TC *d_q;
    HIP_CHECK(hipMalloc(&d_qreal, n * sizeof(TR)));
    HIP_CHECK(hipMalloc(&d_q,     n * sizeof(TC)));
    HIP_CHECK(hipMemcpy(d_qreal, h_qreal, n * sizeof(TR), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q,     h_q,     n * sizeof(TC), hipMemcpyHostToDevice));

    if constexpr (std::is_same_v<TC, gpuDoubleComplex>)
        hip_copy_real_part_to_q_double_complex_FromC(
            (double _Complex*)d_q, (double*)d_qreal, &matrixRows, &l_rows, &l_cols_nev, nullptr);
    else
        hip_copy_real_part_to_q_float_complex_FromC(
            (float _Complex*)d_q, (float*)d_qreal, &matrixRows, &l_rows, &l_cols_nev, nullptr);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_q, d_q, n * sizeof(TC), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_qreal));
    HIP_CHECK(hipFree(d_q));

    int failures = 0;
    for (int col = 0; col < l_cols_nev; col++) {
        for (int row = 0; row < matrixRows; row++) {
            int idx = row + matrixRows * col;
            if (row < l_rows) {
                if (h_q[idx].x != h_qreal[idx] || h_q[idx].y != (TR)0) failures++;
            } else {
                if (h_q[idx].x != (TR)99 || h_q[idx].y != (TR)99) failures++;
            }
        }
    }

    printf("  hip_copy_real_part_to_q_complex_kernel <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: hip_zero_skewsymmetric_q_*_kernel
// ============================================================
template <typename T>
static int run_zero_skewsymmetric_q_test(const char *type_name)
{
    int matrixRows = 3, matrixCols = 2;
    const int n = matrixRows * 2 * matrixCols;

    T h_q[12], h_result[12];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q;
    HIP_CHECK(hipMalloc(&d_q, n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_q, h_q, n * sizeof(T), hipMemcpyHostToDevice));

    call_zero_skewsymmetric_q(d_q, &matrixRows, &matrixCols, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_result, d_q, n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_q));

    T expected[12];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * (col + matrixCols)] = (T)0;

    int failures = 0;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  hip_zero_skewsymmetric_q_kernel        <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: hip_copy_skewsymmetric_second_half_q_*  (plus and minus)
// ============================================================
template <typename T>
static int run_copy_second_half_q_one(const char *type_name, int sign)
{
    int matrixRows = 4, matrixCols = 3, i = 2;
    const int n = matrixRows * 2 * matrixCols;

    T h_q[24];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q;
    HIP_CHECK(hipMalloc(&d_q, n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_q, h_q, n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_second_half_q(d_q, &i, &matrixRows, &matrixCols, &sign, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_result[24];
    HIP_CHECK(hipMemcpy(h_result, d_q, n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_q));

    T expected[24];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = matrixCols; col < 2 * matrixCols; col++) {
        int idx     = (i-1) + matrixRows * col;
        int idxLow  = (i-1) + matrixRows * (col - matrixCols);
        expected[idx]    = (sign == 1) ? h_q[idxLow] : -h_q[idxLow];
        expected[idxLow] = (T)0;
    }

    int failures = 0;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  hip_copy_second_half_q_kernel          <%s> sign=%d: %s\n",
           type_name, sign, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

template <typename T>
static int run_copy_second_half_q_test(const char *type_name)
{
    int failures = 0;
    failures += run_copy_second_half_q_one<T>(type_name, 1);
    failures += run_copy_second_half_q_one<T>(type_name, 0);
    return failures;
}

// ============================================================
// Test: hip_copy_skewsymmetric_first_half_q_*
// ============================================================
template <typename T>
static int run_copy_first_half_q_test(const char *type_name)
{
    int matrixRows = 4, matrixCols = 3, i = 2;
    const int n = matrixRows * matrixCols;
    int sign = 0;

    T h_q[12];
    for (int k = 0; k < n; k++) h_q[k] = (T)(k + 1);

    T *d_q;
    HIP_CHECK(hipMalloc(&d_q, n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_q, h_q, n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_first_half_q(d_q, &i, &matrixRows, &matrixCols, &sign, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_result[12];
    HIP_CHECK(hipMemcpy(h_result, d_q, n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_q));

    T expected[12];
    for (int k = 0; k < n; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        expected[(i-1) + matrixRows * col] = -h_q[(i-1) + matrixRows * col];

    int failures = 0;
    for (int k = 0; k < n; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  hip_copy_first_half_q_kernel           <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: hip_get_skewsymmetric_second_half_q_*
// ============================================================
template <typename T>
static int run_get_second_half_q_test(const char *type_name)
{
    int matrixRows = 3, matrixCols = 2;
    const int n_q  = matrixRows * 2 * matrixCols;
    const int n_q2 = matrixRows * matrixCols;

    T h_q[12];
    for (int k = 0; k < n_q; k++) h_q[k] = (T)(k + 1);

    T *d_q, *d_q2;
    HIP_CHECK(hipMalloc(&d_q,  n_q  * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_q2, n_q2 * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_q, h_q, n_q * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_q2, 0, n_q2 * sizeof(T)));

    call_get_second_half_q(d_q, d_q2, &matrixRows, &matrixCols, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_q2[6];
    HIP_CHECK(hipMemcpy(h_q2, d_q2, n_q2 * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_q2));

    T expected[6];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * col] = h_q[row + matrixRows * (col + matrixCols)];

    int failures = 0;
    for (int k = 0; k < n_q2; k++)
        if (h_q2[k] != expected[k]) failures++;

    printf("  hip_get_second_half_q_kernel           <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
// Test: hip_put_skewsymmetric_second_half_q_*
// ============================================================
template <typename T>
static int run_put_second_half_q_test(const char *type_name)
{
    int matrixRows = 3, matrixCols = 2;
    const int n_q  = matrixRows * 2 * matrixCols;
    const int n_q2 = matrixRows * matrixCols;

    T h_q[12], h_q2[6];
    for (int k = 0; k < n_q;  k++) h_q[k]  = (T)(k + 1);
    for (int k = 0; k < n_q2; k++) h_q2[k] = (T)(k + 101);

    T *d_q, *d_q2;
    HIP_CHECK(hipMalloc(&d_q,  n_q  * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_q2, n_q2 * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_q,  h_q,  n_q  * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q2, h_q2, n_q2 * sizeof(T), hipMemcpyHostToDevice));

    call_put_second_half_q(d_q, d_q2, &matrixRows, &matrixCols, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    T h_result[12];
    HIP_CHECK(hipMemcpy(h_result, d_q, n_q * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_q2));

    T expected[12];
    for (int k = 0; k < n_q; k++) expected[k] = h_q[k];
    for (int col = 0; col < matrixCols; col++)
        for (int row = 0; row < matrixRows; row++)
            expected[row + matrixRows * (col + matrixCols)] = h_q2[row + matrixRows * col];

    int failures = 0;
    for (int k = 0; k < n_q; k++)
        if (h_result[k] != expected[k]) failures++;

    printf("  hip_put_second_half_q_kernel           <%s>: %s\n",
           type_name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ============================================================
int main(void)
{
    int failures = 0;

    printf("Testing hip_copy_real_part_to_q_complex_kernel:\n");
    failures += run_copy_real_part_to_q_test<gpuDoubleComplex>("gpuDoubleComplex");
    failures += run_copy_real_part_to_q_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nTesting hip_zero_skewsymmetric_q_kernel:\n");
    failures += run_zero_skewsymmetric_q_test<double>("double");
    failures += run_zero_skewsymmetric_q_test<float> ("float");

    printf("\nTesting hip_copy_skewsymmetric_second_half_q_kernel:\n");
    failures += run_copy_second_half_q_test<double>("double");
    failures += run_copy_second_half_q_test<float> ("float");

    printf("\nTesting hip_copy_skewsymmetric_first_half_q_kernel:\n");
    failures += run_copy_first_half_q_test<double>("double");
    failures += run_copy_first_half_q_test<float> ("float");

    printf("\nTesting hip_get_skewsymmetric_second_half_q_kernel:\n");
    failures += run_get_second_half_q_test<double>("double");
    failures += run_get_second_half_q_test<float> ("float");

    printf("\nTesting hip_put_skewsymmetric_second_half_q_kernel:\n");
    failures += run_put_second_half_q_test<double>("double");
    failures += run_put_second_half_q_test<float> ("float");

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
