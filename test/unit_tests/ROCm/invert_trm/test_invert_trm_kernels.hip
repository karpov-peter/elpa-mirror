// Unit tests for kernels in
// src/invert_trm/GPU/ROCm/elpa_invert_trm_hip.hip (ROCm/HIP port of CUDA tests):
//
//   hip_copy_a_tmat2     – copy a stripe of a_dev into tmat2_dev
//   hip_copy_tmp2_tmat2  – copy nb columns of tmp2 into tmat2
//   hip_copy_a_tmat1     – copy a block of a_dev into tmat1_dev and zero source
//   hip_copy_tmp1_tmp2   – scatter packed lower-triangular tmp1 into tmp2
//   hip_copy_a_tmp1      – gather lower-triangular block from a_dev into packed tmp1

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

#include "../../../../src/invert_trm/GPU/ROCm/elpa_invert_trm_hip.hip"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _err = (call);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s\n",                             \
              __FILE__, __LINE__, hipGetErrorString(_err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static int g_failures = 0;

#define REPORT(label, ok)                                                     \
  do {                                                                        \
    printf("  %-64s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

// ---- type helpers ----

template <typename T> static T make_val(int n);
template <> double           make_val<double>(int n)          { return (double)n; }
template <> float            make_val<float>(int n)           { return (float)n; }
template <> gpuDoubleComplex make_val<gpuDoubleComplex>(int n)  { return make_gpuDoubleComplex((double)n, (double)(n + 100)); }
template <> gpuFloatComplex  make_val<gpuFloatComplex>(int n)   { return make_gpuFloatComplex((float)n, (float)(n + 100)); }

template <typename T>
static bool vals_eq(T a, T b)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

template <typename T>
static bool is_zero(T v)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>)
        return v.x == 0 && v.y == 0;
    else
        return v == (T)0;
}

// ---- dispatchers: hip_copy_a_tmat2 ----

static void call_copy_a_tmat2(double *a, double *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, hipStream_t s)
{ hip_copy_double_a_tmat2_FromC(a, tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

static void call_copy_a_tmat2(float *a, float *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, hipStream_t s)
{ hip_copy_float_a_tmat2_FromC(a, tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

static void call_copy_a_tmat2(gpuDoubleComplex *a, gpuDoubleComplex *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, hipStream_t s)
{ hip_copy_double_complex_a_tmat2_FromC((double _Complex *)a, (double _Complex *)tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

static void call_copy_a_tmat2(gpuFloatComplex *a, gpuFloatComplex *tmat2,
    int *nblk, int *mR, int *lc, int *lcx, int *lr1, int *nb, hipStream_t s)
{ hip_copy_float_complex_a_tmat2_FromC((float _Complex *)a, (float _Complex *)tmat2, nblk, mR, lc, lcx, lr1, nb, s); }

// ---- dispatchers: hip_copy_tmp2_tmat2 ----

static void call_copy_tmp2_tmat2(double *tmp2, double *tmat2,
    int *nblk, int *l_col1, int *nb, hipStream_t s)
{ hip_copy_double_tmp2_tmat2_FromC(tmp2, tmat2, nblk, l_col1, nb, s); }

static void call_copy_tmp2_tmat2(float *tmp2, float *tmat2,
    int *nblk, int *l_col1, int *nb, hipStream_t s)
{ hip_copy_float_tmp2_tmat2_FromC(tmp2, tmat2, nblk, l_col1, nb, s); }

static void call_copy_tmp2_tmat2(gpuDoubleComplex *tmp2, gpuDoubleComplex *tmat2,
    int *nblk, int *l_col1, int *nb, hipStream_t s)
{ hip_copy_double_complex_tmp2_tmat2_FromC((double _Complex *)tmp2, (double _Complex *)tmat2, nblk, l_col1, nb, s); }

static void call_copy_tmp2_tmat2(gpuFloatComplex *tmp2, gpuFloatComplex *tmat2,
    int *nblk, int *l_col1, int *nb, hipStream_t s)
{ hip_copy_float_complex_tmp2_tmat2_FromC((float _Complex *)tmp2, (float _Complex *)tmat2, nblk, l_col1, nb, s); }

// ---- dispatchers: hip_copy_a_tmat1 ----

static void call_copy_a_tmat1(double *a, double *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, hipStream_t s)
{ hip_copy_double_a_tmat1_FromC(a, tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

static void call_copy_a_tmat1(float *a, float *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, hipStream_t s)
{ hip_copy_float_a_tmat1_FromC(a, tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

static void call_copy_a_tmat1(gpuDoubleComplex *a, gpuDoubleComplex *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, hipStream_t s)
{ hip_copy_double_complex_a_tmat1_FromC((double _Complex *)a, (double _Complex *)tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

static void call_copy_a_tmat1(gpuFloatComplex *a, gpuFloatComplex *tmat1,
    int *l_rows, int *mR, int *nb, int *l_row1, int *l_col1, hipStream_t s)
{ hip_copy_float_complex_a_tmat1_FromC((float _Complex *)a, (float _Complex *)tmat1, l_rows, mR, nb, l_row1, l_col1, s); }

// ---- dispatchers: hip_copy_tmp1_tmp2 ----

static void call_copy_tmp1_tmp2(double *tmp1, double *tmp2,
    int *nblk, int *nb, hipStream_t s)
{ hip_copy_double_tmp1_tmp2_FromC(tmp1, tmp2, nblk, nb, s); }

static void call_copy_tmp1_tmp2(float *tmp1, float *tmp2,
    int *nblk, int *nb, hipStream_t s)
{ hip_copy_float_tmp1_tmp2_FromC(tmp1, tmp2, nblk, nb, s); }

static void call_copy_tmp1_tmp2(gpuDoubleComplex *tmp1, gpuDoubleComplex *tmp2,
    int *nblk, int *nb, hipStream_t s)
{ hip_copy_double_complex_tmp1_tmp2_FromC((double _Complex *)tmp1, (double _Complex *)tmp2, nblk, nb, s); }

static void call_copy_tmp1_tmp2(gpuFloatComplex *tmp1, gpuFloatComplex *tmp2,
    int *nblk, int *nb, hipStream_t s)
{ hip_copy_float_complex_tmp1_tmp2_FromC((float _Complex *)tmp1, (float _Complex *)tmp2, nblk, nb, s); }

// ---- dispatchers: hip_copy_a_tmp1 ----

static void call_copy_a_tmp1(double *a, double *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, hipStream_t s)
{ hip_copy_double_a_tmp1_FromC(a, tmp1, l_row1, l_col1, mR, nb, s); }

static void call_copy_a_tmp1(float *a, float *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, hipStream_t s)
{ hip_copy_float_a_tmp1_FromC(a, tmp1, l_row1, l_col1, mR, nb, s); }

static void call_copy_a_tmp1(gpuDoubleComplex *a, gpuDoubleComplex *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, hipStream_t s)
{ hip_copy_double_complex_a_tmp1_FromC((double _Complex *)a, (double _Complex *)tmp1, l_row1, l_col1, mR, nb, s); }

static void call_copy_a_tmp1(gpuFloatComplex *a, gpuFloatComplex *tmp1,
    int *l_row1, int *l_col1, int *mR, int *nb, hipStream_t s)
{ hip_copy_float_complex_a_tmp1_FromC((float _Complex *)a, (float _Complex *)tmp1, l_row1, l_col1, mR, nb, s); }

// ============================================================
// Test: hip_copy_a_tmat2
// ============================================================
template <typename T>
static void run_copy_a_tmat2_test(const char *type_name)
{
    int nblk = 2, matrixRows = 4, l_cols = 3, l_colx = 2, l_row1 = 1, nb = 2;
    const int a_n     = matrixRows * l_cols;
    const int tmat2_n = nblk * l_cols;

    T h_a[12], h_tmat2[6];
    for (int k = 0; k < a_n; k++)     h_a[k]     = make_val<T>(k + 1);
    for (int k = 0; k < tmat2_n; k++) h_tmat2[k] = make_val<T>(-999);

    T *d_a, *d_tmat2;
    HIP_CHECK(hipMalloc(&d_a,     a_n     * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tmat2, tmat2_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,     h_a,     a_n     * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tmat2, h_tmat2, tmat2_n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_a_tmat2(d_a, d_tmat2, &nblk, &matrixRows, &l_cols, &l_colx, &l_row1, &nb,
                      (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[6];
    HIP_CHECK(hipMemcpy(h_res, d_tmat2, tmat2_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_tmat2));

    const int tmat2_idx[] = {2, 3, 4, 5};
    const int a_idx[]     = {4, 5, 8, 9};

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 4; k++) {
        T exp = make_val<T>(a_idx[k] + 1);
        if (!vals_eq(h_res[tmat2_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_a_tmat2 <%s> tmat2[%d]", type_name, tmat2_idx[k]);
            REPORT(buf, false);
        }
    }
    for (int k = 0; k < 2; k++) {
        if (!vals_eq(h_res[k], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_a_tmat2 <%s> tmat2[%d] sentinel clobbered", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_a_tmat2 <%s> 4 writes + 2 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: hip_copy_tmp2_tmat2
// ============================================================
template <typename T>
static void run_copy_tmp2_tmat2_test(const char *type_name)
{
    int nblk = 2, nb = 2, l_col1 = 3;
    const int tmp2_n  = nblk * nb;
    const int tmat2_n = nblk * (l_col1 - 1 + nb);

    T h_tmp2[4], h_tmat2[8];
    for (int k = 0; k < tmp2_n; k++)  h_tmp2[k]  = make_val<T>(k + 1);
    for (int k = 0; k < tmat2_n; k++) h_tmat2[k] = make_val<T>(-999);

    T *d_tmp2, *d_tmat2;
    HIP_CHECK(hipMalloc(&d_tmp2,  tmp2_n  * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tmat2, tmat2_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_tmp2,  h_tmp2,  tmp2_n  * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tmat2, h_tmat2, tmat2_n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_tmp2_tmat2(d_tmp2, d_tmat2, &nblk, &l_col1, &nb, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[8];
    HIP_CHECK(hipMemcpy(h_res, d_tmat2, tmat2_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_tmp2));
    HIP_CHECK(hipFree(d_tmat2));

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 4; k++) {
        T exp = make_val<T>(k + 1);
        if (!vals_eq(h_res[4 + k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_tmp2_tmat2 <%s> tmat2[%d]", type_name, 4 + k);
            REPORT(buf, false);
        }
    }
    for (int k = 0; k < 4; k++) {
        if (!vals_eq(h_res[k], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_tmp2_tmat2 <%s> tmat2[%d] sentinel clobbered", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_tmp2_tmat2 <%s> 4 writes + 4 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: hip_copy_a_tmat1
// ============================================================
template <typename T>
static void run_copy_a_tmat1_test(const char *type_name)
{
    int l_rows = 3, matrixRows = 4, nb = 2, l_row1 = 3, l_col1 = 2;
    const int a_n     = matrixRows * (l_col1 - 1 + nb);
    const int tmat1_n = l_rows * nb;

    T h_a[12], h_tmat1[6];
    for (int k = 0; k < a_n; k++)     h_a[k]     = make_val<T>(k + 1);
    for (int k = 0; k < tmat1_n; k++) h_tmat1[k] = make_val<T>(-999);

    T *d_a, *d_tmat1;
    HIP_CHECK(hipMalloc(&d_a,     a_n     * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tmat1, tmat1_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,     h_a,     a_n     * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tmat1, h_tmat1, tmat1_n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_a_tmat1(d_a, d_tmat1, &l_rows, &matrixRows, &nb, &l_row1, &l_col1,
                      (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res_tmat1[6], h_res_a[12];
    HIP_CHECK(hipMemcpy(h_res_tmat1, d_tmat1, tmat1_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_res_a,     d_a,     a_n     * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_tmat1));

    char buf[128];
    bool all_ok = true;

    const int tmat1_idx[] = {0, 1, 3, 4};
    const int a_idx[]     = {4, 5, 8, 9};
    for (int k = 0; k < 4; k++) {
        T exp = make_val<T>(a_idx[k] + 1);
        if (!vals_eq(h_res_tmat1[tmat1_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_a_tmat1 <%s> tmat1[%d]", type_name, tmat1_idx[k]);
            REPORT(buf, false);
        }
    }
    const int sent_idx[] = {2, 5};
    for (int k = 0; k < 2; k++) {
        if (!vals_eq(h_res_tmat1[sent_idx[k]], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_a_tmat1 <%s> tmat1[%d] sentinel clobbered", type_name, sent_idx[k]);
            REPORT(buf, false);
        }
    }
    for (int k = 0; k < 4; k++) {
        if (!is_zero(h_res_a[a_idx[k]])) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_a_tmat1 <%s> a[%d] not zeroed", type_name, a_idx[k]);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_a_tmat1 <%s> 4 writes + 2 sentinels + 4 zeros correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: hip_copy_tmp1_tmp2
// ============================================================
template <typename T>
static void run_copy_tmp1_tmp2_test(const char *type_name)
{
    int nb = 3, nblk = 3;
    const int tmp1_n = nb * (nb + 1) / 2;
    const int tmp2_n = nblk * nb;

    T h_tmp1[6], h_tmp2[9];
    for (int k = 0; k < tmp1_n; k++) h_tmp1[k] = make_val<T>(k + 1);
    for (int k = 0; k < tmp2_n; k++) h_tmp2[k] = make_val<T>(-999);

    T *d_tmp1, *d_tmp2;
    HIP_CHECK(hipMalloc(&d_tmp1, tmp1_n * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tmp2, tmp2_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_tmp1, h_tmp1, tmp1_n * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tmp2, h_tmp2, tmp2_n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_tmp1_tmp2(d_tmp1, d_tmp2, &nblk, &nb, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[9];
    HIP_CHECK(hipMemcpy(h_res, d_tmp2, tmp2_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_tmp1));
    HIP_CHECK(hipFree(d_tmp2));

    const int written_tmp2[] = {0, 3, 4, 6, 7, 8};
    const int tmp1_src[]     = {0, 1, 2, 3, 4, 5};
    const int sentinel_idx[] = {1, 2, 5};

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 6; k++) {
        T exp = make_val<T>(tmp1_src[k] + 1);
        if (!vals_eq(h_res[written_tmp2[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_tmp1_tmp2 <%s> tmp2[%d]", type_name, written_tmp2[k]);
            REPORT(buf, false);
        }
    }
    for (int k = 0; k < 3; k++) {
        if (!vals_eq(h_res[sentinel_idx[k]], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_tmp1_tmp2 <%s> tmp2[%d] sentinel clobbered", type_name, sentinel_idx[k]);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_tmp1_tmp2 <%s> 6 writes + 3 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: hip_copy_a_tmp1
// ============================================================
template <typename T>
static void run_copy_a_tmp1_test(const char *type_name)
{
    int nb = 3, l_row1 = 1, l_col1 = 1, matrixRows = 3;
    const int a_n    = matrixRows * nb;
    const int tmp1_n = nb * (nb + 1) / 2;

    T h_a[9], h_tmp1[6];
    for (int k = 0; k < a_n;    k++) h_a[k]    = make_val<T>(k + 1);
    for (int k = 0; k < tmp1_n; k++) h_tmp1[k] = make_val<T>(-999);

    T *d_a, *d_tmp1;
    HIP_CHECK(hipMalloc(&d_a,    a_n    * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_tmp1, tmp1_n * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,    h_a,    a_n    * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tmp1, h_tmp1, tmp1_n * sizeof(T), hipMemcpyHostToDevice));

    call_copy_a_tmp1(d_a, d_tmp1, &l_row1, &l_col1, &matrixRows, &nb, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[6];
    HIP_CHECK(hipMemcpy(h_res, d_tmp1, tmp1_n * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_tmp1));

    const int a_idx[] = {0, 3, 4, 6, 7, 8};

    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < 6; k++) {
        T exp = make_val<T>(a_idx[k] + 1);
        if (!vals_eq(h_res[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_a_tmp1 <%s> tmp1[%d]", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_a_tmp1 <%s> all 6 elements correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for elpa_invert_trm_hip.hip kernels ===\n\n");

    printf("hip_copy_a_tmat2:\n");
    run_copy_a_tmat2_test<double>          ("double");
    run_copy_a_tmat2_test<float>           ("float");
    run_copy_a_tmat2_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_a_tmat2_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nhip_copy_tmp2_tmat2:\n");
    run_copy_tmp2_tmat2_test<double>          ("double");
    run_copy_tmp2_tmat2_test<float>           ("float");
    run_copy_tmp2_tmat2_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_tmp2_tmat2_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nhip_copy_a_tmat1:\n");
    run_copy_a_tmat1_test<double>          ("double");
    run_copy_a_tmat1_test<float>           ("float");
    run_copy_a_tmat1_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_a_tmat1_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nhip_copy_tmp1_tmp2:\n");
    run_copy_tmp1_tmp2_test<double>          ("double");
    run_copy_tmp1_tmp2_test<float>           ("float");
    run_copy_tmp1_tmp2_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_tmp1_tmp2_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\nhip_copy_a_tmp1:\n");
    run_copy_a_tmp1_test<double>          ("double");
    run_copy_a_tmp1_test<float>           ("float");
    run_copy_a_tmp1_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_a_tmp1_test<gpuFloatComplex> ("gpuFloatComplex");

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
