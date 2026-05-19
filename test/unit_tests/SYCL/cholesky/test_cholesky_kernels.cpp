
#include <stdio.h>
#include <stdlib.h>
#include "config-f90.h"
#ifdef WITH_UNIT_TESTS
#ifdef WITH_SYCL_GPU_VERSION
#include <math.h>
#include <complex>
#include <type_traits>
#include <cassert>
#include <sycl/sycl.hpp>
#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

static int g_failures = 0;

#define REPORT(label, ok) do { \
    if (ok) { printf("PASS: %s\n", label); } \
    else   { printf("FAIL: %s\n", label); g_failures++; } \
} while(0)

/* ---------- type helpers ---------- */

template <typename T>
T make_val(int i) {
    if constexpr (std::is_same_v<T, gpuDoubleComplex>) {
        return gpuDoubleComplex((double)i, (double)(i * 2));
    } else if constexpr (std::is_same_v<T, gpuFloatComplex>) {
        return gpuFloatComplex((float)i, (float)(i * 2));
    } else if constexpr (std::is_same_v<T, float>) {
        return (float)i;
    } else {
        return (double)i;
    }
}

template <typename T>
bool vals_equal(T a, T b) {
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>) {
        return a.real() == b.real() && a.imag() == b.imag();
    } else {
        return a == b;
    }
}

/* ---------- kernel bodies (copied verbatim from source) ---------- */

template <typename T>
static void sycl_copy_a_tmatc_kernel(T *a_dev, T *tmatc_dev, const int l_cols,
                                      const int matrixRows, const int l_colx,
                                      const int l_row1, const int nblk,
                                      const sycl::nd_item<1> &it) {
    int ii_index = it.get_local_id(0) + 1;
    int jj_index = it.get_group(0) + 1;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value) {
        tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] =
            std::conj(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
    } else {
        tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] =
            a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
    }
}

template <typename T>
static void sycl_set_a_lower_to_zero_kernel(T *a_dev, int na, int matrixRows,
                                             int my_pcol, int np_cols,
                                             int my_prow, int np_rows,
                                             int nblk,
                                             const sycl::nd_item<1> &it) {
    int J_gl_0  = it.get_group(0) + 1;
    int di_loc_0 = it.get_local_id(0);
    int wgSize   = it.get_local_range(0);
    T Zero = 0;
    for (int J_gl = J_gl_0; J_gl <= na; J_gl += nblk) {
        if (my_pcol == pcol(J_gl, nblk, np_cols)) {
            int l_col1 = local_index(J_gl,   my_pcol, np_cols, nblk, 1);
            int l_row1 = local_index(J_gl+1, my_prow, np_rows, nblk, 1);
            for (int di_loc = di_loc_0; di_loc < (matrixRows - (l_row1-1)); di_loc += wgSize) {
                a_dev[((l_row1-1)+di_loc) + matrixRows*(l_col1-1)] = Zero;
            }
        }
    }
}

/* ================================================================
   Test 1: sycl_check_device_info
   ================================================================ */
static void test_check_device_info(sycl::queue &q) {
    int *d_info = sycl::malloc_device<int>(1, q);
    int h_info = 0;
    q.memcpy(d_info, &h_info, sizeof(int));
    q.wait_and_throw();

    bool ok = true;
    try {
        q.single_task([=] { assert(*d_info == 0); });
        q.wait_and_throw();
    } catch (...) {
        ok = false;
    }

    sycl::free(d_info, q);
    REPORT("sycl_check_device_info", ok);
}

/* ================================================================
   Test 2: sycl_accumulate_device_info
   ================================================================ */
static void test_accumulate_device_info(sycl::queue &q) {
    /* Start with h_abs=5, apply deltas {-3, 0, 7} → expected 15 */
    int *info_abs_dev = sycl::malloc_device<int>(1, q);
    int *info_new_dev = sycl::malloc_device<int>(1, q);

    int h_abs = 5;
    q.memcpy(info_abs_dev, &h_abs, sizeof(int));
    q.wait_and_throw();

    int deltas[] = {-3, 0, 7};
    for (int d : deltas) {
        q.memcpy(info_new_dev, &d, sizeof(int));
        q.wait_and_throw();
        q.single_task([=] { *info_abs_dev += abs(*info_new_dev); });
        q.wait_and_throw();
    }

    int h_result = 0;
    q.memcpy(&h_result, info_abs_dev, sizeof(int));
    q.wait_and_throw();

    sycl::free(info_abs_dev, q);
    sycl::free(info_new_dev, q);

    REPORT("sycl_accumulate_device_info", h_result == 15);
}

/* ================================================================
   Test 3: sycl_copy_a_tmatc_kernel<T>
   ================================================================ */
template <typename T>
static void test_copy_a_tmatc(sycl::queue &q, const char *label) {
    const int nblk       = 2;
    const int matrixRows = 3;
    const int l_cols     = 3;
    const int l_colx     = 1;
    const int l_row1     = 1;

    /* a is 3×3 = 9 elements (column-major) */
    const int na = matrixRows * l_cols;
    T h_a[9];
    for (int i = 0; i < na; i++) h_a[i] = make_val<T>(i + 1);

    /* tmatc is l_cols × (matrixRows or more) — index = (col-1) + (row-1)*l_cols */
    /* Maximum index used: (l_colx-1)+(jj_max-1) + (ii_max-1)*l_cols
       jj_max = nblk*(l_cols-l_colx+1)/nblk = l_cols-l_colx+1 = 3
       ii_max = nblk = 2
       → max index = 2 + 1*3 = 5 → need 6 elements */
    const int ntmatc = l_cols * nblk; /* 6 */
    T h_tmatc[6];
    for (int i = 0; i < ntmatc; i++) h_tmatc[i] = make_val<T>(0);

    T *d_a     = sycl::malloc_device<T>(na,     q);
    T *d_tmatc = sycl::malloc_device<T>(ntmatc, q);

    q.memcpy(d_a,     h_a,     na     * sizeof(T));
    q.memcpy(d_tmatc, h_tmatc, ntmatc * sizeof(T));
    q.wait_and_throw();

    /* global_range = nblk * (l_cols - l_colx + 1) = 2*3 = 6
       local_range  = nblk = 2
       → 3 work-groups of size 2 */
    const int global_range = nblk * (l_cols - l_colx + 1);
    const int local_range  = nblk;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_a_tmatc_kernel<T>(d_a, d_tmatc, l_cols, matrixRows,
                                        l_colx, l_row1, nblk, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_tmatc, d_tmatc, ntmatc * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_a,     q);
    sycl::free(d_tmatc, q);

    /* Verify: tmatc[(jj-1) + (ii-1)*l_cols] = conj(a[(ii-1) + (jj-1)*matrixRows])
       ii in [1..nblk], jj in [l_colx .. l_colx+nblk*(l_cols-l_colx+1)/nblk - 1]
       = jj in [1..3] */
    bool ok = true;
    for (int ii = 1; ii <= nblk; ii++) {
        for (int jj = l_colx; jj <= l_cols; jj++) {
            T src = h_a[(l_row1-1)+(ii-1) + (l_colx-1+jj-l_colx)*matrixRows];
            T expected;
            if constexpr (std::is_same_v<T, gpuDoubleComplex> ||
                          std::is_same_v<T, gpuFloatComplex>) {
                expected = std::conj(src);
            } else {
                expected = src;
            }
            T got = h_tmatc[(l_colx-1)+(jj-1) + (ii-1)*l_cols];
            if (!vals_equal(got, expected)) { ok = false; break; }
        }
        if (!ok) break;
    }
    REPORT(label, ok);
}

/* ================================================================
   Test 4: sycl_set_a_lower_to_zero_kernel<T>
   ================================================================ */
template <typename T>
static void test_set_a_lower_to_zero(sycl::queue &q, const char *label) {
    const int na         = 3;
    const int matrixRows = 3;
    const int my_pcol    = 0;
    const int np_cols    = 1;
    const int my_prow    = 0;
    const int np_rows    = 1;
    const int nblk       = 1;

    const int na_total = matrixRows * na; /* 9 */
    T h_a[9];
    for (int i = 0; i < na_total; i++) h_a[i] = make_val<T>(i + 1);

    T *d_a = sycl::malloc_device<T>(na_total, q);
    q.memcpy(d_a, h_a, na_total * sizeof(T));
    q.wait_and_throw();

    const int global_range = 32 * nblk; /* 32 */
    const int local_range  = 32;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_set_a_lower_to_zero_kernel<T>(d_a, na, matrixRows,
                                               my_pcol, np_cols,
                                               my_prow, np_rows,
                                               nblk, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_a, d_a, na_total * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_a, q);

    /* Expected zeros at 0-based indices [1], [2], [5] (column-major) */
    T zero = make_val<T>(0);
    bool ok = vals_equal(h_a[1], zero) &&
              vals_equal(h_a[2], zero) &&
              vals_equal(h_a[5], zero);
    REPORT(label, ok);
}

/* ================================================================
   main
   ================================================================ */
int main(void) {
    sycl::queue q(sycl::default_selector_v);

    test_check_device_info(q);
    test_accumulate_device_info(q);

    test_copy_a_tmatc<double>         (q, "sycl_copy_a_tmatc_kernel<double>");
    test_copy_a_tmatc<float>          (q, "sycl_copy_a_tmatc_kernel<float>");
    test_copy_a_tmatc<gpuDoubleComplex>(q, "sycl_copy_a_tmatc_kernel<gpuDoubleComplex>");
    test_copy_a_tmatc<gpuFloatComplex> (q, "sycl_copy_a_tmatc_kernel<gpuFloatComplex>");

    test_set_a_lower_to_zero<double>         (q, "sycl_set_a_lower_to_zero_kernel<double>");
    test_set_a_lower_to_zero<float>          (q, "sycl_set_a_lower_to_zero_kernel<float>");
    test_set_a_lower_to_zero<gpuDoubleComplex>(q, "sycl_set_a_lower_to_zero_kernel<gpuDoubleComplex>");
    test_set_a_lower_to_zero<gpuFloatComplex> (q, "sycl_set_a_lower_to_zero_kernel<gpuFloatComplex>");

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif /* WITH_SYCL_GPU_VERSION */
#else
int main(void) {
    fprintf(stderr, "Error: unit tests not enabled (recompile with WITH_UNIT_TESTS)\n");
    abort();
}
#endif /* WITH_UNIT_TESTS */
