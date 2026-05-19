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
static void sycl_copy_a_tmat2_kernel(T *a_dev, T *tmat2_dev, const int nblk,
                                      const int matrixRows, const int l_colx,
                                      const int l_row1, sycl::nd_item<1> it) {
    int nb_index    = it.get_local_id(0) + 1;
    int l_col_index = it.get_group(0) + 1;
    tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] =
        a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1) * matrixRows];
}

template <typename T>
static void sycl_copy_tmp2_tmat2_kernel(T *tmp2_dev, T *tmat2_dev, const int nblk,
                                         const int l_col1, sycl::nd_item<1> it) {
    int nb_index    = it.get_local_id(0) + 1;
    int l_col_index = it.get_group(0) + 1;
    tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] =
        tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1) * nblk];
}

template <typename T>
static void sycl_copy_a_tmat1_kernel(T *a_dev, T *tmat1_dev, const int l_rows,
                                      const int matrixRows, const int l_col1,
                                      const int nb, const int l_row1,
                                      sycl::nd_item<1> it) {
    int nb_index     = it.get_local_id(0) + 1;
    int l_row1_index = it.get_group(0) + 1;
    tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] =
        a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1) * matrixRows];
    a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;
}

template <typename T>
static void sycl_copy_tmp1_tmp2_kernel(T *tmp1_dev, T *tmp2_dev, const int nblk,
                                        const int nb, sycl::nd_item<1> it) {
    int i_index = it.get_local_id(0) + 1;
    int j_index = it.get_group(0) + 1;
    if (j_index < i_index+1) {
        tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] =
            tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
    }
}

template <typename T>
static void sycl_copy_a_tmp1_kernel(T *a_dev, T *tmp1_dev, const int l_row1,
                                     const int l_col1, const int matrixRows,
                                     const int nb, sycl::nd_item<1> it) {
    int i_index = it.get_local_id(0) + 1;
    int j_index = it.get_group(0) + 1;
    if (j_index < i_index+1) {
        tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] =
            a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
    }
}

/* ================================================================
   Test 1: sycl_copy_a_tmat2_kernel<T>
   ================================================================ */
template <typename T>
static void test_copy_a_tmat2(sycl::queue &q, const char *label) {
    const int nblk       = 2;
    const int matrixRows = 3;
    const int l_cols     = 3;
    const int l_colx     = 1;
    const int l_row1     = 1;

    const int na = matrixRows * l_cols; /* 9 */
    T h_a[9];
    for (int i = 0; i < na; i++) h_a[i] = make_val<T>(i + 1);

    /* tmat2 is nblk × l_cols = 6 elements */
    const int ntmat2 = nblk * l_cols;
    T h_tmat2[6];
    for (int i = 0; i < ntmat2; i++) h_tmat2[i] = make_val<T>(0);

    T *d_a     = sycl::malloc_device<T>(na,     q);
    T *d_tmat2 = sycl::malloc_device<T>(ntmat2, q);

    q.memcpy(d_a,     h_a,     na     * sizeof(T));
    q.memcpy(d_tmat2, h_tmat2, ntmat2 * sizeof(T));
    q.wait_and_throw();

    /* global_range = nblk*(l_cols-l_colx+1) = 2*3 = 6
       local_range  = nblk = 2  → 3 work-groups */
    const int global_range = nblk * (l_cols - l_colx + 1);
    const int local_range  = nblk;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_a_tmat2_kernel<T>(d_a, d_tmat2, nblk, matrixRows,
                                        l_colx, l_row1, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_tmat2, d_tmat2, ntmat2 * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_a,     q);
    sycl::free(d_tmat2, q);

    /* Verify: tmat2[nb-1 + (jj-1)*nblk] = a[(l_row1-1)+(nb-1) + (jj-1)*matrixRows]
       nb in [1..nblk], jj in [l_colx..l_cols] */
    bool ok = true;
    for (int jj = l_colx; jj <= l_cols; jj++) {
        for (int nb = 1; nb <= nblk; nb++) {
            T expected = h_a[(l_row1-1)+(nb-1) + (l_colx-1+jj-l_colx)*matrixRows];
            T got      = h_tmat2[(nb-1) + (jj-1)*nblk];
            if (!vals_equal(got, expected)) { ok = false; break; }
        }
        if (!ok) break;
    }
    REPORT(label, ok);
}

/* ================================================================
   Test 2: sycl_copy_tmp2_tmat2_kernel<T>
   ================================================================ */
template <typename T>
static void test_copy_tmp2_tmat2(sycl::queue &q, const char *label) {
    const int nblk   = 2;
    const int nb     = 2;
    const int l_col1 = 1;

    /* tmp2 is nb×nb = 4 elements, fill 1..4 */
    T h_tmp2[4];
    for (int i = 0; i < nb * nb; i++) h_tmp2[i] = make_val<T>(i + 1);

    /* tmat2 is nblk×nb = 4 elements (zeros) */
    T h_tmat2[4];
    for (int i = 0; i < nblk * nb; i++) h_tmat2[i] = make_val<T>(0);

    T *d_tmp2  = sycl::malloc_device<T>(nb * nb,   q);
    T *d_tmat2 = sycl::malloc_device<T>(nblk * nb, q);

    q.memcpy(d_tmp2,  h_tmp2,  nb * nb   * sizeof(T));
    q.memcpy(d_tmat2, h_tmat2, nblk * nb * sizeof(T));
    q.wait_and_throw();

    /* global_range = nb*nb = 4, local_range = nb = 2 → 2 work-groups */
    const int global_range = nb * nb;
    const int local_range  = nb;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_tmp2_tmat2_kernel<T>(d_tmp2, d_tmat2, nblk, l_col1, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_tmat2, d_tmat2, nblk * nb * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_tmp2,  q);
    sycl::free(d_tmat2, q);

    /* Verify: tmat2[nb-1 + (l_col1-1+l_col-1)*nblk] = tmp2[nb-1 + (l_col-1)*nblk]
       This is an identity copy from tmp2 into tmat2 starting at l_col1 */
    bool ok = true;
    for (int l_col = 1; l_col <= nb; l_col++) {
        for (int nb_i = 1; nb_i <= nb; nb_i++) {
            T expected = h_tmp2[(nb_i-1) + (l_col-1)*nblk];
            T got      = h_tmat2[(nb_i-1) + (l_col1-1+l_col-1)*nblk];
            if (!vals_equal(got, expected)) { ok = false; break; }
        }
        if (!ok) break;
    }
    REPORT(label, ok);
}

/* ================================================================
   Test 3: sycl_copy_a_tmat1_kernel<T>
   ================================================================ */
template <typename T>
static void test_copy_a_tmat1(sycl::queue &q, const char *label) {
    const int l_rows     = 2;
    const int matrixRows = 3;
    const int l_row1     = 2; /* l_row1=2 so l_row1-1=1 → global_range = nb*1 = 2 */
    const int l_col1     = 1;
    const int nb         = 2;

    /* a is 3×3 = 9 elements, fill 1..9 */
    T h_a[9];
    for (int i = 0; i < matrixRows * nb + matrixRows; i++) h_a[i] = make_val<T>(i + 1);
    /* Pad remaining if needed — allocate full 3×3 */
    const int na = matrixRows * 3;
    T h_a_full[9];
    for (int i = 0; i < na; i++) h_a_full[i] = make_val<T>(i + 1);

    /* tmat1 is l_rows × nb = 2×2 = 4 elements (zeros) */
    const int ntmat1 = l_rows * nb;
    T h_tmat1[4];
    for (int i = 0; i < ntmat1; i++) h_tmat1[i] = make_val<T>(0);

    T *d_a     = sycl::malloc_device<T>(na,     q);
    T *d_tmat1 = sycl::malloc_device<T>(ntmat1, q);

    q.memcpy(d_a,     h_a_full, na     * sizeof(T));
    q.memcpy(d_tmat1, h_tmat1,  ntmat1 * sizeof(T));
    q.wait_and_throw();

    /* global_range = nb * (l_row1-1) = 2*1 = 2, local_range = nb = 2 → 1 work-group */
    const int global_range = nb * (l_row1 - 1);
    const int local_range  = nb;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_a_tmat1_kernel<T>(d_a, d_tmat1, l_rows, matrixRows,
                                        l_col1, nb, l_row1, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_tmat1,  d_tmat1, ntmat1 * sizeof(T));
    q.memcpy(h_a_full, d_a,     na     * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_a,     q);
    sycl::free(d_tmat1, q);

    /* Verify tmat1 and zeroed a entries
       With l_row1=2, l_row1_index runs over 1 block only (1 work-group of size nb=2)
       nb_index=1: tmat1[0 + 0*l_rows] = a[0 + (l_col1-1+0)*matrixRows] = a[0]
                   a[0 + 0*matrixRows] = 0
       nb_index=2: tmat1[0 + 1*l_rows] = a[0 + (l_col1-1+1)*matrixRows] = a[3]
                   a[0 + 1*matrixRows] = 0
    */
    T zero = make_val<T>(0);
    bool ok = vals_equal(h_tmat1[0], make_val<T>(1)) &&   /* a[0] = 1 */
              vals_equal(h_tmat1[2], make_val<T>(4)) &&   /* a[3] = 4 */
              vals_equal(h_a_full[0], zero) &&             /* a[0] zeroed */
              vals_equal(h_a_full[3], zero);               /* a[3] zeroed */
    REPORT(label, ok);
}

/* ================================================================
   Test 4: sycl_copy_tmp1_tmp2_kernel<T>
   ================================================================ */
template <typename T>
static void test_copy_tmp1_tmp2(sycl::queue &q, const char *label) {
    const int nblk = 2;
    const int nb   = 2;

    /* tmp1 is lower-triangular packed: nb*(nb+1)/2 = 3 elements */
    const int ntmp1 = nb * (nb + 1) / 2;
    T h_tmp1[3];
    for (int i = 0; i < ntmp1; i++) h_tmp1[i] = make_val<T>(i + 1);

    /* tmp2 is nblk × nb = 4 elements (zeros) */
    const int ntmp2 = nblk * nb;
    T h_tmp2[4];
    for (int i = 0; i < ntmp2; i++) h_tmp2[i] = make_val<T>(0);

    T *d_tmp1 = sycl::malloc_device<T>(ntmp1, q);
    T *d_tmp2 = sycl::malloc_device<T>(ntmp2, q);

    q.memcpy(d_tmp1, h_tmp1, ntmp1 * sizeof(T));
    q.memcpy(d_tmp2, h_tmp2, ntmp2 * sizeof(T));
    q.wait_and_throw();

    /* global_range = nb*nb = 4, local_range = nb = 2 → 2 work-groups */
    const int global_range = nb * nb;
    const int local_range  = nb;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_tmp1_tmp2_kernel<T>(d_tmp1, d_tmp2, nblk, nb, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_tmp2, d_tmp2, ntmp2 * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_tmp1, q);
    sycl::free(d_tmp2, q);

    /* Expected from derivation:
       i=1,j=1: tmp2[0+0*(nblk)] = tmp1[0] = v1  → tmp2[0] = v1
       i=2,j=1: tmp2[0+1*(nblk)] = tmp1[1] = v2  → tmp2[2] = v2
       i=2,j=2: tmp2[1+1*(nblk)] = tmp1[2] = v3  → tmp2[3] = v3
       tmp2[1] stays 0 (j=2,i=1: j < i+1 → 2 < 2 is false → skip)
    */
    T zero = make_val<T>(0);
    bool ok = vals_equal(h_tmp2[0], make_val<T>(1)) &&  /* v1 */
              vals_equal(h_tmp2[1], zero)            &&  /* untouched */
              vals_equal(h_tmp2[2], make_val<T>(2)) &&  /* v2 */
              vals_equal(h_tmp2[3], make_val<T>(3));    /* v3 */
    REPORT(label, ok);
}

/* ================================================================
   Test 5: sycl_copy_a_tmp1_kernel<T>
   ================================================================ */
template <typename T>
static void test_copy_a_tmp1(sycl::queue &q, const char *label) {
    const int l_row1     = 1;
    const int l_col1     = 1;
    const int matrixRows = 3;
    const int nb         = 2;

    /* a is 3×3 = 9 elements, fill 1..9 */
    const int na = matrixRows * 3;
    T h_a[9];
    for (int i = 0; i < na; i++) h_a[i] = make_val<T>(i + 1);

    /* tmp1 is nb*(nb+1)/2 = 3 elements (zeros) */
    const int ntmp1 = nb * (nb + 1) / 2;
    T h_tmp1[3];
    for (int i = 0; i < ntmp1; i++) h_tmp1[i] = make_val<T>(0);

    T *d_a    = sycl::malloc_device<T>(na,    q);
    T *d_tmp1 = sycl::malloc_device<T>(ntmp1, q);

    q.memcpy(d_a,    h_a,    na    * sizeof(T));
    q.memcpy(d_tmp1, h_tmp1, ntmp1 * sizeof(T));
    q.wait_and_throw();

    /* global_range = nb*nb = 4, local_range = nb = 2 → 2 work-groups */
    const int global_range = nb * nb;
    const int local_range  = nb;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
            sycl_copy_a_tmp1_kernel<T>(d_a, d_tmp1, l_row1, l_col1,
                                       matrixRows, nb, it);
        });
    });
    q.wait_and_throw();

    q.memcpy(h_tmp1, d_tmp1, ntmp1 * sizeof(T));
    q.wait_and_throw();

    sycl::free(d_a,    q);
    sycl::free(d_tmp1, q);

    /* Expected:
       i=1,j=1: tmp1[0] = a[(l_row1-1)+0 + (l_col1-1+0)*matrixRows] = a[0] = 1
       i=2,j=1: tmp1[1] = a[(l_row1-1)+0 + (l_col1-1+1)*matrixRows] = a[3] = 4
       i=2,j=2: tmp1[2] = a[(l_row1-1)+1 + (l_col1-1+1)*matrixRows] = a[4] = 5
    */
    bool ok = vals_equal(h_tmp1[0], make_val<T>(1)) &&
              vals_equal(h_tmp1[1], make_val<T>(4)) &&
              vals_equal(h_tmp1[2], make_val<T>(5));
    REPORT(label, ok);
}

/* ================================================================
   main
   ================================================================ */
int main(void) {
    sycl::queue q(sycl::default_selector_v);

    test_copy_a_tmat2<double>          (q, "sycl_copy_a_tmat2_kernel<double>");
    test_copy_a_tmat2<float>           (q, "sycl_copy_a_tmat2_kernel<float>");
    test_copy_a_tmat2<gpuDoubleComplex>(q, "sycl_copy_a_tmat2_kernel<gpuDoubleComplex>");
    test_copy_a_tmat2<gpuFloatComplex> (q, "sycl_copy_a_tmat2_kernel<gpuFloatComplex>");

    test_copy_tmp2_tmat2<double>          (q, "sycl_copy_tmp2_tmat2_kernel<double>");
    test_copy_tmp2_tmat2<float>           (q, "sycl_copy_tmp2_tmat2_kernel<float>");
    test_copy_tmp2_tmat2<gpuDoubleComplex>(q, "sycl_copy_tmp2_tmat2_kernel<gpuDoubleComplex>");
    test_copy_tmp2_tmat2<gpuFloatComplex> (q, "sycl_copy_tmp2_tmat2_kernel<gpuFloatComplex>");

    test_copy_a_tmat1<double>          (q, "sycl_copy_a_tmat1_kernel<double>");
    test_copy_a_tmat1<float>           (q, "sycl_copy_a_tmat1_kernel<float>");
    test_copy_a_tmat1<gpuDoubleComplex>(q, "sycl_copy_a_tmat1_kernel<gpuDoubleComplex>");
    test_copy_a_tmat1<gpuFloatComplex> (q, "sycl_copy_a_tmat1_kernel<gpuFloatComplex>");

    test_copy_tmp1_tmp2<double>          (q, "sycl_copy_tmp1_tmp2_kernel<double>");
    test_copy_tmp1_tmp2<float>           (q, "sycl_copy_tmp1_tmp2_kernel<float>");
    test_copy_tmp1_tmp2<gpuDoubleComplex>(q, "sycl_copy_tmp1_tmp2_kernel<gpuDoubleComplex>");
    test_copy_tmp1_tmp2<gpuFloatComplex> (q, "sycl_copy_tmp1_tmp2_kernel<gpuFloatComplex>");

    test_copy_a_tmp1<double>          (q, "sycl_copy_a_tmp1_kernel<double>");
    test_copy_a_tmp1<float>           (q, "sycl_copy_a_tmp1_kernel<float>");
    test_copy_a_tmp1<gpuDoubleComplex>(q, "sycl_copy_a_tmp1_kernel<gpuDoubleComplex>");
    test_copy_a_tmp1<gpuFloatComplex> (q, "sycl_copy_a_tmp1_kernel<gpuFloatComplex>");

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif /* WITH_SYCL_GPU_VERSION */
#else
int main(void) {
    fprintf(stderr, "Error: unit tests not enabled (recompile with WITH_UNIT_TESTS)\n");
    abort();
}
#endif /* WITH_UNIT_TESTS */
