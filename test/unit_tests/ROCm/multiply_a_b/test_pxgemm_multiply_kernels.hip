// Unit tests for kernels in
// src/multiply_a_b/GPU/elpa_pxgemm_multiply_gpu.h (ROCm/HIP port):
//
//   gpu_copy_aux_full
//   gpu_copy_and_set_zeros_aux_full
//   gpu_copy_and_set_zeros_aux_a_full
//   gpu_copy_and_set_zeros_aux_b_full
//   gpu_ccl_copy_buf_send
//   gpu_ccl_copy_buf_recv
//   gpu_copy_and_set_zeros_aux_ab_full_tn
//   gpu_copy_and_set_zeros_aux_ab_full_nt
//   gpu_update_c_tn_nt (TN and NT, beta=0)

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

#include "../../../../src/multiply_a_b/GPU/ROCm/elpa_pxgemm_multiply_hip.hip"

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
    printf("  %-68s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

// ---- type helpers ----

template <typename T> static T make_val(int n);
template <> double           make_val<double>(int n)          { return (double)n; }
template <> float            make_val<float>(int n)           { return (float)n; }
template <> gpuDoubleComplex make_val<gpuDoubleComplex>(int n) { return make_gpuDoubleComplex((double)n, (double)(n+100)); }
template <> gpuFloatComplex  make_val<gpuFloatComplex>(int n)  { return make_gpuFloatComplex((float)n, (float)(n+100)); }

template <typename T>
static bool vals_eq(T a, T b)
{
    if constexpr (std::is_same_v<T, gpuDoubleComplex> || std::is_same_v<T, gpuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
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

template <typename T> static T make_zero();
template <> double           make_zero<double>()          { return 0.0; }
template <> float            make_zero<float>()           { return 0.0f; }
template <> gpuDoubleComplex make_zero<gpuDoubleComplex>() { return make_gpuDoubleComplex(0.0, 0.0); }
template <> gpuFloatComplex  make_zero<gpuFloatComplex>()  { return make_gpuFloatComplex(0.0f, 0.0f); }

template <typename T> static char dtype();
template <> char dtype<double>()          { return 'D'; }
template <> char dtype<float>()           { return 'S'; }
template <> char dtype<gpuDoubleComplex>() { return 'Z'; }
template <> char dtype<gpuFloatComplex>()  { return 'C'; }

// ============================================================
// Test: gpu_copy_aux_full
// ============================================================
template <typename T>
static void run_copy_aux_full_test(const char *type_name)
{
    int l_rows=3, l_cols=2, lld_lhs=4, lld_rhs=3;
    const int rhs_n = lld_rhs * l_cols;
    const int lhs_n = lld_lhs * l_cols;

    T h_rhs[6], h_lhs[8];
    for (int k=0; k<rhs_n; k++) h_rhs[k] = make_val<T>(k+1);
    for (int k=0; k<lhs_n; k++) h_lhs[k] = make_val<T>(-999);

    T *d_lhs, *d_rhs;
    HIP_CHECK(hipMalloc(&d_lhs, lhs_n*sizeof(T)));
    HIP_CHECK(hipMalloc(&d_rhs, rhs_n*sizeof(T)));
    HIP_CHECK(hipMemcpy(d_lhs, h_lhs, lhs_n*sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs, h_rhs, rhs_n*sizeof(T), hipMemcpyHostToDevice));

    hip_copy_aux_full_FromC(dtype<T>(), (intptr_t)d_lhs, (intptr_t)d_rhs,
                            l_rows, l_cols, lld_lhs, lld_rhs, 0,
                            (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[8];
    HIP_CHECK(hipMemcpy(h_res, d_lhs, lhs_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_lhs)); HIP_CHECK(hipFree(d_rhs));

    const int lhs_idx[] = {0,1,2, 4,5,6};
    const int rhs_idx[] = {0,1,2, 3,4,5};

    char buf[128];
    bool all_ok = true;
    for (int k=0; k<6; k++) {
        T exp = make_val<T>(rhs_idx[k]+1);
        if (!vals_eq(h_res[lhs_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_aux_full <%s> lhs[%d]", type_name, lhs_idx[k]);
            REPORT(buf, false);
        }
    }
    const int sent[] = {3, 7};
    for (int k=0; k<2; k++) {
        if (!vals_eq(h_res[sent[k]], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_aux_full <%s> lhs[%d] sentinel clobbered", type_name, sent[k]);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_aux_full <%s> 6 writes + 2 sentinels correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: gpu_copy_and_set_zeros_aux_full
// ============================================================
template <typename T>
static void run_copy_and_set_zeros_aux_full_test(const char *type_name)
{
    int l_rows=2, l_cols=2, nblk_mult=3;
    const int a_n   = l_rows * l_cols;
    const int aux_n = nblk_mult * nblk_mult;

    T h_a[4], h_aux[9];
    for (int k=0; k<a_n;   k++) h_a[k]   = make_val<T>(k+1);
    for (int k=0; k<aux_n; k++) h_aux[k] = make_val<T>(-999);

    T *d_a, *d_aux;
    HIP_CHECK(hipMalloc(&d_a,   a_n  *sizeof(T)));
    HIP_CHECK(hipMalloc(&d_aux, aux_n*sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,   h_a,   a_n  *sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_aux, h_aux, aux_n*sizeof(T), hipMemcpyHostToDevice));

    hip_copy_and_set_zeros_aux_full_FromC(dtype<T>(), (intptr_t)d_a, (intptr_t)d_aux,
                                          l_rows, l_cols, nblk_mult, 0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[9];
    HIP_CHECK(hipMemcpy(h_res, d_aux, aux_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_aux));

    const int written_idx[] = {0,1, 3,4};
    const int a_src[]       = {0,1, 2,3};
    const int zero_idx[]    = {2, 5, 6, 7, 8};

    char buf[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(a_src[k]+1);
        if (!vals_eq(h_res[written_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_zeros_aux_full <%s> aux[%d]", type_name, written_idx[k]);
            REPORT(buf, false);
        }
    }
    for (int k=0; k<5; k++) {
        if (!vals_eq(h_res[zero_idx[k]], make_zero<T>())) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_zeros_aux_full <%s> aux[%d] not zero", type_name, zero_idx[k]);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_zeros_aux_full <%s> 4 writes + 5 zeros correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: gpu_copy_and_set_zeros_aux_a_full
// ============================================================
template <typename T>
static void run_copy_aux_a_full_test(const char *type_name)
{
    int l_rows=2, nblk_mult_cols=2, nblk=2, np_bc_fine=2, np_cols_fine=2, np_cols=2;
    const int a_n   = 8;
    const int aux_n = l_rows * nblk_mult_cols;

    T h_a[8], h_aux[4];
    for (int k=0; k<a_n;  k++) h_a[k]   = make_val<T>(k+1);
    for (int k=0; k<aux_n; k++) h_aux[k] = make_val<T>(-999);

    T *d_a, *d_aux;
    HIP_CHECK(hipMalloc(&d_a,   a_n  *sizeof(T)));
    HIP_CHECK(hipMalloc(&d_aux, aux_n*sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,   h_a,   a_n  *sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_aux, h_aux, aux_n*sizeof(T), hipMemcpyHostToDevice));

    hip_copy_and_set_zeros_aux_a_full_FromC(dtype<T>(), (intptr_t)d_a, (intptr_t)d_aux,
                                            l_rows, nblk_mult_cols, nblk,
                                            np_bc_fine, np_cols_fine, np_cols,
                                            0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[4];
    HIP_CHECK(hipMemcpy(h_res, d_aux, aux_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_aux));

    char buf[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(k+5);
        if (!vals_eq(h_res[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_aux_a_full <%s> aux_a[%d]", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_aux_a_full <%s> all 4 elements correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: gpu_copy_and_set_zeros_aux_b_full
// ============================================================
template <typename T>
static void run_copy_aux_b_full_test(const char *type_name)
{
    int l_rows=4, l_cols=2, nblk_mult=2, nblk_mult_rows=2, nblk=2;
    int np_fine=2, np_rows_fine=2, np_rows=2, SM_count=1;
    const int b_n   = l_rows * l_cols;
    const int aux_n = nblk_mult * l_cols;

    T h_b[8], h_aux[4];
    for (int k=0; k<b_n;   k++) h_b[k]   = make_val<T>(k+1);
    for (int k=0; k<aux_n; k++) h_aux[k] = make_val<T>(-999);

    T *d_b, *d_aux;
    HIP_CHECK(hipMalloc(&d_b,   b_n  *sizeof(T)));
    HIP_CHECK(hipMalloc(&d_aux, aux_n*sizeof(T)));
    HIP_CHECK(hipMemcpy(d_b,   h_b,   b_n  *sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_aux, h_aux, aux_n*sizeof(T), hipMemcpyHostToDevice));

    hip_copy_and_set_zeros_aux_b_full_FromC(dtype<T>(), (intptr_t)d_b, (intptr_t)d_aux,
                                            l_rows, l_cols, nblk_mult, nblk_mult_rows, nblk,
                                            np_fine, np_rows_fine, np_rows,
                                            SM_count, 0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[4];
    HIP_CHECK(hipMemcpy(h_res, d_aux, aux_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_b)); HIP_CHECK(hipFree(d_aux));

    const int exp_vals[] = {3,4,7,8};
    char buf[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(exp_vals[k]);
        if (!vals_eq(h_res[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_aux_b_full <%s> aux_b[%d]", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_aux_b_full <%s> all 4 elements correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: gpu_ccl_copy_buf_send
// ============================================================
template <typename T>
static void run_ccl_copy_buf_send_test(const char *type_name)
{
    int l_rows=4, l_cols=2, lld_buf=2, nblk=2;
    int np_fine=2, np_bc_fine=0, np_rows_fine=2, np_cols_fine=1, np_rows=2, np_cols=1;
    int i_block_loc_fine_max=0, j_block_loc_fine_max=0, SM_count=1;
    const int a_n   = l_rows * l_cols;
    const int buf_n = lld_buf * nblk;

    T h_a[8], h_buf[4];
    for (int k=0; k<a_n;  k++) h_a[k]   = make_val<T>(k+1);
    for (int k=0; k<buf_n; k++) h_buf[k] = make_val<T>(-999);

    T *d_a, *d_buf;
    HIP_CHECK(hipMalloc(&d_a,   a_n  *sizeof(T)));
    HIP_CHECK(hipMalloc(&d_buf, buf_n*sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,   h_a,   a_n  *sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buf, h_buf, buf_n*sizeof(T), hipMemcpyHostToDevice));

    hip_ccl_copy_buf_send_FromC(dtype<T>(), (intptr_t)d_a, (intptr_t)d_buf,
                                l_rows, l_cols, lld_buf, nblk,
                                i_block_loc_fine_max, j_block_loc_fine_max,
                                np_fine, np_bc_fine, np_rows_fine, np_cols_fine,
                                np_rows, np_cols, SM_count, 0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[4];
    HIP_CHECK(hipMemcpy(h_res, d_buf, buf_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_buf));

    const int exp_vals[] = {3,4,7,8};
    char buf[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(exp_vals[k]);
        if (!vals_eq(h_res[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "ccl_copy_buf_send <%s> buf_send[%d]", type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "ccl_copy_buf_send <%s> all 4 elements correct", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: gpu_ccl_copy_buf_recv
// ============================================================
template <typename T>
static void run_ccl_copy_buf_recv_test(const char *type_name)
{
    int l_rows=4, l_cols=2, lld_buf=2, nblk=2;
    int np_fine=2, np_bc_fine=0, np_rows_fine=2, np_cols_fine=1, np_rows=2, np_cols=1;
    int i_block_loc_fine_max=0, j_block_loc_fine_max=0, SM_count=1;
    const int buf_n = lld_buf * nblk;
    const int at_n  = l_rows * l_cols;

    T h_buf[4], h_at[8];
    for (int k=0; k<buf_n; k++) h_buf[k] = make_val<T>(k+1);
    for (int k=0; k<at_n;  k++) h_at[k]  = make_val<T>(-999);

    T *d_buf, *d_at;
    HIP_CHECK(hipMalloc(&d_buf, buf_n*sizeof(T)));
    HIP_CHECK(hipMalloc(&d_at,  at_n *sizeof(T)));
    HIP_CHECK(hipMemcpy(d_buf, h_buf, buf_n*sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_at,  h_at,  at_n *sizeof(T), hipMemcpyHostToDevice));

    hip_ccl_copy_buf_recv_FromC(dtype<T>(), (intptr_t)d_at, (intptr_t)d_buf,
                                l_rows, l_cols, lld_buf, nblk,
                                i_block_loc_fine_max, j_block_loc_fine_max,
                                np_fine, np_bc_fine, np_rows_fine, np_cols_fine,
                                np_rows, np_cols, SM_count, 0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[8];
    HIP_CHECK(hipMemcpy(h_res, d_at, at_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_buf)); HIP_CHECK(hipFree(d_at));

    const int at_idx[]  = {2, 3, 6, 7};
    const int buf_src[] = {0, 2, 1, 3};

    char buf2[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = conj_val(make_val<T>(buf_src[k]+1));
        if (!vals_eq(h_res[at_idx[k]], exp)) {
            all_ok = false;
            snprintf(buf2, sizeof(buf2), "ccl_copy_buf_recv <%s> at_col[%d]", type_name, at_idx[k]);
            REPORT(buf2, false);
        }
    }
    const int sent[] = {0,1,4,5};
    for (int k=0; k<4; k++) {
        if (!vals_eq(h_res[sent[k]], make_val<T>(-999))) {
            all_ok = false;
            snprintf(buf2, sizeof(buf2), "ccl_copy_buf_recv <%s> at_col[%d] sentinel clobbered", type_name, sent[k]);
            REPORT(buf2, false);
        }
    }
    if (all_ok) {
        snprintf(buf2, sizeof(buf2), "ccl_copy_buf_recv <%s> 4 conj-writes + 4 sentinels correct", type_name);
        REPORT(buf2, true);
    }
}

// ============================================================
// Test: gpu_copy_and_set_zeros_aux_ab_full_tn / _nt
// ============================================================
template <typename T>
static void run_copy_aux_ab_full_tn_nt_test(const char *type_name, int a_transposed, const char *path_name)
{
    int l_rows=2, l_cols=2, nblk=2, nblk_mult_max=2, nblk_mult=2;
    int np_rows=1, np_cols=1, np_ab_fine=0, np_t_fine=0, np_dirs_fine=1;
    int my_prow=0, my_pcol=0, SM_count=1;
    const int a_n    = l_rows * l_cols;
    const int b_n    = l_rows * l_cols;
    const int aux_a_n = nblk_mult * nblk_mult_max;
    const int aux_b_n = nblk_mult * nblk_mult_max;

    T h_a[4], h_b[4], h_aux_a[4], h_aux_b[4];
    for (int k=0; k<a_n;     k++) h_a[k]     = make_val<T>(k+1);
    for (int k=0; k<b_n;     k++) h_b[k]     = make_val<T>(k+5);
    for (int k=0; k<aux_a_n; k++) h_aux_a[k] = make_val<T>(-999);
    for (int k=0; k<aux_b_n; k++) h_aux_b[k] = make_val<T>(-999);

    T *d_a, *d_b, *d_aux_a, *d_aux_b;
    HIP_CHECK(hipMalloc(&d_a,     a_n    *sizeof(T)));
    HIP_CHECK(hipMalloc(&d_b,     b_n    *sizeof(T)));
    HIP_CHECK(hipMalloc(&d_aux_a, aux_a_n*sizeof(T)));
    HIP_CHECK(hipMalloc(&d_aux_b, aux_b_n*sizeof(T)));
    HIP_CHECK(hipMemcpy(d_a,     h_a,     a_n    *sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b,     h_b,     b_n    *sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_aux_a, h_aux_a, aux_a_n*sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_aux_b, h_aux_b, aux_b_n*sizeof(T), hipMemcpyHostToDevice));

    hip_copy_and_set_zeros_aux_ab_full_tn_nt_FromC(
        dtype<T>(), a_transposed,
        (intptr_t)d_a, (intptr_t)d_b, (intptr_t)d_aux_a, (intptr_t)d_aux_b,
        l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
        np_ab_fine, np_rows, my_prow,
        np_t_fine,  np_cols, my_pcol,
        np_dirs_fine, SM_count, 0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res_a[4], h_res_b[4];
    HIP_CHECK(hipMemcpy(h_res_a, d_aux_a, aux_a_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_res_b, d_aux_b, aux_b_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_aux_a)); HIP_CHECK(hipFree(d_aux_b));

    char buf[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(k+1);
        if (!vals_eq(h_res_a[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_aux_ab_%s <%s> aux_a[%d]", path_name, type_name, k);
            REPORT(buf, false);
        }
    }
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(k+5);
        if (!vals_eq(h_res_b[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "copy_aux_ab_%s <%s> aux_b[%d]", path_name, type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "copy_aux_ab_%s <%s> aux_a and aux_b correct", path_name, type_name);
        REPORT(buf, true);
    }
}

// ============================================================
// Test: gpu_update_c_tn_nt
// ============================================================
template <typename T>
static void run_update_c_tn_nt_test(const char *type_name, int a_transposed, const char *path_name)
{
    int l_rows=2, l_cols=2, nblk=2, nblk_mult_max=2, nblk_mult=2;
    int np_rows=1, np_cols=1, np_dirs_fine=1, np_dirs_t=1, my_pdir_t=0, np_fine=0, SM_count=1;
    int beta_int=0;
    const int c_n   = l_rows * l_cols;
    const int tmp_n = 4;

    T h_tmp[4], h_c[4];
    for (int k=0; k<tmp_n; k++) h_tmp[k] = make_val<T>(k+1);
    for (int k=0; k<c_n;   k++) h_c[k]   = make_val<T>(-999);

    T *d_tmp, *d_c;
    HIP_CHECK(hipMalloc(&d_tmp, tmp_n*sizeof(T)));
    HIP_CHECK(hipMalloc(&d_c,   c_n  *sizeof(T)));
    HIP_CHECK(hipMemcpy(d_tmp, h_tmp, tmp_n*sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c,   h_c,   c_n  *sizeof(T), hipMemcpyHostToDevice));

    hip_update_c_tn_nt_FromC(dtype<T>(), a_transposed,
                              (intptr_t)d_c, (intptr_t)d_tmp, beta_int,
                              l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
                              np_rows, np_cols, np_dirs_fine,
                              np_dirs_t, my_pdir_t, np_fine,
                              SM_count, 0, (hipStream_t)0);
    HIP_CHECK(hipDeviceSynchronize());

    T h_res[4];
    HIP_CHECK(hipMemcpy(h_res, d_c, c_n*sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_tmp)); HIP_CHECK(hipFree(d_c));

    char buf[128];
    bool all_ok = true;
    for (int k=0; k<4; k++) {
        T exp = make_val<T>(k+1);
        if (!vals_eq(h_res[k], exp)) {
            all_ok = false;
            snprintf(buf, sizeof(buf), "update_c_%s <%s> c[%d]", path_name, type_name, k);
            REPORT(buf, false);
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf), "update_c_%s <%s> all 4 elements correct", path_name, type_name);
        REPORT(buf, true);
    }
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for elpa_pxgemm_multiply_gpu.h kernels (ROCm/HIP) ===\n\n");

    printf("gpu_copy_aux_full:\n");
    run_copy_aux_full_test<double>          ("double");
    run_copy_aux_full_test<float>           ("float");
    run_copy_aux_full_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_aux_full_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_full:\n");
    run_copy_and_set_zeros_aux_full_test<double>          ("double");
    run_copy_and_set_zeros_aux_full_test<float>           ("float");
    run_copy_and_set_zeros_aux_full_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_and_set_zeros_aux_full_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_a_full:\n");
    run_copy_aux_a_full_test<double>          ("double");
    run_copy_aux_a_full_test<float>           ("float");
    run_copy_aux_a_full_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_aux_a_full_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_b_full:\n");
    run_copy_aux_b_full_test<double>          ("double");
    run_copy_aux_b_full_test<float>           ("float");
    run_copy_aux_b_full_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_copy_aux_b_full_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\ngpu_ccl_copy_buf_send:\n");
    run_ccl_copy_buf_send_test<double>          ("double");
    run_ccl_copy_buf_send_test<float>           ("float");
    run_ccl_copy_buf_send_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_ccl_copy_buf_send_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\ngpu_ccl_copy_buf_recv:\n");
    run_ccl_copy_buf_recv_test<double>          ("double");
    run_ccl_copy_buf_recv_test<float>           ("float");
    run_ccl_copy_buf_recv_test<gpuDoubleComplex>("gpuDoubleComplex");
    run_ccl_copy_buf_recv_test<gpuFloatComplex> ("gpuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_ab_full (TN path):\n");
    run_copy_aux_ab_full_tn_nt_test<double>          ("double",         1, "tn");
    run_copy_aux_ab_full_tn_nt_test<float>           ("float",          1, "tn");
    run_copy_aux_ab_full_tn_nt_test<gpuDoubleComplex>("gpuDoubleComplex", 1, "tn");
    run_copy_aux_ab_full_tn_nt_test<gpuFloatComplex> ("gpuFloatComplex",  1, "tn");

    printf("\ngpu_copy_and_set_zeros_aux_ab_full (NT path):\n");
    run_copy_aux_ab_full_tn_nt_test<double>          ("double",         0, "nt");
    run_copy_aux_ab_full_tn_nt_test<float>           ("float",          0, "nt");
    run_copy_aux_ab_full_tn_nt_test<gpuDoubleComplex>("gpuDoubleComplex", 0, "nt");
    run_copy_aux_ab_full_tn_nt_test<gpuFloatComplex> ("gpuFloatComplex",  0, "nt");

    printf("\ngpu_update_c_tn_nt (TN path, beta=0):\n");
    run_update_c_tn_nt_test<double>          ("double",         1, "tn");
    run_update_c_tn_nt_test<float>           ("float",          1, "tn");
    run_update_c_tn_nt_test<gpuDoubleComplex>("gpuDoubleComplex", 1, "tn");
    run_update_c_tn_nt_test<gpuFloatComplex> ("gpuFloatComplex",  1, "tn");

    printf("\ngpu_update_c_tn_nt (NT path, beta=0):\n");
    run_update_c_tn_nt_test<double>          ("double",         0, "nt");
    run_update_c_tn_nt_test<float>           ("float",          0, "nt");
    run_update_c_tn_nt_test<gpuDoubleComplex>("gpuDoubleComplex", 0, "nt");
    run_update_c_tn_nt_test<gpuFloatComplex> ("gpuFloatComplex",  0, "nt");

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
