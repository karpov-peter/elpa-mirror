// Unit tests for kernels in
// src/multiply_a_b/GPU/elpa_pxgemm_multiply_gpu.h:
//
//   gpu_copy_aux_full                      – strided column copy
//   gpu_copy_and_set_zeros_aux_full        – copy + zero-pad to nblk_mult×nblk_mult
//   gpu_copy_and_set_zeros_aux_a_full      – block-permuted column copy
//   gpu_copy_and_set_zeros_aux_b_full      – block-permuted row copy
//   gpu_ccl_copy_buf_send                  – block scatter into send buffer
//   gpu_ccl_copy_buf_recv                  – transposed scatter+conj into recv buffer
//   gpu_copy_and_set_zeros_aux_ab_full_tn  – TN path: aux_a and aux_b fill
//   gpu_copy_and_set_zeros_aux_ab_full_nt  – NT path: aux_a and aux_b fill
//   gpu_update_c_tn_nt (TN, beta=0)        – scatter tmp1_full into c (TN path)
//   gpu_update_c_tn_nt (NT, beta=0)        – scatter tmp1_full into c (NT path)
//
// All tests use small integer values so equality comparisons are exact.
// Index derivations are given in per-test comments.

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

#include "../../../../src/multiply_a_b/GPU/CUDA/elpa_pxgemm_multiply_cuda.cu"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
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
template <> double          make_val<double>(int n)         { return (double)n; }
template <> float           make_val<float>(int n)          { return (float)n; }
template <> cuDoubleComplex make_val<cuDoubleComplex>(int n) { return make_cuDoubleComplex((double)n, (double)(n+100)); }
template <> cuFloatComplex  make_val<cuFloatComplex>(int n)  { return make_cuFloatComplex((float)n, (float)(n+100)); }

template <typename T>
static bool vals_eq(T a, T b)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        return a.x == b.x && a.y == b.y;
    else
        return a == b;
}

// complex conjugate of make_val(n): negate imaginary part for complex, identity for real
template <typename T>
static T conj_val(T v)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>)
        return make_cuDoubleComplex(v.x, -v.y);
    else if constexpr (std::is_same_v<T, cuFloatComplex>)
        return make_cuFloatComplex(v.x, -v.y);
    else
        return v;
}

template <typename T> static T make_zero();
template <> double          make_zero<double>()          { return 0.0; }
template <> float           make_zero<float>()           { return 0.0f; }
template <> cuDoubleComplex make_zero<cuDoubleComplex>()  { return make_cuDoubleComplex(0.0, 0.0); }
template <> cuFloatComplex  make_zero<cuFloatComplex>()   { return make_cuFloatComplex(0.0f, 0.0f); }

template <typename T> static char dtype();
template <> char dtype<double>()         { return 'D'; }
template <> char dtype<float>()          { return 'S'; }
template <> char dtype<cuDoubleComplex>() { return 'Z'; }
template <> char dtype<cuFloatComplex>()  { return 'C'; }

// ============================================================
// Test: gpu_copy_aux_full
//
// l_rows=3, l_cols=2, lld_lhs=4, lld_rhs=3
// Kernel: lhs[i + j*lld_lhs] = rhs[i + j*lld_rhs]  for i<l_rows, j<l_cols
//
// rhs[k]=make_val(k+1): rhs[0..5]={1..6}
// lhs size=lld_lhs*l_cols=8, init to sentinel
//
// Written (0-based i,j):
//   j=0: lhs[0]=rhs[0]=1, lhs[1]=rhs[1]=2, lhs[2]=rhs[2]=3
//   j=1: lhs[4]=rhs[3]=4, lhs[5]=rhs[4]=5, lhs[6]=rhs[5]=6
// Sentinels: lhs[3], lhs[7]
// ============================================================
template <typename T>
static void run_copy_aux_full_test(const char *type_name)
{
    int l_rows=3, l_cols=2, lld_lhs=4, lld_rhs=3;
    const int rhs_n = lld_rhs * l_cols;   // 6
    const int lhs_n = lld_lhs * l_cols;   // 8

    T h_rhs[6], h_lhs[8];
    for (int k=0; k<rhs_n; k++) h_rhs[k] = make_val<T>(k+1);
    for (int k=0; k<lhs_n; k++) h_lhs[k] = make_val<T>(-999);

    T *d_lhs, *d_rhs;
    CUDA_CHECK(cudaMalloc(&d_lhs, lhs_n*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs, rhs_n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_lhs, h_lhs, lhs_n*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs, rhs_n*sizeof(T), cudaMemcpyHostToDevice));

    cuda_copy_aux_full_FromC(dtype<T>(), (intptr_t)d_lhs, (intptr_t)d_rhs,
                             l_rows, l_cols, lld_lhs, lld_rhs, 0,
                             (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[8];
    CUDA_CHECK(cudaMemcpy(h_res, d_lhs, lhs_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_lhs)); CUDA_CHECK(cudaFree(d_rhs));

    // Written positions and the rhs index they come from
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
//
// l_rows=2, l_cols=2, nblk_mult=3
// Kernel (i_loc=threadIdx.x, j_loc=blockIdx.x, both 0..nblk_mult-1):
//   if (i_loc<l_rows && j_loc<l_cols): aux[i+j*nblk_mult] = a[i+j*l_rows]
//   else:                              aux[i+j*nblk_mult] = 0
//
// a[0..3]={1,2,3,4}
// Expected aux[0..8]:
//   j=0: [0]=1, [1]=2, [2]=0
//   j=1: [3]=3, [4]=4, [5]=0
//   j=2: [6]=0, [7]=0, [8]=0
// ============================================================
template <typename T>
static void run_copy_and_set_zeros_aux_full_test(const char *type_name)
{
    int l_rows=2, l_cols=2, nblk_mult=3;
    const int a_n   = l_rows * l_cols;      // 4
    const int aux_n = nblk_mult * nblk_mult; // 9

    T h_a[4], h_aux[9];
    for (int k=0; k<a_n;   k++) h_a[k]   = make_val<T>(k+1);
    for (int k=0; k<aux_n; k++) h_aux[k] = make_val<T>(-999);

    T *d_a, *d_aux;
    CUDA_CHECK(cudaMalloc(&d_a,   a_n  *sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_aux, aux_n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,   h_a,   a_n  *sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux, aux_n*sizeof(T), cudaMemcpyHostToDevice));

    cuda_copy_and_set_zeros_aux_full_FromC(dtype<T>(), (intptr_t)d_a, (intptr_t)d_aux,
                                           l_rows, l_cols, nblk_mult, 0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[9];
    CUDA_CHECK(cudaMemcpy(h_res, d_aux, aux_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_aux));

    // aux[i + j*nblk_mult] expected:
    // i<l_rows && j<l_cols → a[i+j*l_rows]; otherwise 0
    // Written positions and expected values
    const int written_idx[] = {0,1, 3,4};
    const int a_src[]       = {0,1, 2,3};  // a[a_src[k]] → make_val(a_src[k]+1)
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
//
// l_rows=2, nblk_mult_cols=2, nblk=2, np_bc_fine=2, np_cols_fine=2, np_cols=2
// blocks=nblk=2 (blockIdx.x=dj0), threads=MAX_THREADS_PER_BLOCK (inner i loop)
//
// nblk_mult_cols/nblk=1, no partial block.
// j_block_loc_fine=0: j_block_loc=(2+0*2)/2=1
//
//   dj0=0: aux_a[i + (0+0*2)*2] = a[i + (0+1*2)*2] = a[i+4]
//   dj0=1: aux_a[i + (1+0*2)*2] = a[i + (1+1*2)*2] = a[i+6]
//
// a[0..7]={1..8}, aux_a size=4, init to sentinel
// Expected: aux_a={a[4],a[5],a[6],a[7]}={5,6,7,8}
// ============================================================
template <typename T>
static void run_copy_aux_a_full_test(const char *type_name)
{
    int l_rows=2, nblk_mult_cols=2, nblk=2, np_bc_fine=2, np_cols_fine=2, np_cols=2;
    const int a_n    = 8;  // needs to reach a[7]
    const int aux_n  = l_rows * nblk_mult_cols; // 4

    T h_a[8], h_aux[4];
    for (int k=0; k<a_n;  k++) h_a[k]   = make_val<T>(k+1);
    for (int k=0; k<aux_n; k++) h_aux[k] = make_val<T>(-999);

    T *d_a, *d_aux;
    CUDA_CHECK(cudaMalloc(&d_a,   a_n  *sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_aux, aux_n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,   h_a,   a_n  *sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux, aux_n*sizeof(T), cudaMemcpyHostToDevice));

    cuda_copy_and_set_zeros_aux_a_full_FromC(dtype<T>(), (intptr_t)d_a, (intptr_t)d_aux,
                                             l_rows, nblk_mult_cols, nblk,
                                             np_bc_fine, np_cols_fine, np_cols,
                                             0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[4];
    CUDA_CHECK(cudaMemcpy(h_res, d_aux, aux_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_aux));

    // aux[k] = a[k+4] = make_val(k+5)
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
//
// l_rows=4, l_cols=2, nblk_mult=2, nblk_mult_rows=2, nblk=2,
// np_fine=2, np_rows_fine=2, np_rows=2, SM_count=1
//
// nblk_mult_rows/nblk=1, no partial block.
// i_block_loc_fine=0: i_block_loc=(2+0*2)/2=1
//
//   j=0, di=0: aux_b[0+0+0*2] = b[0+2+0*4] = b[2] = 3
//   j=0, di=1: aux_b[1+0+0*2] = b[1+2+0*4] = b[3] = 4
//   j=1, di=0: aux_b[0+0+1*2] = b[0+2+1*4] = b[6] = 7
//   j=1, di=1: aux_b[1+0+1*2] = b[1+2+1*4] = b[7] = 8
//
// b[0..7]={1..8}, aux_b size=4
// Expected: aux_b={3,4,7,8}
// ============================================================
template <typename T>
static void run_copy_aux_b_full_test(const char *type_name)
{
    int l_rows=4, l_cols=2, nblk_mult=2, nblk_mult_rows=2, nblk=2;
    int np_fine=2, np_rows_fine=2, np_rows=2, SM_count=1;
    const int b_n   = l_rows * l_cols;   // 8
    const int aux_n = nblk_mult * l_cols; // 4

    T h_b[8], h_aux[4];
    for (int k=0; k<b_n;   k++) h_b[k]   = make_val<T>(k+1);
    for (int k=0; k<aux_n; k++) h_aux[k] = make_val<T>(-999);

    T *d_b, *d_aux;
    CUDA_CHECK(cudaMalloc(&d_b,   b_n  *sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_aux, aux_n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_b,   h_b,   b_n  *sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux, h_aux, aux_n*sizeof(T), cudaMemcpyHostToDevice));

    cuda_copy_and_set_zeros_aux_b_full_FromC(dtype<T>(), (intptr_t)d_b, (intptr_t)d_aux,
                                             l_rows, l_cols, nblk_mult, nblk_mult_rows, nblk,
                                             np_fine, np_rows_fine, np_rows,
                                             SM_count, 0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[4];
    CUDA_CHECK(cudaMemcpy(h_res, d_aux, aux_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_aux));

    // Expected: {3,4,7,8}
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
//
// l_rows=4, l_cols=2, lld_buf=2, nblk=2
// np_fine=2, np_bc_fine=0, np_rows_fine=2, np_cols_fine=1, np_rows=2, np_cols=1
// i_block_loc_fine_max=0, j_block_loc_fine_max=0, SM_count=1
//
// j_block_loc=(0+0)/1=0, i_block_loc=(2+0)/2=1
// nblk_cut_col=min(2,2-0)=2, nblk_cut_row=min(2,4-2)=2
//
//   di=0,dj=0: buf_send[0+0*2] = a[0+2+0*4] = a[2] = 3
//   di=1,dj=0: buf_send[1+0*2] = a[1+2+0*4] = a[3] = 4
//   di=0,dj=1: buf_send[0+1*2] = a[0+2+1*4] = a[6] = 7
//   di=1,dj=1: buf_send[1+1*2] = a[1+2+1*4] = a[7] = 8
//
// a[0..7]={1..8}, buf_send size=4, init to sentinel
// Expected: buf_send={3,4,7,8}
// ============================================================
template <typename T>
static void run_ccl_copy_buf_send_test(const char *type_name)
{
    int l_rows=4, l_cols=2, lld_buf=2, nblk=2;
    int np_fine=2, np_bc_fine=0, np_rows_fine=2, np_cols_fine=1, np_rows=2, np_cols=1;
    int i_block_loc_fine_max=0, j_block_loc_fine_max=0, SM_count=1;
    const int a_n    = l_rows * l_cols;   // 8
    const int buf_n  = lld_buf * nblk;    // 4

    T h_a[8], h_buf[4];
    for (int k=0; k<a_n;  k++) h_a[k]   = make_val<T>(k+1);
    for (int k=0; k<buf_n; k++) h_buf[k] = make_val<T>(-999);

    T *d_a, *d_buf;
    CUDA_CHECK(cudaMalloc(&d_a,   a_n  *sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_buf, buf_n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,   h_a,   a_n  *sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, buf_n*sizeof(T), cudaMemcpyHostToDevice));

    cuda_ccl_copy_buf_send_FromC(dtype<T>(), (intptr_t)d_a, (intptr_t)d_buf,
                                 l_rows, l_cols, lld_buf, nblk,
                                 i_block_loc_fine_max, j_block_loc_fine_max,
                                 np_fine, np_bc_fine, np_rows_fine, np_cols_fine,
                                 np_rows, np_cols, SM_count, 0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[4];
    CUDA_CHECK(cudaMemcpy(h_res, d_buf, buf_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_buf));

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
//
// Same distribution params as buf_send (np_fine=2, np_bc_fine=0, ...).
// i_block_loc=(2+0)/2=1, j_block_loc=(0+0)/1=0
// nblk_cut_row=2, nblk_cut_col=2
//
// Kernel writes (note index swap: buf→at transposed, plus complex conjugate):
//   at_col[(di+1*2)+(dj+0)*4] = conj(buf_recv[(dj+0)+(di+0)*2])
//
//   di=0,dj=0: at_col[2] = conj(buf_recv[0])
//   di=1,dj=0: at_col[3] = conj(buf_recv[0+2]) = conj(buf_recv[2])
//   di=0,dj=1: at_col[6] = conj(buf_recv[1+0]) = conj(buf_recv[1])
//   di=1,dj=1: at_col[7] = conj(buf_recv[1+2]) = conj(buf_recv[3])
//
// buf_recv[0..3]={make_val(1..4)}, at_col size=8, init to sentinel
// Sentinels: at_col[0,1,4,5]
//
// For real types: conj is identity.
// For complex types: conj(x,y)=(x,-y), so conj(make_val(n))=(n,-(n+100)).
// ============================================================
template <typename T>
static void run_ccl_copy_buf_recv_test(const char *type_name)
{
    int l_rows=4, l_cols=2, lld_buf=2, nblk=2;
    int np_fine=2, np_bc_fine=0, np_rows_fine=2, np_cols_fine=1, np_rows=2, np_cols=1;
    int i_block_loc_fine_max=0, j_block_loc_fine_max=0, SM_count=1;
    const int buf_n  = lld_buf * nblk;    // 4
    const int at_n   = l_rows * l_cols;   // 8

    T h_buf[4], h_at[8];
    for (int k=0; k<buf_n; k++) h_buf[k] = make_val<T>(k+1);
    for (int k=0; k<at_n;  k++) h_at[k]  = make_val<T>(-999);

    T *d_buf, *d_at;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_n*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_at,  at_n *sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, buf_n*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_at,  h_at,  at_n *sizeof(T), cudaMemcpyHostToDevice));

    cuda_ccl_copy_buf_recv_FromC(dtype<T>(), (intptr_t)d_at, (intptr_t)d_buf,
                                 l_rows, l_cols, lld_buf, nblk,
                                 i_block_loc_fine_max, j_block_loc_fine_max,
                                 np_fine, np_bc_fine, np_rows_fine, np_cols_fine,
                                 np_rows, np_cols, SM_count, 0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[8];
    CUDA_CHECK(cudaMemcpy(h_res, d_at, at_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buf)); CUDA_CHECK(cudaFree(d_at));

    // at_col[2]=conj(buf[0]), at_col[3]=conj(buf[2]), at_col[6]=conj(buf[1]), at_col[7]=conj(buf[3])
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
// Test: gpu_copy_and_set_zeros_aux_ab_full_tn (a_transposed=1) and
//       gpu_copy_and_set_zeros_aux_ab_full_nt (a_transposed=0)
//
// Both use: l_rows=2, l_cols=2, nblk=2, nblk_mult_max=2, nblk_mult=2,
//           np_rows=1, np_cols=1, np_ab_fine=0, np_t_fine=0,
//           np_dirs_fine=1, my_prow=0, my_pcol=0, SM_count=1
//
// TN path (a_transposed=1):
//   Condition np_t_fine%np_cols==my_pcol: 0%1==0 ✓
//   j_block_loc_fine=0: j_block_loc=0, i_block_loc_fine=0: i_block_loc=0
//   nblk_rows_cut=nblk_cols_cut=2 (no zeros needed)
//   aux_a[di+(dj)*2] = a[di+(dj)*2]  → aux_a = a
//   b part (dnp_ab_t=0): aux_b[di+(dj)*2] = b[di+(dj)*2]  → aux_b = b
//
// NT path (a_transposed=0):
//   Condition np_t_fine%np_rows==my_prow: 0%1==0 ✓
//   Same identity mapping via different code path.
//
// a[0..3]={1,2,3,4}, b[0..3]={5,6,7,8}
// aux_a size=nblk_mult*nblk_mult_max=4, aux_b size=nblk_mult*nblk_mult_max*1=4
// Expected: aux_a={1,2,3,4}, aux_b={5,6,7,8}
// ============================================================
template <typename T>
static void run_copy_aux_ab_full_tn_nt_test(const char *type_name, int a_transposed, const char *path_name)
{
    int l_rows=2, l_cols=2, nblk=2, nblk_mult_max=2, nblk_mult=2;
    int np_rows=1, np_cols=1, np_ab_fine=0, np_t_fine=0, np_dirs_fine=1;
    int my_prow=0, my_pcol=0, SM_count=1;
    const int a_n    = l_rows * l_cols;              // 4
    const int b_n    = l_rows * l_cols;              // 4
    const int aux_a_n = nblk_mult * nblk_mult_max;  // 4
    const int aux_b_n = nblk_mult * nblk_mult_max;  // 4 (np_dirs_fine/np_cols=1)

    T h_a[4], h_b[4], h_aux_a[4], h_aux_b[4];
    for (int k=0; k<a_n;     k++) h_a[k]     = make_val<T>(k+1);
    for (int k=0; k<b_n;     k++) h_b[k]     = make_val<T>(k+5);
    for (int k=0; k<aux_a_n; k++) h_aux_a[k] = make_val<T>(-999);
    for (int k=0; k<aux_b_n; k++) h_aux_b[k] = make_val<T>(-999);

    T *d_a, *d_b, *d_aux_a, *d_aux_b;
    CUDA_CHECK(cudaMalloc(&d_a,     a_n    *sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b,     b_n    *sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_aux_a, aux_a_n*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_aux_b, aux_b_n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_a,     h_a,     a_n    *sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,     h_b,     b_n    *sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux_a, h_aux_a, aux_a_n*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux_b, h_aux_b, aux_b_n*sizeof(T), cudaMemcpyHostToDevice));

    cuda_copy_and_set_zeros_aux_ab_full_tn_nt_FromC(
        dtype<T>(), a_transposed,
        (intptr_t)d_a, (intptr_t)d_b, (intptr_t)d_aux_a, (intptr_t)d_aux_b,
        l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
        np_ab_fine, np_rows, my_prow,
        np_t_fine,  np_cols, my_pcol,
        np_dirs_fine, SM_count, 0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res_a[4], h_res_b[4];
    CUDA_CHECK(cudaMemcpy(h_res_a, d_aux_a, aux_a_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_b, d_aux_b, aux_b_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_aux_a)); CUDA_CHECK(cudaFree(d_aux_b));

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
// Test: gpu_update_c_tn_nt (TN and NT, beta_int=0)
//
// l_rows=2, l_cols=2, nblk=2, nblk_mult_max=2, nblk_mult=2,
// np_rows=1, np_cols=1, np_dirs_fine=1, np_dirs_t=1, my_pdir_t=0, np_fine=0, SM_count=1
//
// TN (a_transposed=1), beta_int=0:
//   dnp_ab_t=0: np_ab_t_fine=0, j_blf=0: j_block_loc=0, i_blf=0: i_block_loc=0
//   nblk_rows_cut=nblk_cols_cut=2
//   c[di+dj*2] = tmp1_full[di + 0 + (dj + 0 + 0*2)*2] = tmp1_full[di+dj*2]
//   → c = tmp1_full
//
// NT (a_transposed=0), beta_int=0:
//   ld_tmp1 = nblk_mult_max*(np_dirs_fine/np_dirs_t)=2
//   c[di+dj*2] = tmp1_full[di+0+0*2+(dj+0)*2] = tmp1_full[di+dj*2]
//   → c = tmp1_full
//
// tmp1_full[0..3]={1,2,3,4}, c init to sentinel
// Expected: c={1,2,3,4}
// ============================================================
template <typename T>
static void run_update_c_tn_nt_test(const char *type_name, int a_transposed, const char *path_name)
{
    int l_rows=2, l_cols=2, nblk=2, nblk_mult_max=2, nblk_mult=2;
    int np_rows=1, np_cols=1, np_dirs_fine=1, np_dirs_t=1, my_pdir_t=0, np_fine=0, SM_count=1;
    int beta_int=0;
    const int c_n    = l_rows * l_cols; // 4
    const int tmp_n  = 4;               // nblk_mult * nblk_mult_max

    T h_tmp[4], h_c[4];
    for (int k=0; k<tmp_n; k++) h_tmp[k] = make_val<T>(k+1);
    for (int k=0; k<c_n;   k++) h_c[k]   = make_val<T>(-999);

    T *d_tmp, *d_c;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_n*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c,   c_n  *sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_tmp, h_tmp, tmp_n*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c,   h_c,   c_n  *sizeof(T), cudaMemcpyHostToDevice));

    cuda_update_c_tn_nt_FromC(dtype<T>(), a_transposed,
                               (intptr_t)d_c, (intptr_t)d_tmp, beta_int,
                               l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
                               np_rows, np_cols, np_dirs_fine,
                               np_dirs_t, my_pdir_t, np_fine,
                               SM_count, 0, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_res[4];
    CUDA_CHECK(cudaMemcpy(h_res, d_c, c_n*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_c));

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
    printf("=== Unit tests for elpa_pxgemm_multiply_gpu.h kernels ===\n\n");

    printf("gpu_copy_aux_full:\n");
    run_copy_aux_full_test<double>         ("double");
    run_copy_aux_full_test<float>          ("float");
    run_copy_aux_full_test<cuDoubleComplex>("cuDoubleComplex");
    run_copy_aux_full_test<cuFloatComplex> ("cuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_full:\n");
    run_copy_and_set_zeros_aux_full_test<double>         ("double");
    run_copy_and_set_zeros_aux_full_test<float>          ("float");
    run_copy_and_set_zeros_aux_full_test<cuDoubleComplex>("cuDoubleComplex");
    run_copy_and_set_zeros_aux_full_test<cuFloatComplex> ("cuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_a_full:\n");
    run_copy_aux_a_full_test<double>         ("double");
    run_copy_aux_a_full_test<float>          ("float");
    run_copy_aux_a_full_test<cuDoubleComplex>("cuDoubleComplex");
    run_copy_aux_a_full_test<cuFloatComplex> ("cuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_b_full:\n");
    run_copy_aux_b_full_test<double>         ("double");
    run_copy_aux_b_full_test<float>          ("float");
    run_copy_aux_b_full_test<cuDoubleComplex>("cuDoubleComplex");
    run_copy_aux_b_full_test<cuFloatComplex> ("cuFloatComplex");

    printf("\ngpu_ccl_copy_buf_send:\n");
    run_ccl_copy_buf_send_test<double>         ("double");
    run_ccl_copy_buf_send_test<float>          ("float");
    run_ccl_copy_buf_send_test<cuDoubleComplex>("cuDoubleComplex");
    run_ccl_copy_buf_send_test<cuFloatComplex> ("cuFloatComplex");

    printf("\ngpu_ccl_copy_buf_recv:\n");
    run_ccl_copy_buf_recv_test<double>         ("double");
    run_ccl_copy_buf_recv_test<float>          ("float");
    run_ccl_copy_buf_recv_test<cuDoubleComplex>("cuDoubleComplex");
    run_ccl_copy_buf_recv_test<cuFloatComplex> ("cuFloatComplex");

    printf("\ngpu_copy_and_set_zeros_aux_ab_full (TN path):\n");
    run_copy_aux_ab_full_tn_nt_test<double>         ("double",         1, "tn");
    run_copy_aux_ab_full_tn_nt_test<float>          ("float",          1, "tn");
    run_copy_aux_ab_full_tn_nt_test<cuDoubleComplex>("cuDoubleComplex", 1, "tn");
    run_copy_aux_ab_full_tn_nt_test<cuFloatComplex> ("cuFloatComplex",  1, "tn");

    printf("\ngpu_copy_and_set_zeros_aux_ab_full (NT path):\n");
    run_copy_aux_ab_full_tn_nt_test<double>         ("double",         0, "nt");
    run_copy_aux_ab_full_tn_nt_test<float>          ("float",          0, "nt");
    run_copy_aux_ab_full_tn_nt_test<cuDoubleComplex>("cuDoubleComplex", 0, "nt");
    run_copy_aux_ab_full_tn_nt_test<cuFloatComplex> ("cuFloatComplex",  0, "nt");

    printf("\ngpu_update_c_tn_nt (TN path, beta=0):\n");
    run_update_c_tn_nt_test<double>         ("double",         1, "tn");
    run_update_c_tn_nt_test<float>          ("float",          1, "tn");
    run_update_c_tn_nt_test<cuDoubleComplex>("cuDoubleComplex", 1, "tn");
    run_update_c_tn_nt_test<cuFloatComplex> ("cuFloatComplex",  1, "tn");

    printf("\ngpu_update_c_tn_nt (NT path, beta=0):\n");
    run_update_c_tn_nt_test<double>         ("double",         0, "nt");
    run_update_c_tn_nt_test<float>          ("float",          0, "nt");
    run_update_c_tn_nt_test<cuDoubleComplex>("cuDoubleComplex", 0, "nt");
    run_update_c_tn_nt_test<cuFloatComplex> ("cuFloatComplex",  0, "nt");

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
    fprintf(stderr, "Error: this test requires WITH_NVIDIA_GPU_VERSION\n");
    abort();
}

#endif /* WITH_NVIDIA_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
