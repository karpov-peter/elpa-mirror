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

// Unit tests for kernels in src/elpa1/GPU/CUDA/trans_ev_cuda.cu:
//   cuda_scale_qmat_double_complex_kernel
//   cuda_scale_qmat_float_complex_kernel

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

#include "../../../../src/elpa1/GPU/CUDA/trans_ev_cuda.cu"

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
    printf("  %-60s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
    if (!(ok)) g_failures++;                                                  \
  } while (0)

// ---- type helpers ----

template <typename T>
static T make_cx(double re, double im)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>)
        return make_cuDoubleComplex(re, im);
    else
        return make_cuFloatComplex((float)re, (float)im);
}

template <typename T>
static bool cx_eq(T a, double re, double im)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex>)
        return a.x == re && a.y == im;
    else
        return a.x == (float)re && a.y == (float)im;
}

template <typename T>
static T dev_read_elem(T *d_ptr)
{
    T h;
    CUDA_CHECK(cudaMemcpy(&h, d_ptr, sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// ---- dispatcher ----

template <typename T>
static void call_scale_qmat(int ldq, int l_cols, T *q_dev, T *tau_dev, cudaStream_t s);

template <>
void call_scale_qmat<cuDoubleComplex>(int ldq, int l_cols,
                                      cuDoubleComplex *q_dev, cuDoubleComplex *tau_dev,
                                      cudaStream_t s)
{
    cuda_scale_qmat_double_complex_FromC(ldq, l_cols,
                                         (double _Complex *)q_dev,
                                         (double _Complex *)tau_dev, s);
}

template <>
void call_scale_qmat<cuFloatComplex>(int ldq, int l_cols,
                                     cuFloatComplex *q_dev, cuFloatComplex *tau_dev,
                                     cudaStream_t s)
{
    cuda_scale_qmat_float_complex_FromC(ldq, l_cols,
                                        (float _Complex *)q_dev,
                                        (float _Complex *)tau_dev, s);
}

// ============================================================
// Test: cuda_scale_qmat_{double,float}_complex_kernel
//
// q[ldq*col] = q[ldq*col] * (1 - tau[1])   for col < l_cols
//
// Parameters: ldq=4, l_cols=3
// tau[1] = {0.5, 0.25}  →  factor = {0.5, -0.25}
//
// Active elements (at indices 0, 4, 8 in col-major q):
//   q[0]={1,2}   →  {1.0, 0.75}
//   q[4]={5,6}   →  {4.0, 1.75}
//   q[8]={9,10}  →  {7.0, 2.75}
//
// All other elements (ldq padding rows) are sentinels and must
// remain unchanged.
// ============================================================
template <typename T>
static void run_scale_qmat_test(const char *type_name)
{
    int ldq = 4, l_cols = 3;
    const int q_n = ldq * l_cols;  // 12

    T h_q[12], h_tau[2];
    h_tau[0] = make_cx<T>(99.0, 99.0);   // not accessed by kernel
    h_tau[1] = make_cx<T>(0.5,  0.25);

    for (int i = 0; i < q_n; i++) h_q[i] = make_cx<T>(99.0, 99.0);
    h_q[0] = make_cx<T>(1.0,  2.0);
    h_q[4] = make_cx<T>(5.0,  6.0);
    h_q[8] = make_cx<T>(9.0, 10.0);

    T *d_q, *d_tau;
    CUDA_CHECK(cudaMalloc(&d_q,   q_n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tau, 2   * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q,   h_q,   q_n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, h_tau, 2   * sizeof(T), cudaMemcpyHostToDevice));

    call_scale_qmat(ldq, l_cols, d_q, d_tau, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[12];
    CUDA_CHECK(cudaMemcpy(h_result, d_q, q_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_tau));

    char buf[128];
    // active entries
    snprintf(buf, sizeof(buf), "scale_qmat <%s> q[0]={1,2}→{1.0,0.75}",   type_name);
    REPORT(buf, cx_eq(h_result[0], 1.0, 0.75));
    snprintf(buf, sizeof(buf), "scale_qmat <%s> q[4]={5,6}→{4.0,1.75}",   type_name);
    REPORT(buf, cx_eq(h_result[4], 4.0, 1.75));
    snprintf(buf, sizeof(buf), "scale_qmat <%s> q[8]={9,10}→{7.0,2.75}",  type_name);
    REPORT(buf, cx_eq(h_result[8], 7.0, 2.75));
    // padding rows must be untouched
    snprintf(buf, sizeof(buf), "scale_qmat <%s> q[1] sentinel unchanged",  type_name);
    REPORT(buf, cx_eq(h_result[1], 99.0, 99.0));
    snprintf(buf, sizeof(buf), "scale_qmat <%s> q[5] sentinel unchanged",  type_name);
    REPORT(buf, cx_eq(h_result[5], 99.0, 99.0));
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for trans_ev_cuda.cu kernels (CUDA) ===\n\n");

    printf("cuda_scale_qmat_double_complex_kernel:\n");
    run_scale_qmat_test<cuDoubleComplex>("cuDoubleComplex");

    printf("\ncuda_scale_qmat_float_complex_kernel:\n");
    run_scale_qmat_test<cuFloatComplex>("cuFloatComplex");

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
