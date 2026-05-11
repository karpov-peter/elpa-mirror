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

// Unit tests for kernels in
// src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_complex.cu:
//   compute_hh_trafo_cuda_kernel_complex_double
//   compute_hh_trafo_cuda_kernel_complex_single
//
// The kernel applies a single Householder reflection to a distributed q
// matrix.  For one Householder vector h and scalar tau it computes:
//
//   q_col = q_col − tau · (hᴴ·q_col) · h
//
// Test parameters: blk=nb=2, ldq=2, ncols=1, nev=2.
//
// hh   = [{1,0}, {0,1}],  hh_tau[0] = {2,0}
//
// Block bid=0 acts on q[0]={3,1} and q[2]={1,2}:
//   dot = {3,1}·conj({1,0}) + {1,2}·conj({0,1}) = {5,0}
//   q[0] → {-7,1},  q[2] → {1,-8}
//
// Block bid=1 acts on q[1]={1,0} and q[3]={0,1}:
//   dot = {1,0}·conj({1,0}) + {0,1}·conj({0,1}) = {2,0}
//   q[1] → {-3,0},  q[3] → {0,-3}
//
// All expected values are small integers, exact in both double and float.

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

#include "../../../../src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_complex.cu"

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
    printf("  %-64s [%s]\n", (label), (ok) ? "PASS" : "FAIL");               \
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

// ---- launcher dispatchers ----

static void call_launch(cuDoubleComplex *q, const cuDoubleComplex *hh,
                        const cuDoubleComplex *hh_tau,
                        int nev, int nb, int ldq, int ncols, cudaStream_t s)
{
    launch_compute_hh_trafo_c_cuda_kernel_complex_double(
        q, hh, hh_tau, nev, nb, ldq, ncols, s);
}

static void call_launch(cuFloatComplex *q, const cuFloatComplex *hh,
                        const cuFloatComplex *hh_tau,
                        int nev, int nb, int ldq, int ncols, cudaStream_t s)
{
    launch_compute_hh_trafo_c_cuda_kernel_complex_single(
        q, hh, hh_tau, nev, nb, ldq, ncols, s);
}

// ============================================================
// Test: compute_hh_trafo_cuda_kernel_complex_{double,single}
//
// Layout: blk=nb=2, ldq=2, ncols=1, nev=2.
//
// q is 2×2 in column-major order (ldq=2):
//   q[0]=bid0_row0, q[1]=bid1_row0, q[2]=bid0_row1, q[3]=bid1_row1
//
// Each block bid applies:
//   q[bid + j*ldq] -= dot * tau * hh[j]    for j=0..nb-1
// where dot = sum_j q[bid + j*ldq] * conj(hh[j]).
// ============================================================
template <typename T>
static void run_hh_trafo_complex_test(const char *type_name)
{
    // nev=2 blocks each of nb=2 threads; ldq=2, ncols=1
    const int nev = 2, nb = 2, ldq = 2, ncols = 1;
    const int q_n   = ldq * (ncols + nb - 1);  // 2*(1+2-1)=4
    const int hh_n  = nb * ncols;               // 2
    const int tau_n = ncols;                     // 1

    T h_q[4], h_hh[2], h_tau[1];

    // q block: column-major, bid selects row
    h_q[0] = make_cx<T>( 3.0,  1.0);  // bid=0, j=0
    h_q[1] = make_cx<T>( 1.0,  0.0);  // bid=1, j=0
    h_q[2] = make_cx<T>( 1.0,  2.0);  // bid=0, j=1
    h_q[3] = make_cx<T>( 0.0,  1.0);  // bid=1, j=1

    // Householder vector h = [{1,0},{0,1}], tau = {2,0}
    h_hh[0]  = make_cx<T>(1.0, 0.0);
    h_hh[1]  = make_cx<T>(0.0, 1.0);
    h_tau[0] = make_cx<T>(2.0, 0.0);

    T *d_q, *d_hh, *d_tau;
    CUDA_CHECK(cudaMalloc(&d_q,   q_n   * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_hh,  hh_n  * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tau, tau_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q,   h_q,   q_n   * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hh,  h_hh,  hh_n  * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, h_tau, tau_n * sizeof(T), cudaMemcpyHostToDevice));

    call_launch(d_q, d_hh, d_tau, nev, nb, ldq, ncols, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[4];
    CUDA_CHECK(cudaMemcpy(h_result, d_q, q_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_hh));
    CUDA_CHECK(cudaFree(d_tau));

    char buf[128];

    // bid=0: dot={5,0}, q -= dot*tau*h
    snprintf(buf, sizeof(buf), "hh_trafo_complex <%s> bid=0 row0: {3,1}→{-7,1}",  type_name);
    REPORT(buf, cx_eq(h_result[0], -7.0, 1.0));
    snprintf(buf, sizeof(buf), "hh_trafo_complex <%s> bid=0 row1: {1,2}→{1,-8}",  type_name);
    REPORT(buf, cx_eq(h_result[2],  1.0, -8.0));

    // bid=1: dot={2,0}, q -= dot*tau*h
    snprintf(buf, sizeof(buf), "hh_trafo_complex <%s> bid=1 row0: {1,0}→{-3,0}", type_name);
    REPORT(buf, cx_eq(h_result[1], -3.0,  0.0));
    snprintf(buf, sizeof(buf), "hh_trafo_complex <%s> bid=1 row1: {0,1}→{0,-3}", type_name);
    REPORT(buf, cx_eq(h_result[3],  0.0, -3.0));
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for ev_tridi_band_nvidia_gpu_complex.cu kernels ===\n\n");

    printf("compute_hh_trafo_cuda_kernel_complex_double:\n");
    run_hh_trafo_complex_test<cuDoubleComplex>("cuDoubleComplex");

    printf("\ncompute_hh_trafo_cuda_kernel_complex_single:\n");
    run_hh_trafo_complex_test<cuFloatComplex>("cuFloatComplex");

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
