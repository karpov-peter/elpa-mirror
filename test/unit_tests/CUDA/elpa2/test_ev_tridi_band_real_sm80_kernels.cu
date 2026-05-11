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
// src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_real_sm80.cu:
//   compute_hh_trafo_gpu_new<double, ...>  (via launch_..._real_double)
//   compute_hh_trafo_gpu_new<float, ...>   (via launch_..._real_single)
//
// The kernel applies one Householder step:
//   q = q - tau * hh * (hh^T * q)
//
// Test parameters: nb=4, nev=8, ldq=8, ncols=1.
//
// q[k*ldq+n] = k*8+n+1  (k=0..3, n=0..7, values 1..32)
// hh[k]      = 1.0       (k=0..3)
// hh_tau[0]  = 0.5
//
// dot_n = sum_{k=0}^{3} hh[k] * q[k*8+n] = sum_{k=0}^{3} (k*8+n+1) = 4n + 52
// q[k*8+n]_new = (k*8+n+1) - 0.5*(4n+52) = 8k - n - 25
//
// All expected values are small negative integers, exact in both double and
// float.  The double kernel exercises the DMMA (MMA) inner product path;
// the float kernel uses the scalar fallback path (if constexpr).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "../../../../src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_real_sm80.cu"

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

// ---- launcher dispatchers ----

static void call_launch(double *q, const double *hh, const double *hh_tau,
                        int nev, int nb, int ldq, int ncols, cudaStream_t s)
{
    launch_compute_hh_trafo_c_cuda_sm80_kernel_real_double(
        q, hh, hh_tau, nev, nb, ldq, ncols, s);
}

static void call_launch(float *q, const float *hh, const float *hh_tau,
                        int nev, int nb, int ldq, int ncols, cudaStream_t s)
{
    launch_compute_hh_trafo_c_cuda_sm80_kernel_real_single(
        q, hh, hh_tau, nev, nb, ldq, ncols, s);
}

// ============================================================
// Test: compute_hh_trafo_gpu_new<T, 4, 8, 8, 4>
//        (reached via nb=4 in the switch dispatch)
//
// Layout: nb=4, nev=8, ldq=8, ncols=1.
//   q[k*ldq+n]  initial  = k*8+n+1    k=0..3, n=0..7
//   hh[k]                = 1.0        k=0..3
//   hh_tau[0]            = 0.5
//
//   dot_n = 4n+52
//   q[k*8+n]_new = 8k - n - 25
//
// The double variant goes through the DMMA tensor-core path (USE_MMA).
// The float variant uses the scalar register loop (if constexpr fallback).
// ============================================================
template <typename T>
static void run_hh_trafo_real_sm80_test(const char *type_name)
{
    const int nb = 4, nev = 8, ldq = 8, ncols = 1;
    const int q_n   = (ncols + nb - 1) * ldq;  // (1+4-1)*8 = 32
    const int hh_n  = nb * ncols;               // 4
    const int tau_n = ncols;                     // 1

    T h_q[32], h_hh[4], h_tau[1];

    for (int k = 0; k < nb; k++)
        for (int n = 0; n < nev; n++)
            h_q[k * ldq + n] = (T)(k * 8 + n + 1);

    for (int k = 0; k < nb; k++) h_hh[k] = (T)1.0;
    h_tau[0] = (T)0.5;

    T *d_q, *d_hh, *d_tau;
    CUDA_CHECK(cudaMalloc(&d_q,   q_n   * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_hh,  hh_n  * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tau, tau_n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_q,   h_q,   q_n   * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hh,  h_hh,  hh_n  * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, h_tau, tau_n * sizeof(T), cudaMemcpyHostToDevice));

    call_launch(d_q, d_hh, d_tau, nev, nb, ldq, ncols, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_result[32];
    CUDA_CHECK(cudaMemcpy(h_result, d_q, q_n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_hh));
    CUDA_CHECK(cudaFree(d_tau));

    // Expected: q[k*8+n] = 8k - n - 25
    char buf[128];
    bool all_ok = true;
    for (int k = 0; k < nb; k++) {
        for (int n = 0; n < nev; n++) {
            T expected = (T)(8 * k - n - 25);
            if (h_result[k * ldq + n] != expected) {
                all_ok = false;
                snprintf(buf, sizeof(buf),
                         "hh_trafo_sm80 <%s> q[k=%d,n=%d]: expected %.0f got %.0f",
                         type_name, k, n, (double)expected, (double)h_result[k * ldq + n]);
                REPORT(buf, false);
            }
        }
    }
    if (all_ok) {
        snprintf(buf, sizeof(buf),
                 "hh_trafo_sm80 <%s> all 32 elements correct (8k-n-25)", type_name);
        REPORT(buf, true);
    }
}

// ============================================================
int main(void)
{
    printf("=== Unit tests for ev_tridi_band_nvidia_gpu_real_sm80.cu kernels ===\n\n");

    printf("compute_hh_trafo_gpu_new<double> (DMMA path):\n");
    run_hh_trafo_real_sm80_test<double>("double");

    printf("\ncompute_hh_trafo_gpu_new<float> (scalar fallback):\n");
    run_hh_trafo_real_sm80_test<float>("float");

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
