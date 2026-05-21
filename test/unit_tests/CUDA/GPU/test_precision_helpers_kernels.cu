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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config-f90.h"

#ifdef WITH_UNIT_TESTS
#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "../../../../src/GPU/CUDA/precision_helpers_cuda.cu"

#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Number of elements per test.  Chosen > PHCU_BLOCK_SIZE=256 to exercise
// the two-block code path.
static const int N = 300;

// Half-precision tolerance for comparisons via __half2float.
static const float HALF_TOL = 1e-3f;

#endif /* WANT_HALF_PRECISION_REAL || WANT_HALF_PRECISION_COMPLEX */

// -------------------------------------------------------------------------
// Real tests
// -------------------------------------------------------------------------
#ifdef WANT_HALF_PRECISION_REAL

// double[] → __half[]: every element i holds the value (i % 16) + 1.0.
// These are small integers, exactly representable in fp16.
static int test_double_to_half(void)
{
    double *h_src  = (double *)malloc(N * sizeof(double));
    float  *h_exp  = (float  *)malloc(N * sizeof(float));
    __half *h_dst  = (__half *)malloc(N * sizeof(__half));
    double *d_src; __half *d_dst;

    for (int i = 0; i < N; i++) {
        double v = (double)((i % 16) + 1);
        h_src[i] = v;
        h_exp[i] = (float)v;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(double), cudaMemcpyHostToDevice));

    cuda_copy_double_to_half_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float got = __half2float(h_dst[i]);
        if (fabsf(got - h_exp[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_exp); free(h_dst);
    printf("  cuda_copy_double_to_half      (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// __half[] → double[]: reverse of the above.
static int test_half_to_double(void)
{
    __half  *h_src = (__half  *)malloc(N * sizeof(__half));
    double  *h_dst = (double  *)malloc(N * sizeof(double));
    double  *h_exp = (double  *)malloc(N * sizeof(double));
    __half *d_src; double *d_dst;

    for (int i = 0; i < N; i++) {
        double v = (double)((i % 16) + 1);
        h_src[i] = __float2half((float)v);
        h_exp[i] = v;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(__half), cudaMemcpyHostToDevice));

    cuda_copy_half_to_double_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        // half→float→double; tolerance covers the float→double cast
        if (fabs(h_dst[i] - h_exp[i]) > (double)HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_dst); free(h_exp);
    printf("  cuda_copy_half_to_double      (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// float[] → __half[]: use values with 0.5 fractional part (exact in fp16).
static int test_float_to_half(void)
{
    float  *h_src = (float  *)malloc(N * sizeof(float));
    float  *h_exp = (float  *)malloc(N * sizeof(float));
    __half *h_dst = (__half *)malloc(N * sizeof(__half));
    float *d_src; __half *d_dst;

    for (int i = 0; i < N; i++) {
        float v = (float)(i % 16) + 0.5f;
        h_src[i] = v;
        h_exp[i] = v;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    cuda_copy_float_to_half_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float got = __half2float(h_dst[i]);
        if (fabsf(got - h_exp[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_exp); free(h_dst);
    printf("  cuda_copy_float_to_half       (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// __half[] → float[]: reverse of the above.
static int test_half_to_float(void)
{
    __half *h_src = (__half *)malloc(N * sizeof(__half));
    float  *h_dst = (float  *)malloc(N * sizeof(float));
    float  *h_exp = (float  *)malloc(N * sizeof(float));
    __half *d_src; float *d_dst;

    for (int i = 0; i < N; i++) {
        float v = (float)(i % 16) + 0.5f;
        h_src[i] = __float2half(v);
        h_exp[i] = v;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(__half), cudaMemcpyHostToDevice));

    cuda_copy_half_to_float_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_dst[i] - h_exp[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_dst); free(h_exp);
    printf("  cuda_copy_half_to_float       (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif /* WANT_HALF_PRECISION_REAL */

// -------------------------------------------------------------------------
// Complex tests
// -------------------------------------------------------------------------
#ifdef WANT_HALF_PRECISION_COMPLEX

// cuDoubleComplex[] → __half2[]: element i holds (re=i%16+1, im=i%8+1).
static int test_double_complex_to_half2(void)
{
    cuDoubleComplex *h_src = (cuDoubleComplex *)malloc(N * sizeof(cuDoubleComplex));
    __half2         *h_dst = (__half2         *)malloc(N * sizeof(__half2));
    float           *h_exp_re = (float *)malloc(N * sizeof(float));
    float           *h_exp_im = (float *)malloc(N * sizeof(float));
    cuDoubleComplex *d_src; __half2 *d_dst;

    for (int i = 0; i < N; i++) {
        double re = (double)(i % 16 + 1);
        double im = (double)(i % 8  + 1);
        h_src[i]     = make_cuDoubleComplex(re, im);
        h_exp_re[i]  = (float)re;
        h_exp_im[i]  = (float)im;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(__half2)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuda_copy_double_complex_to_half2_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(__half2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float got_re = __half2float(h_dst[i].x);
        float got_im = __half2float(h_dst[i].y);
        if (fabsf(got_re - h_exp_re[i]) > HALF_TOL ||
            fabsf(got_im - h_exp_im[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_dst); free(h_exp_re); free(h_exp_im);
    printf("  cuda_copy_double_complex_to_half2 (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// __half2[] → cuDoubleComplex[]: reverse of the above.
static int test_half2_to_double_complex(void)
{
    __half2         *h_src    = (__half2         *)malloc(N * sizeof(__half2));
    cuDoubleComplex *h_dst    = (cuDoubleComplex *)malloc(N * sizeof(cuDoubleComplex));
    float           *h_exp_re = (float *)malloc(N * sizeof(float));
    float           *h_exp_im = (float *)malloc(N * sizeof(float));
    __half2 *d_src; cuDoubleComplex *d_dst;

    for (int i = 0; i < N; i++) {
        float re = (float)(i % 16 + 1);
        float im = (float)(i % 8  + 1);
        h_src[i].x  = __float2half(re);
        h_src[i].y  = __float2half(im);
        h_exp_re[i] = re;
        h_exp_im[i] = im;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(__half2), cudaMemcpyHostToDevice));

    cuda_copy_half2_to_double_complex_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float got_re = (float)h_dst[i].x;
        float got_im = (float)h_dst[i].y;
        if (fabsf(got_re - h_exp_re[i]) > HALF_TOL ||
            fabsf(got_im - h_exp_im[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_dst); free(h_exp_re); free(h_exp_im);
    printf("  cuda_copy_half2_to_double_complex (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// cuFloatComplex[] → __half2[].
static int test_single_complex_to_half2(void)
{
    cuFloatComplex *h_src    = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
    __half2        *h_dst    = (__half2        *)malloc(N * sizeof(__half2));
    float          *h_exp_re = (float *)malloc(N * sizeof(float));
    float          *h_exp_im = (float *)malloc(N * sizeof(float));
    cuFloatComplex *d_src; __half2 *d_dst;

    for (int i = 0; i < N; i++) {
        float re = (float)(i % 16 + 1) + 0.5f;
        float im = (float)(i % 8  + 1) + 0.5f;
        h_src[i]     = make_cuFloatComplex(re, im);
        h_exp_re[i]  = re;
        h_exp_im[i]  = im;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(__half2)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    cuda_copy_single_complex_to_half2_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(__half2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float got_re = __half2float(h_dst[i].x);
        float got_im = __half2float(h_dst[i].y);
        if (fabsf(got_re - h_exp_re[i]) > HALF_TOL ||
            fabsf(got_im - h_exp_im[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_dst); free(h_exp_re); free(h_exp_im);
    printf("  cuda_copy_single_complex_to_half2 (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// __half2[] → cuFloatComplex[]: reverse of the above.
static int test_half2_to_single_complex(void)
{
    __half2        *h_src    = (__half2        *)malloc(N * sizeof(__half2));
    cuFloatComplex *h_dst    = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
    float          *h_exp_re = (float *)malloc(N * sizeof(float));
    float          *h_exp_im = (float *)malloc(N * sizeof(float));
    __half2 *d_src; cuFloatComplex *d_dst;

    for (int i = 0; i < N; i++) {
        float re = (float)(i % 16 + 1) + 0.5f;
        float im = (float)(i % 8  + 1) + 0.5f;
        h_src[i].x  = __float2half(re);
        h_src[i].y  = __float2half(im);
        h_exp_re[i] = re;
        h_exp_im[i] = im;
    }
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(__half2), cudaMemcpyHostToDevice));

    cuda_copy_half2_to_single_complex_FromC((intptr_t)d_src, (intptr_t)d_dst, N, (intptr_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float got_re = h_dst[i].x;
        float got_im = h_dst[i].y;
        if (fabsf(got_re - h_exp_re[i]) > HALF_TOL ||
            fabsf(got_im - h_exp_im[i]) > HALF_TOL) { ok = 0; break; }
    }
    free(h_src); free(h_dst); free(h_exp_re); free(h_exp_im);
    printf("  cuda_copy_half2_to_single_complex (N=%d): [%s]\n", N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif /* WANT_HALF_PRECISION_COMPLEX */

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)

int main(void)
{
    int failures = 0;

    printf("=== Unit tests for precision helper kernels (CUDA) ===\n\n");

#ifdef WANT_HALF_PRECISION_REAL
    printf("Real precision conversions:\n");
    failures += test_double_to_half();
    failures += test_half_to_double();
    failures += test_float_to_half();
    failures += test_half_to_float();
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
    printf("\nComplex precision conversions:\n");
    failures += test_double_complex_to_half2();
    failures += test_half2_to_double_complex();
    failures += test_single_complex_to_half2();
    failures += test_half2_to_single_complex();
#endif

    printf("\n=== Summary: %d failure(s) ===\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else /* neither WANT_HALF_PRECISION_REAL nor WANT_HALF_PRECISION_COMPLEX */

int main(void)
{
    fprintf(stderr,
            "Error: this test should only be called when the ELPA build "
            "is configured with half-precision support enabled\n");
    abort();
}

#endif /* WANT_HALF_PRECISION_REAL || WANT_HALF_PRECISION_COMPLEX */

#else /* !WITH_NVIDIA_GPU_VERSION */

int main(void)
{
    fprintf(stderr,
            "Error: this test should only be called when the ELPA build "
            "is configured with NVIDIA GPU support enabled\n");
    abort();
}

#endif /* WITH_NVIDIA_GPU_VERSION */
#endif /* WITH_UNIT_TESTS */
