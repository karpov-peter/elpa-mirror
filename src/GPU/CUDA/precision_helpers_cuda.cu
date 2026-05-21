#if 0
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
//
// Author: Andreas Marek, MPCDF
#endif


#include "config-f90.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define PHCU_BLOCK_SIZE 256

// =========================================================================
// Real: double <-> half
// =========================================================================
#ifdef WANT_HALF_PRECISION_REAL

#include <cuda_fp16.h>

__global__ static void kernel_double_to_half(const double *src, __half *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __double2half(src[i]);
}

__global__ static void kernel_half_to_double(const __half *src, double *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)__half2float(src[i]);
}

__global__ static void kernel_float_to_half(const float *src, __half *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ static void kernel_half_to_float(const __half *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

extern "C" void cuda_copy_double_to_half_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                         int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const double *src = (const double *)src_ptr;
    __half       *dst = (__half *)dst_ptr;
    cudaStream_t  stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_double_to_half<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_double_to_half<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_double_to_half failed: %s\n", cudaGetErrorString(err));
}

extern "C" void cuda_copy_half_to_double_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                         int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const __half *src = (const __half *)src_ptr;
    double       *dst = (double *)dst_ptr;
    cudaStream_t  stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_half_to_double<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_half_to_double<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_half_to_double failed: %s\n", cudaGetErrorString(err));
}

extern "C" void cuda_copy_float_to_half_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                        int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const float *src = (const float *)src_ptr;
    __half      *dst = (__half *)dst_ptr;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_float_to_half<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_float_to_half<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_float_to_half failed: %s\n", cudaGetErrorString(err));
}

extern "C" void cuda_copy_half_to_float_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                        int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const __half *src = (const __half *)src_ptr;
    float        *dst = (float *)dst_ptr;
    cudaStream_t  stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_half_to_float<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_half_to_float<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_half_to_float failed: %s\n", cudaGetErrorString(err));
}

#endif /* WANT_HALF_PRECISION_REAL */

// =========================================================================
// Complex: double complex <-> half2,  single complex <-> half2
// n = number of complex numbers; each maps to one __half2 (re, im).
// =========================================================================
#ifdef WANT_HALF_PRECISION_COMPLEX

#ifndef WANT_HALF_PRECISION_REAL
#include <cuda_fp16.h>
#endif
#include <cuComplex.h>

__global__ static void kernel_double_complex_to_half2(const cuDoubleComplex *src,
                                                       __half2 *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i].x = __double2half(src[i].x);
        dst[i].y = __double2half(src[i].y);
    }
}

__global__ static void kernel_half2_to_double_complex(const __half2 *src,
                                                       cuDoubleComplex *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i].x = (double)__half2float(src[i].x);
        dst[i].y = (double)__half2float(src[i].y);
    }
}

__global__ static void kernel_single_complex_to_half2(const cuFloatComplex *src,
                                                       __half2 *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i].x = __float2half(src[i].x);
        dst[i].y = __float2half(src[i].y);
    }
}

__global__ static void kernel_half2_to_single_complex(const __half2 *src,
                                                       cuFloatComplex *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i].x = __half2float(src[i].x);
        dst[i].y = __half2float(src[i].y);
    }
}

extern "C" void cuda_copy_double_complex_to_half2_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                                   int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const cuDoubleComplex *src = (const cuDoubleComplex *)src_ptr;
    __half2               *dst = (__half2 *)dst_ptr;
    cudaStream_t           stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_double_complex_to_half2<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_double_complex_to_half2<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_double_complex_to_half2 failed: %s\n", cudaGetErrorString(err));
}

extern "C" void cuda_copy_half2_to_double_complex_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                                   int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const __half2         *src = (const __half2 *)src_ptr;
    cuDoubleComplex       *dst = (cuDoubleComplex *)dst_ptr;
    cudaStream_t           stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_half2_to_double_complex<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_half2_to_double_complex<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_half2_to_double_complex failed: %s\n", cudaGetErrorString(err));
}

extern "C" void cuda_copy_single_complex_to_half2_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                                   int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const cuFloatComplex *src = (const cuFloatComplex *)src_ptr;
    __half2              *dst = (__half2 *)dst_ptr;
    cudaStream_t          stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_single_complex_to_half2<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_single_complex_to_half2<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_single_complex_to_half2 failed: %s\n", cudaGetErrorString(err));
}

extern "C" void cuda_copy_half2_to_single_complex_FromC(intptr_t src_ptr, intptr_t dst_ptr,
                                                   int n, intptr_t stream_ptr)
{
    if (n <= 0) return;
    const __half2        *src = (const __half2 *)src_ptr;
    cuFloatComplex       *dst = (cuFloatComplex *)dst_ptr;
    cudaStream_t          stream = (cudaStream_t)stream_ptr;
    int grid = (n + PHCU_BLOCK_SIZE - 1) / PHCU_BLOCK_SIZE;
#ifdef WITH_GPU_STREAMS
    kernel_half2_to_single_complex<<<grid, PHCU_BLOCK_SIZE, 0, stream>>>(src, dst, n);
#else
    kernel_half2_to_single_complex<<<grid, PHCU_BLOCK_SIZE>>>(src, dst, n);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cuda_copy_half2_to_single_complex failed: %s\n", cudaGetErrorString(err));
}

#endif /* WANT_HALF_PRECISION_COMPLEX */
