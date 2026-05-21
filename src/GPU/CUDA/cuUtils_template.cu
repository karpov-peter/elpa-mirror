#if 0
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
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
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
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//
// --------------------------------------------------------------------------------------------------
//
// This file was originally written by NVIDIA
// and re-written by A. Marek, MPCDF
#endif

#include "config-f90.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cuComplex.h>
#include <type_traits>

#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

#define MAX_BLOCK_SIZE 1024

// ============================================================
// Layer 1: kernel templates — logic inline, no _body indirection
// ============================================================

template <typename T>
__global__ void my_pack_c_cuda_kernel(const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int l_nev,
    T *src, T *dst, int i_off)
{
    int b_id = blockIdx.y;
    int t_id = threadIdx.x + i_off * blockDim.x;
    int dst_ind = b_id * stripe_width + t_id;

    if (dst_ind < max_idx)
    {
        if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        {
            dst[dst_ind + (l_nev * blockIdx.x)].x = src[t_id + (stripe_width * (n_offset + blockIdx.x)) + (b_id * stripe_width * a_dim2)].x;
            dst[dst_ind + (l_nev * blockIdx.x)].y = src[t_id + (stripe_width * (n_offset + blockIdx.x)) + (b_id * stripe_width * a_dim2)].y;
        }
        else
        {
            dst[dst_ind + (l_nev * blockIdx.x)] = src[t_id + (stripe_width * (n_offset + blockIdx.x)) + (b_id * stripe_width * a_dim2)];
        }
    }
}

template <typename T>
__global__ void my_unpack_c_cuda_kernel(const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int l_nev,
    T *src, T *dst, int i_off)
{
    int b_id = blockIdx.y;
    int t_id = threadIdx.x + i_off * blockDim.x;
    int src_ind = b_id * stripe_width + t_id;

    if (src_ind < max_idx)
    {
        if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        {
            dst[t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2)].x = src[src_ind + blockIdx.x * l_nev].x;
            dst[t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2)].y = src[src_ind + blockIdx.x * l_nev].y;
        }
        else
        {
            dst[t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2)] = src[src_ind + blockIdx.x * l_nev];
        }
    }
}

template <typename T>
__global__ void extract_hh_tau_c_cuda_kernel(T *hh, T *hh_tau, const int nbw, const int n, int val)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_idx < n)
    {
        hh_tau[h_idx] = hh[h_idx * nbw];
        if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>)
        {
            if (val == 0) { hh[h_idx * nbw].x = 1.0; hh[h_idx * nbw].y = 0.0; }
            else          { hh[h_idx * nbw].x = 0.0; hh[h_idx * nbw].y = 0.0; }
        }
#if defined(WANT_HALF_PRECISION_COMPLEX)
        else if constexpr (std::is_same_v<T, __half2>)
        {
            if (val == 0) { hh[h_idx * nbw].x = __float2half(1.0f); hh[h_idx * nbw].y = __float2half(0.0f); }
            else          { hh[h_idx * nbw].x = __float2half(0.0f); hh[h_idx * nbw].y = __float2half(0.0f); }
        }
#endif
        else
        {
            hh[h_idx * nbw] = (val == 0) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
        }
    }
}

// ============================================================
// Layer 2: launcher templates — grid/block setup, kernel launch
// ============================================================

template <typename T>
void my_pack_c_cuda(const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    T *a_dev, T *row_group_dev, cudaStream_t my_stream)
{
    if (stripe_width <= 0) return;

    dim3 grid_size = dim3(row_count, stripe_count, 1);
    int blocksize = stripe_width > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : stripe_width;

    for (int i_off = 0; i_off < stripe_width / blocksize; i_off++)
    {
#ifdef WITH_GPU_STREAMS
        my_pack_c_cuda_kernel<T><<<grid_size, blocksize, 0, my_stream>>>(n_offset, max_idx, stripe_width, a_dim2, l_nev, a_dev, row_group_dev, i_off);
#else
        my_pack_c_cuda_kernel<T><<<grid_size, blocksize>>>(n_offset, max_idx, stripe_width, a_dim2, l_nev, a_dev, row_group_dev, i_off);
#endif
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("\n my_pack_c_cuda_kernel failed %s \n", cudaGetErrorString(err));
}

template <typename T>
void my_unpack_c_cuda(const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    T *row_group_dev, T *a_dev, cudaStream_t my_stream)
{
    if (stripe_width <= 0) return;

    dim3 grid_size = dim3(row_count, stripe_count, 1);
    int blocksize = stripe_width > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : stripe_width;

    for (int i_off = 0; i_off < stripe_width / blocksize; i_off++)
    {
#ifdef WITH_GPU_STREAMS
        my_unpack_c_cuda_kernel<T><<<grid_size, blocksize, 0, my_stream>>>(n_offset, max_idx, stripe_width, a_dim2, l_nev, row_group_dev, a_dev, i_off);
#else
        my_unpack_c_cuda_kernel<T><<<grid_size, blocksize>>>(n_offset, max_idx, stripe_width, a_dim2, l_nev, row_group_dev, a_dev, i_off);
#endif
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("\n my_unpack_c_cuda_kernel failed %s \n", cudaGetErrorString(err));
}

template <typename T>
void extract_hh_tau_c_cuda(T *bcast_buffer_dev, T *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    int grid_size = 1 + (n - 1) / MAX_BLOCK_SIZE;

#ifdef WITH_GPU_STREAMS
    extract_hh_tau_c_cuda_kernel<T><<<grid_size, MAX_BLOCK_SIZE, 0, my_stream>>>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero);
#else
    extract_hh_tau_c_cuda_kernel<T><<<grid_size, MAX_BLOCK_SIZE>>>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero);
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("\n extract_hh_tau_c_cuda_kernel failed %s \n", cudaGetErrorString(err));
}

// ============================================================
// Layer 3: extern "C" wrappers — one per type, Fortran-callable
// ============================================================

extern "C" void launch_my_pack_c_cuda_kernel_real_double(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    double *a_dev, double *row_group_dev, cudaStream_t my_stream)
{
    my_pack_c_cuda<double>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream);
}

extern "C" void launch_my_pack_c_cuda_kernel_real_single(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    float *a_dev, float *row_group_dev, cudaStream_t my_stream)
{
    my_pack_c_cuda<float>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream);
}

extern "C" void launch_my_pack_c_cuda_kernel_complex_double(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    cuDoubleComplex *a_dev, cuDoubleComplex *row_group_dev, cudaStream_t my_stream)
{
    my_pack_c_cuda<cuDoubleComplex>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream);
}

extern "C" void launch_my_pack_c_cuda_kernel_complex_single(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    cuFloatComplex *a_dev, cuFloatComplex *row_group_dev, cudaStream_t my_stream)
{
    my_pack_c_cuda<cuFloatComplex>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void launch_my_pack_c_cuda_kernel_real_half(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    __half *a_dev, __half *row_group_dev, cudaStream_t my_stream)
{
    my_pack_c_cuda<__half>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void launch_my_pack_c_cuda_kernel_complex_half(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    __half2 *a_dev, __half2 *row_group_dev, cudaStream_t my_stream)
{
    my_pack_c_cuda<__half2>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream);
}
#endif

extern "C" void launch_my_unpack_c_cuda_kernel_real_double(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    double *row_group_dev, double *a_dev, cudaStream_t my_stream)
{
    my_unpack_c_cuda<double>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev, my_stream);
}

extern "C" void launch_my_unpack_c_cuda_kernel_real_single(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    float *row_group_dev, float *a_dev, cudaStream_t my_stream)
{
    my_unpack_c_cuda<float>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev, my_stream);
}

extern "C" void launch_my_unpack_c_cuda_kernel_complex_double(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    cuDoubleComplex *row_group_dev, cuDoubleComplex *a_dev, cudaStream_t my_stream)
{
    my_unpack_c_cuda<cuDoubleComplex>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev, my_stream);
}

extern "C" void launch_my_unpack_c_cuda_kernel_complex_single(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    cuFloatComplex *row_group_dev, cuFloatComplex *a_dev, cudaStream_t my_stream)
{
    my_unpack_c_cuda<cuFloatComplex>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void launch_my_unpack_c_cuda_kernel_real_half(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    __half *row_group_dev, __half *a_dev, cudaStream_t my_stream)
{
    my_unpack_c_cuda<__half>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void launch_my_unpack_c_cuda_kernel_complex_half(
    const int row_count, const int n_offset, const int max_idx,
    const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev,
    __half2 *row_group_dev, __half2 *a_dev, cudaStream_t my_stream)
{
    my_unpack_c_cuda<__half2>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev, my_stream);
}
#endif

extern "C" void launch_extract_hh_tau_c_cuda_kernel_real_double(
    double *bcast_buffer_dev, double *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    extract_hh_tau_c_cuda<double>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream);
}

extern "C" void launch_extract_hh_tau_c_cuda_kernel_real_single(
    float *bcast_buffer_dev, float *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    extract_hh_tau_c_cuda<float>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream);
}

extern "C" void launch_extract_hh_tau_c_cuda_kernel_complex_double(
    cuDoubleComplex *bcast_buffer_dev, cuDoubleComplex *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    extract_hh_tau_c_cuda<cuDoubleComplex>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream);
}

extern "C" void launch_extract_hh_tau_c_cuda_kernel_complex_single(
    cuFloatComplex *bcast_buffer_dev, cuFloatComplex *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    extract_hh_tau_c_cuda<cuFloatComplex>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void launch_extract_hh_tau_c_cuda_kernel_real_half(
    __half *bcast_buffer_dev, __half *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    extract_hh_tau_c_cuda<__half>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void launch_extract_hh_tau_c_cuda_kernel_complex_half(
    __half2 *bcast_buffer_dev, __half2 *hh_tau_dev, const int nbw, const int n,
    const int is_zero, cudaStream_t my_stream)
{
    extract_hh_tau_c_cuda<__half2>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream);
}
#endif

extern "C" int cuda_MemcpyDeviceToDevice(int val)
{
    val = cudaMemcpyDeviceToDevice;
    return val;
}
