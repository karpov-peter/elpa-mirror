//    Copyright 2021, A. Marek
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
//    This file was written by A. Marek, MPCDF

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>
#include <cuComplex.h>
#include <stdint.h>
#include <type_traits>
#include "config-f90.h"

#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

// Set a device element to zero; for complex types both components are zeroed.
template <typename T>
__device__ inline void set_to_zero(T &v)
{
    if constexpr (std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuFloatComplex>) {
        v.x = 0;
        v.y = 0;
    } else {
        v = (T)0;
    }
}

#ifdef WANT_HALF_PRECISION_REAL
template <>
__device__ inline void set_to_zero(__half &v) { v = __float2half(0.0f); }
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
template <>
__device__ inline void set_to_zero(__half2 &v) {
    v.x = __float2half(0.0f);
    v.y = __float2half(0.0f);
}
#endif

//_____________________________________________________________________________
// cuda_copy_a_tmat2

template <typename T>
__global__ void cuda_copy_a_tmat2_kernel(T *a_dev, T *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1)
{
  int nb_index    = threadIdx.x + 1; // range 1..nb
  int l_col_index = blockIdx.x  + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index-1) * nblk] =
    a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index-1) * matrixRows];
}

template <typename T>
void cuda_copy_a_tmat2(T *a_dev, T *tmat2_dev, int nblk, int matrixRows, int l_cols, int l_colx, int l_row1, int nb, cudaStream_t my_stream)
{
  dim3 blocks = dim3(l_cols-l_colx+1, 1, 1);
  dim3 threadsPerBlock = dim3(nb, 1, 1);
#ifdef WITH_GPU_STREAMS
  cuda_copy_a_tmat2_kernel<T><<<blocks, threadsPerBlock, 0, my_stream>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
  cuda_copy_a_tmat2_kernel<T><<<blocks, threadsPerBlock>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) printf("Error in cuda_copy_a_tmat2_kernel: %s\n", cudaGetErrorString(cuerr));
}

extern "C" void cuda_copy_double_a_tmat2_FromC(double *a_dev, double *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmat2<double>(a_dev, tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_a_tmat2_FromC(float *a_dev, float *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmat2<float>(a_dev, tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_double_complex_a_tmat2_FromC(double _Complex *a_dev, double _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmat2<cuDoubleComplex>((cuDoubleComplex*)a_dev, (cuDoubleComplex*)tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_complex_a_tmat2_FromC(float _Complex *a_dev, float _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmat2<cuFloatComplex>((cuFloatComplex*)a_dev, (cuFloatComplex*)tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void cuda_copy_half_a_tmat2_FromC(__half *a_dev, __half *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmat2<__half>(a_dev, tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void cuda_copy_half_complex_a_tmat2_FromC(__half2 *a_dev, __half2 *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmat2<__half2>(a_dev, tmat2_dev, *nblk_in, *matrixRows_in, *l_cols_in, *l_colx_in, *l_row1_in, *nb_in, my_stream);
}
#endif

//_____________________________________________________________________________
// cuda_copy_tmp2_tmat2

template <typename T>
__global__ void cuda_copy_tmp2_tmat2_kernel(T *tmp2_dev, T *tmat2_dev, const int nblk, const int l_col1)
{
  int nb_index    = threadIdx.x + 1; // range 1..nb
  int l_col_index = blockIdx.x  + 1; // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index-1) * nblk] =
    tmp2_dev[nb_index-1 + (1-1 + l_col_index-1) * nblk];
}

template <typename T>
void cuda_copy_tmp2_tmat2(T *tmp2_dev, T *tmat2_dev, int nblk, int l_col1, int nb, cudaStream_t my_stream)
{
  dim3 blocks = dim3(nb, 1, 1);
  dim3 threadsPerBlock = dim3(nb, 1, 1);
#ifdef WITH_GPU_STREAMS
  cuda_copy_tmp2_tmat2_kernel<T><<<blocks, threadsPerBlock, 0, my_stream>>>(tmp2_dev, tmat2_dev, nblk, l_col1);
#else
  cuda_copy_tmp2_tmat2_kernel<T><<<blocks, threadsPerBlock>>>(tmp2_dev, tmat2_dev, nblk, l_col1);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) printf("Error in cuda_copy_tmp2_tmat2_kernel: %s\n", cudaGetErrorString(cuerr));
}

extern "C" void cuda_copy_double_tmp2_tmat2_FromC(double *tmp2_dev, double *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp2_tmat2<double>(tmp2_dev, tmat2_dev, *nblk_in, *l_col1_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_tmp2_tmat2_FromC(float *tmp2_dev, float *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp2_tmat2<float>(tmp2_dev, tmat2_dev, *nblk_in, *l_col1_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_double_complex_tmp2_tmat2_FromC(double _Complex *tmp2_dev, double _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp2_tmat2<cuDoubleComplex>((cuDoubleComplex*)tmp2_dev, (cuDoubleComplex*)tmat2_dev, *nblk_in, *l_col1_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_complex_tmp2_tmat2_FromC(float _Complex *tmp2_dev, float _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp2_tmat2<cuFloatComplex>((cuFloatComplex*)tmp2_dev, (cuFloatComplex*)tmat2_dev, *nblk_in, *l_col1_in, *nb_in, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void cuda_copy_half_tmp2_tmat2_FromC(__half *tmp2_dev, __half *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp2_tmat2<__half>(tmp2_dev, tmat2_dev, *nblk_in, *l_col1_in, *nb_in, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void cuda_copy_half_complex_tmp2_tmat2_FromC(__half2 *tmp2_dev, __half2 *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp2_tmat2<__half2>(tmp2_dev, tmat2_dev, *nblk_in, *l_col1_in, *nb_in, my_stream);
}
#endif

//_____________________________________________________________________________
// cuda_copy_a_tmat1
// Copies a block of a_dev into tmat1_dev and zeros the source elements.
// The zeroing uses set_to_zero<T> to handle both real and complex types.

template <typename T>
__global__ void cuda_copy_a_tmat1_kernel(T *a_dev, T *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1)
{
  int nb_index     = threadIdx.x + 1; // range 1..nb
  int l_row1_index = blockIdx.x  + 1; // range 1..l_row1-1

  tmat1_dev[l_row1_index-1 + (nb_index-1) * l_rows] =
    a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1) * matrixRows];
  set_to_zero(a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1) * matrixRows]);
}

template <typename T>
void cuda_copy_a_tmat1(T *a_dev, T *tmat1_dev, int l_rows, int matrixRows, int nb, int l_row1, int l_col1, cudaStream_t my_stream)
{
  dim3 blocks = dim3(l_row1-1, 1, 1);
  dim3 threadsPerBlock = dim3(nb, 1, 1);
#ifdef WITH_GPU_STREAMS
  cuda_copy_a_tmat1_kernel<T><<<blocks, threadsPerBlock, 0, my_stream>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  cuda_copy_a_tmat1_kernel<T><<<blocks, threadsPerBlock>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) printf("Error in cuda_copy_a_tmat1_kernel: %s\n", cudaGetErrorString(cuerr));
}

extern "C" void cuda_copy_double_a_tmat1_FromC(double *a_dev, double *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, cudaStream_t my_stream){
  cuda_copy_a_tmat1<double>(a_dev, tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

extern "C" void cuda_copy_float_a_tmat1_FromC(float *a_dev, float *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, cudaStream_t my_stream){
  cuda_copy_a_tmat1<float>(a_dev, tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

extern "C" void cuda_copy_double_complex_a_tmat1_FromC(double _Complex *a_dev, double _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, cudaStream_t my_stream){
  cuda_copy_a_tmat1<cuDoubleComplex>((cuDoubleComplex*)a_dev, (cuDoubleComplex*)tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

extern "C" void cuda_copy_float_complex_a_tmat1_FromC(float _Complex *a_dev, float _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, cudaStream_t my_stream){
  cuda_copy_a_tmat1<cuFloatComplex>((cuFloatComplex*)a_dev, (cuFloatComplex*)tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void cuda_copy_half_a_tmat1_FromC(__half *a_dev, __half *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, cudaStream_t my_stream){
  cuda_copy_a_tmat1<__half>(a_dev, tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void cuda_copy_half_complex_a_tmat1_FromC(__half2 *a_dev, __half2 *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, cudaStream_t my_stream){
  cuda_copy_a_tmat1<__half2>(a_dev, tmat1_dev, *l_rows_in, *matrixRows_in, *nb_in, *l_row1_in, *l_col1_in, my_stream);
}
#endif

//_____________________________________________________________________________
// cuda_copy_tmp1_tmp2
// Scatter the lower-triangular packed tmp1 into the full column-major tmp2.

template <typename T>
__global__ void cuda_copy_tmp1_tmp2_kernel(T *tmp1_dev, T *tmp2_dev, const int nblk, const int nb)
{
  int i_index = threadIdx.x + 1; // range 1..nb
  int j_index = blockIdx.x  + 1; // range 1..nb (should be 1..i)

  if (j_index < i_index + 1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1) * nblk] =
      tmp1_dev[(i_index*(i_index+1) - 2*i_index) / 2 + 1 - 1 + j_index - 1];
  }
}

template <typename T>
void cuda_copy_tmp1_tmp2(T *tmp1_dev, T *tmp2_dev, int nblk, int nb, cudaStream_t my_stream)
{
  dim3 blocks = dim3(nb, 1, 1);
  dim3 threadsPerBlock = dim3(nb, 1, 1);
#ifdef WITH_GPU_STREAMS
  cuda_copy_tmp1_tmp2_kernel<T><<<blocks, threadsPerBlock, 0, my_stream>>>(tmp1_dev, tmp2_dev, nblk, nb);
#else
  cuda_copy_tmp1_tmp2_kernel<T><<<blocks, threadsPerBlock>>>(tmp1_dev, tmp2_dev, nblk, nb);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) printf("Error in cuda_copy_tmp1_tmp2_kernel: %s\n", cudaGetErrorString(cuerr));
}

extern "C" void cuda_copy_double_tmp1_tmp2_FromC(double *tmp1_dev, double *tmp2_dev, int *nblk_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp1_tmp2<double>(tmp1_dev, tmp2_dev, *nblk_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_tmp1_tmp2_FromC(float *tmp1_dev, float *tmp2_dev, int *nblk_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp1_tmp2<float>(tmp1_dev, tmp2_dev, *nblk_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_double_complex_tmp1_tmp2_FromC(double _Complex *tmp1_dev, double _Complex *tmp2_dev, int *nblk_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp1_tmp2<cuDoubleComplex>((cuDoubleComplex*)tmp1_dev, (cuDoubleComplex*)tmp2_dev, *nblk_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_complex_tmp1_tmp2_FromC(float _Complex *tmp1_dev, float _Complex *tmp2_dev, int *nblk_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp1_tmp2<cuFloatComplex>((cuFloatComplex*)tmp1_dev, (cuFloatComplex*)tmp2_dev, *nblk_in, *nb_in, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void cuda_copy_half_tmp1_tmp2_FromC(__half *tmp1_dev, __half *tmp2_dev, int *nblk_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp1_tmp2<__half>(tmp1_dev, tmp2_dev, *nblk_in, *nb_in, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void cuda_copy_half_complex_tmp1_tmp2_FromC(__half2 *tmp1_dev, __half2 *tmp2_dev, int *nblk_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_tmp1_tmp2<__half2>(tmp1_dev, tmp2_dev, *nblk_in, *nb_in, my_stream);
}
#endif

//_____________________________________________________________________________
// cuda_copy_a_tmp1
// Gather the lower-triangular block from a_dev into packed tmp1_dev.

template <typename T>
__global__ void cuda_copy_a_tmp1_kernel(T *a_dev, T *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb)
{
  int i_index = threadIdx.x + 1; // range 1..nb
  int j_index = blockIdx.x  + 1; // range 1..nb (should be 1..i)

  if (j_index < i_index + 1) {
    tmp1_dev[(i_index*(i_index+1) - 2*i_index) / 2 + 1 - 1 + j_index - 1] =
      a_dev[l_row1-1 + j_index-1 + (l_col1-1 + i_index-1) * matrixRows];
  }
}

template <typename T>
void cuda_copy_a_tmp1(T *a_dev, T *tmp1_dev, int l_row1, int l_col1, int matrixRows, int nb, cudaStream_t my_stream)
{
  dim3 blocks = dim3(nb, 1, 1);
  dim3 threadsPerBlock = dim3(nb, 1, 1);
#ifdef WITH_GPU_STREAMS
  cuda_copy_a_tmp1_kernel<T><<<blocks, threadsPerBlock, 0, my_stream>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  cuda_copy_a_tmp1_kernel<T><<<blocks, threadsPerBlock>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) printf("Error in cuda_copy_a_tmp1_kernel: %s\n", cudaGetErrorString(cuerr));
}

extern "C" void cuda_copy_double_a_tmp1_FromC(double *a_dev, double *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmp1<double>(a_dev, tmp1_dev, *l_row1_in, *l_col1_in, *matrixRows_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_a_tmp1_FromC(float *a_dev, float *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmp1<float>(a_dev, tmp1_dev, *l_row1_in, *l_col1_in, *matrixRows_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_double_complex_a_tmp1_FromC(double _Complex *a_dev, double _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmp1<cuDoubleComplex>((cuDoubleComplex*)a_dev, (cuDoubleComplex*)tmp1_dev, *l_row1_in, *l_col1_in, *matrixRows_in, *nb_in, my_stream);
}

extern "C" void cuda_copy_float_complex_a_tmp1_FromC(float _Complex *a_dev, float _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmp1<cuFloatComplex>((cuFloatComplex*)a_dev, (cuFloatComplex*)tmp1_dev, *l_row1_in, *l_col1_in, *matrixRows_in, *nb_in, my_stream);
}

#ifdef WANT_HALF_PRECISION_REAL
extern "C" void cuda_copy_half_a_tmp1_FromC(__half *a_dev, __half *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmp1<__half>(a_dev, tmp1_dev, *l_row1_in, *l_col1_in, *matrixRows_in, *nb_in, my_stream);
}
#endif

#ifdef WANT_HALF_PRECISION_COMPLEX
extern "C" void cuda_copy_half_complex_a_tmp1_FromC(__half2 *a_dev, __half2 *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, cudaStream_t my_stream){
  cuda_copy_a_tmp1<__half2>(a_dev, tmp1_dev, *l_row1_in, *l_col1_in, *matrixRows_in, *nb_in, my_stream);
}
#endif
