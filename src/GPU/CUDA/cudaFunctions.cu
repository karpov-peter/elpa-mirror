//
//    Copyright 2014, A. Marek
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
//
// --------------------------------------------------------------------------------------------------
//
// This file was written by A. Marek, MPCDF
#include "config-f90.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <stddef.h>
#include <complex.h>
#include <cublas_v2.h>


#ifdef WITH_NVIDIA_CUSOLVER
#include <cusolverDn.h>
#endif

#undef BLAS_status
#undef BLAS_handle
//#undef BLAS_float_complex
#undef BLAS_set_stream
#undef BLAS_status_success
#undef BLAS_status_invalid_handle
#undef BLAS_create_handle
#undef BLAS_destroy_handle
#undef BLAS_double_complex
#undef BLAS_float_complex
#undef BLAS_strsm
#undef BLAS_dtrsm
#undef BLAS_ctrsm
#undef BLAS_ztrsm
#undef BLAS_dtrmm
#undef BLAS_strmm
#undef BLAS_ztrmm
#undef BLAS_ctrmm
#undef BLAS_dcopy
#undef BLAS_scopy
#undef BLAS_zcopy
#undef BLAS_ccopy
#undef BLAS_dgemm
#undef BLAS_sgemm
#undef BLAS_zgemm
#undef BLAS_cgemm
#undef BLAS_dgemv
#undef BLAS_sgemv
#undef BLAS_zgemv
#undef BLAS_cgemv
#undef BLAS_operation
#undef BLAS_operation_none
#undef BLAS_operation_transpose
#undef BLAS_operation_conjugate_transpose
#undef BLAS_operation_none
#undef BLAS_fill
#undef BLAS_fill_lower
#undef BLAS_fill_upper
#undef BLAS_side
#undef BLAS_side_left
#undef BLAS_side_right
#undef BLAS_diagonal
#undef BLAS_diagonal_non_unit
#undef BLAS_diagonal_unit
#undef SOLVER_HANLDE
#undef BLAS_status_not_initialized

#define BLAS cublas
#define BLAS_status cublasStatus_t
#define BLAS_handle cublasHandle_t
#define SOLVER_handle cusolverDnHandle_t
#define BLAS_set_stream cublasSetStream
#define BLAS_status_success CUBLAS_STATUS_SUCCESS
#define BLAS_status_not_initialized CUBLAS_STATUS_NOT_INITIALIZED
#define BLAS_status_invalid_handle CUBLAS_STATUS_INVALID_VALUE
#define BLAS_create_handle cublasCreate
#define BLAS_destroy_handle cublasDestroy
#define BLAS_double_complex cuDoubleComplex
#define BLAS_float_complex cuFloatComplex
#define BLAS_ctrsm cublasCtrsm
#define BLAS_ztrsm cublasZtrsm
#define BLAS_dtrsm cublasDtrsm
#define BLAS_strsm cublasStrsm
#define BLAS_ctrmm cublasCtrmm
#define BLAS_ztrmm cublasZtrmm
#define BLAS_dtrmm cublasDtrmm
#define BLAS_strmm cublasStrmm
#define BLAS_ccopy cublasCcopy
#define BLAS_zcopy cublasZcopy
#define BLAS_dcopy cublasDcopy
#define BLAS_scopy cublasScopy
#define BLAS_cgemm cublasCgemm
#define BLAS_zgemm cublasZgemm
#define BLAS_dgemm cublasDgemm
#define BLAS_sgemm cublasSgemm
#define BLAS_cgemv cublasCgemv
#define BLAS_zgemv cublasZgemv
#define BLAS_dgemv cublasDgemv
#define BLAS_sgemv cublasSgemv
#define BLAS_operation cublasOperation_t
#define BLAS_operation_none CUBLAS_OP_N
#define BLAS_operation_transpose CUBLAS_OP_T
#define BLAS_operation_conjugate_transpose CUBLAS_OP_C
#define BLAS_fill cublasFillMode_t
#define BLAS_fill_lower CUBLAS_FILL_MODE_LOWER
#define BLAS_fill_upper CUBLAS_FILL_MODE_UPPER
#define BLAS_side cublasSideMode_t
#define BLAS_side_left CUBLAS_SIDE_LEFT
#define BLAS_side_right CUBLAS_SIDE_RIGHT
#define BLAS_diagonal cublasDiagType_t
#define BLAS_diagonal_non_unit CUBLAS_DIAG_NON_UNIT
#define BLAS_diagonal_unit CUBLAS_DIAG_UNIT





#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif


#ifdef WITH_NVIDIA_GPU_VERSION
extern "C" {
  int cudaStreamCreateFromC(cudaStream_t *cudaStream) {
    //*stream = (intptr_t) malloc(sizeof(cudaStream_t));

    if (sizeof(intptr_t) != sizeof(cudaStream_t)) {
      printf("Stream sizes do not match \n");
    }

    cudaError_t status = cudaStreamCreate(cudaStream);

    if (status == cudaSuccess) {
//       printf("all OK\n");
      return 1;
    }
    else{
      errormessage("Error in cudaStreamCreate: %s\n", "unknown error");
      return 0;
    }

  }

  int cudaStreamDestroyFromC(cudaStream_t cudaStream){
    cudaError_t status = cudaStreamDestroy(cudaStream);
    if (status == cudaSuccess) {
//       printf("all OK\n");
	 //free((void*) *stream);
      return 1;
    }
    else{
      errormessage("Error in cudaStreamDestroy: %s\n", "unknown error");
      return 0;
    }
  }

  int cudaStreamSynchronizeExplicitFromC(cudaStream_t cudaStream) {
    cudaError_t status = cudaStreamSynchronize(cudaStream);
    if (status == cudaSuccess) {
      return 1;
    }
    else{
      errormessage("Error in cudaStreamSynchronizeExplicit: %s\n", "unknown error");
      return 0;
    }
  }

  int cudaStreamSynchronizeImplicitFromC() {
    cudaError_t status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status == cudaSuccess) {
      return 1;
    }
    else{
      errormessage("Error in cudaStreamSynchronizeImplicit: %s\n", "unknown error");
      return 0;
    }
  }

  int BLAS_set_streamFromC(BLAS_handle cudaHandle, cudaStream_t cudaStream) {
    //BLAS_status status = BLAS_set_stream(*((BLAS_handle*)handle), *((cudaStream_t*)stream));
    BLAS_status status = BLAS_set_stream(cudaHandle, cudaStream);
    if (status == BLAS_status_success) {
      return 1;
    }
    else if (status == BLAS_status_not_initialized) {
      errormessage("Error in BLAS_set_stream: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else{
      errormessage("Error in BLAS_set_stream: %s\n", "unknown error");
      return 0;
    }
  }


#ifdef WITH_NVIDIA_CUSOLVER
  int cusolverSetStreamFromC(cusolverDnHandle_t cusolver_handle, cudaStream_t cudaStream) {
    //cusolverStatus_t status = cusolverDnSetStream(*((cusolverDnHandle_t*)cusolver_handle), *((cudaStream_t*)stream));
    cusolverStatus_t status = cusolverDnSetStream(cusolver_handle, cudaStream);
    if (status == CUSOLVER_STATUS_SUCCESS) {
      return 1;
    }
    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cusolverDnSetStream: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else{
      errormessage("Error in cusolverDnSetStream: %s\n", "unknown error");
      return 0;
    }
  }
#endif

  int cudaMemcpy2dAsyncFromC(intptr_t *dest, size_t dpitch, intptr_t *src, size_t spitch, size_t width, size_t height, int dir, cudaStream_t cudaStream) {
  
    cudaError_t cuerr = cudaMemcpy2DAsync( dest, dpitch, src, spitch, width, height, (cudaMemcpyKind)dir, cudaStream );
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpy2dAsync: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int BLAS_create_handleFromC(BLAS_handle *cublas_handle) {
    //*cublas_handle = (intptr_t) malloc(sizeof(BLAS_handle));
    if (sizeof(intptr_t) != sizeof(BLAS_handle)) {
      //errormessage("Error in BLAS_create_handle: sizes not the same");
	printf("ERROR on sizes\n");
      return 0;
    }
    BLAS_status status = BLAS_create_handle(cublas_handle);
    if (status == BLAS_status_success) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == BLAS_status_not_initialized) {
      errormessage("Error in BLAS_create_handle: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else if (status == CUBLAS_STATUS_ALLOC_FAILED) {
      errormessage("Error in BLAS_create_handle: %s\n", "the resources could not be allocated");
      return 0;
    }
    else{
      errormessage("Error in BLAS_create_handle: %s\n", "unknown error");
      return 0;
    }
  }

  int BLAS_destroy_handleFromC(BLAS_handle cublas_handle) {
    BLAS_status status = BLAS_destroy_handle(cublas_handle);
    if (status == BLAS_status_success) {
//	 free((void*) *cublas_handle);
//       printf("all OK\n");
      return 1;
    }
    else if (status == BLAS_status_not_initialized) {
      errormessage("Error in BLAS_destroy_handle: %s\n", "the library has not been initialized");
      return 0;
    }
    else{
      errormessage("Error in BLAS_destroy_handle: %s\n", "unknown error");
      return 0;
    }
  }

#ifdef WITH_NVIDIA_CUSOLVER

  int cusolverCreateFromC(cusolverDnHandle_t *cusolver_handle) {
    //*cusolver_handle = (intptr_t) malloc(sizeof(cusolverDnHandle_t));
    //cusolverStatus_t status = cusolverDnCreate((cusolverDnHandle_t*) *cusolver_handle);
    if (sizeof(intptr_t) != sizeof(cusolverDnHandle_t)) {
      printf("cusolver sizes wrong\n");
    }
    cusolverStatus_t status = cusolverDnCreate(cusolver_handle);
    if (status == CUSOLVER_STATUS_SUCCESS) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cusolverCreate: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else if (status == CUSOLVER_STATUS_ALLOC_FAILED) {
      errormessage("Error in cusolverCreate: %s\n", "the resources could not be allocated");
      return 0;
    }
    else{
      errormessage("Error in cusolverCreate: %s\n", "unknown error");
      return 0;
    }
  }

  int cusolverDestroyFromC(cusolverDnHandle_t cusolver_handle) {
    //cusolverStatus_t status = cusolverDnDestroy(*((cusolverDnHandle_t*) *cusolver_handle));
    cusolverStatus_t status = cusolverDnDestroy(cusolver_handle);
    if (status == CUSOLVER_STATUS_SUCCESS) {
//       printf("all OK\n");
      //free((void*) *cusolver_handle);
      return 1;
    }
    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cusolverDestroy: %s\n", "the library has not been initialized");
      return 0;
    }
    else{
      errormessage("Error in cusolverDestroy: %s\n", "unknown error");
      return 0;
    }
  }
#endif /* WITH_NVIDIA_CUSOLVER */

  int cudaSetDeviceFromC(int n) {

    cudaError_t cuerr = cudaSetDevice(n);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaSetDevice: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaGetDeviceCountFromC(int *count) {

    cudaError_t cuerr = cudaGetDeviceCount(count);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaGetDeviceCount: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaDeviceSynchronizeFromC() {

    cudaError_t cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMallocFromC(intptr_t *a, size_t width_height) {

    cudaError_t cuerr = cudaMalloc((void **) a, width_height);
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *a, width_height);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMalloc: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaFreeFromC(intptr_t *a) {
#ifdef DEBUG_CUDA
    printf("CUDA Free, pointer address: %p \n", a);
#endif
    cudaError_t cuerr = cudaFree(a);

    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaFree: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMallocHostFromC(intptr_t *a, size_t width_height) {

    cudaError_t cuerr = cudaMallocHost((void **) a, width_height);
#ifdef DEBUG_CUDA
    printf("MallocHost pointer address: %p \n", *a);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMallocHost: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaFreeHostFromC(intptr_t *a) {
#ifdef DEBUG_CUDA
    printf("FreeHost pointer address: %p \n", a);
#endif
    cudaError_t cuerr = cudaFreeHost(a);

    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaFreeHost: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemsetFromC(intptr_t *a, int value, size_t count) {

    cudaError_t cuerr = cudaMemset( a, value, count);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemset: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemsetAsyncFromC(intptr_t *a, int value, size_t count, cudaStream_t cudaStream) {

    cudaError_t cuerr = cudaMemsetAsync( a, value, count, cudaStream);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemsetAsync: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemcpyFromC(intptr_t *dest, intptr_t *src, size_t count, int dir) {

    cudaError_t cuerr = cudaMemcpy( dest, src, count, (cudaMemcpyKind)dir);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpy: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemcpyAsyncFromC(intptr_t *dest, intptr_t *src, size_t count, int dir, cudaStream_t cudaStream) {

    cudaError_t cuerr = cudaMemcpyAsync( dest, src, count, (cudaMemcpyKind)dir, cudaStream);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpyAsync: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemcpy2dFromC(intptr_t *dest, size_t dpitch, intptr_t *src, size_t spitch, size_t width, size_t height, int dir) {
  
    cudaError_t cuerr = cudaMemcpy2D( dest, dpitch, src, spitch, width, height, (cudaMemcpyKind)dir);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpy2d: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaHostRegisterFromC(intptr_t *a, intptr_t value, int flag) {

    cudaError_t cuerr = cudaHostRegister( a, value, flag);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaHostRegister: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaHostUnregisterFromC(intptr_t *a) {

    cudaError_t cuerr = cudaHostUnregister( a);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaHostUnregister: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemcpyDeviceToDeviceFromC(void) {
      int val = cudaMemcpyDeviceToDevice;
      return val;
  }
  int cudaMemcpyHostToDeviceFromC(void) {
      int val = cudaMemcpyHostToDevice;
      return val;
  }
  int cudaMemcpyDeviceToHostFromC(void) {
      int val = cudaMemcpyDeviceToHost;
      return val;
  }
  int cudaHostRegisterDefaultFromC(void) {
      int val = cudaHostRegisterDefault;
      return val;
  }
  int cudaHostRegisterPortableFromC(void) {
      int val = cudaHostRegisterPortable;
      return val;
  }
  int cudaHostRegisterMappedFromC(void) {
      int val = cudaHostRegisterMapped;
      return val;
  }

  BLAS_operation operation_new_api(char trans) {
    if (trans == 'N' || trans == 'n') {
      return BLAS_operation_none;
    }
    else if (trans == 'T' || trans == 't') {
      return BLAS_operation_transpose;
    }
    else if (trans == 'C' || trans == 'c') {
      return BLAS_operation_conjugate_transpose;
    }
    else {
      errormessage("Error when transfering %c to BLAS_operation\n",trans);
      // or abort?
      return BLAS_operation_none;
    }
  }


  BLAS_fill fill_mode_new_api(char uplo) {
    if (uplo == 'L' || uplo == 'l') {
      return BLAS_fill_lower;
    }
    else if(uplo == 'U' || uplo == 'u') {
      return BLAS_fill_upper;
    }
    else {
      errormessage("Error when transfering %c to BLAS_fill\n", uplo);
      // or abort?
      return BLAS_fill_lower;
    }
  }

  BLAS_side side_mode_new_api(char side) {
    if (side == 'L' || side == 'l') {
      return BLAS_side_left;
    }
    else if (side == 'R' || side == 'r') {
      return BLAS_side_right;
    }
    else{
      errormessage("Error when transfering %c to BLAS_side\n", side);
      // or abort?
      return BLAS_side_left;
    }
  }

  BLAS_diagonal diag_type_new_api(char diag) {
    if (diag == 'N' || diag == 'n') {
      return BLAS_diagonal_non_unit;
    }
    else if (diag == 'U' || diag == 'u') {
      return BLAS_diagonal_unit;
    }
    else {
      errormessage("Error when transfering %c to cublasDiagMode_t\n", diag);
      // or abort?
      return BLAS_diagonal_non_unit;
    }
  }

#ifdef WITH_NVIDIA_CUSOLVER
  void cusolverDtrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, double *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    double *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;
    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnDtrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Dtrtri host work array needed of size=: %d\n",h_lwork);
    }

    //cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverStrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, float *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }

    float *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;

    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnStrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Strtri host work array needed of size=: %d\n",h_lwork);
    }

    //cuerr = cudaMalloc((void**) &d_work, sizeof(float) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri d_work: %s\n",cudaGetErrorString(cuerr));
    }

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Strtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverZtrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, double _Complex *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }

    //BLAS_double_complex A_casted = *((BLAS_double_complex*)(A));
    double _Complex *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;

    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnZtrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Ztrtri host work array needed of size=: %d\n",h_lwork);
    }

    //cuerr = cudaMalloc((void**) &d_work, sizeof(double _Complex) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork in bytes
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri d_work: %s\n",cudaGetErrorString(cuerr));
    }

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Ztrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverCtrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, float _Complex *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }

    //BLAS_float_complex A_casted = *((BLAS_float_complex*)(A));
    float _Complex *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;

    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnCtrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Ctrtri host work array needed of size=: %d\n",h_lwork);
    }

    //cuerr = cudaMalloc((void**) &d_work, sizeof(float _Complex) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri d_work: %s\n",cudaGetErrorString(cuerr));
    }

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Ctrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }


  void cusolverDpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, double *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    double *d_work = NULL;
    int d_lwork = 0;

    //status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    status = cusolverDnDpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnDpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);
    status = cusolverDnDpotrf(cudaHandle, fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverSpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, float *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    float *d_work = NULL;
    int d_lwork = 0;

    //status = cusolverDnSpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    status = cusolverDnSpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnSpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(float) * d_lwork);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnSpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);
    status = cusolverDnSpotrf(cudaHandle, fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverZpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, double _Complex *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    BLAS_double_complex *d_work = NULL;
    int d_lwork = 0;
    BLAS_double_complex* A_casted = (BLAS_double_complex*) A;

    //status = cusolverDnZpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    status = cusolverDnZpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnZpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(BLAS_double_complex) * d_lwork);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnZpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);
    status = cusolverDnZpotrf(cudaHandle, fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverCpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, float _Complex *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    BLAS_float_complex *d_work = NULL;
    int d_lwork = 0;
    BLAS_float_complex* A_casted = (BLAS_float_complex*) A;

    //status = cusolverDnCpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    status = cusolverDnCpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnCpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(BLAS_float_complex) * d_lwork);
    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork is already in bytes
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnCpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);
    status = cusolverDnCpotrf(cudaHandle, fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }
#endif /* WITH_NVIDIA_CUSOLVER */


  void BLAS_dgemv_elpa_wrapper (BLAS_handle cudaHandle, char trans, int m, int n, double alpha,
                               const double *A, int lda,  const double *x, int incx,
                               double beta, double *y, int incy) {

    //BLAS_status status = BLAS_dgemv(*((BLAS_handle*)handle), operation_new_api(trans),
    BLAS_status status = BLAS_dgemv(cudaHandle, operation_new_api(trans),
                                        m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_dgemv\n");
    }

  }

  void BLAS_sgemv_elpa_wrapper (BLAS_handle cudaHandle, char trans, int m, int n, float alpha,
                               const float *A, int lda,  const float *x, int incx,
                               float beta, float *y, int incy) {

    //BLAS_status status = BLAS_sgemv(*((BLAS_handle*)handle), operation_new_api(trans),
    BLAS_status status = BLAS_sgemv(cudaHandle, operation_new_api(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_sgemv\n");
    }
  }

  void BLAS_zgemv_elpa_wrapper (BLAS_handle cudaHandle, char trans, int m, int n, double _Complex alpha,
                               const double _Complex *A, int lda,  const double _Complex *x, int incx,
                               double _Complex beta, double _Complex *y, int incy) {

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));
    BLAS_double_complex beta_casted = *((BLAS_double_complex*)(&beta));

    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    const BLAS_double_complex* x_casted = (const BLAS_double_complex*) x;
    BLAS_double_complex* y_casted = (BLAS_double_complex*) y;

    //BLAS_status status = BLAS_zgemv(*((BLAS_handle*)handle), operation_new_api(trans),
    BLAS_status status = BLAS_zgemv(cudaHandle, operation_new_api(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_zgemv\n");
    }
  }

  void BLAS_cgemv_elpa_wrapper (BLAS_handle cudaHandle, char trans, int m, int n, float _Complex alpha,
                               const float _Complex *A, int lda,  const float _Complex *x, int incx,
                               float _Complex beta, float _Complex *y, int incy) {

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));
    BLAS_float_complex beta_casted = *((BLAS_float_complex*)(&beta));

    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    const BLAS_float_complex* x_casted = (const BLAS_float_complex*) x;
    BLAS_float_complex* y_casted = (BLAS_float_complex*) y;

    //BLAS_status status = BLAS_cgemv(*((BLAS_handle*)handle), operation_new_api(trans),
    BLAS_status status = BLAS_cgemv(cudaHandle, operation_new_api(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_cgemv\n");
    }
  }


  void BLAS_dgemm_elpa_wrapper (BLAS_handle cudaHandle, char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda,
                               const double *B, int ldb, double beta,
                               double *C, int ldc) {

    //BLAS_status status = BLAS_dgemm(*((BLAS_handle*)handle), operation_new_api(transa), operation_new_api(transb),
    BLAS_status status = BLAS_dgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_dgemm\n");
    }
  }

  void BLAS_sgemm_elpa_wrapper (BLAS_handle cudaHandle, char transa, char transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {

    //BLAS_status status = BLAS_sgemm(((BLAS_handle*)handle), operation_new_api(transa), operation_new_api(transb),
    BLAS_status status = BLAS_sgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_sgemm\n");
    }
  }

  void BLAS_zgemm_elpa_wrapper (BLAS_handle cudaHandle, char transa, char transb, int m, int n, int k,
                               double _Complex alpha, const double _Complex *A, int lda,
                               const double _Complex *B, int ldb, double _Complex beta,
                               double _Complex *C, int ldc) {

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));
    BLAS_double_complex beta_casted = *((BLAS_double_complex*)(&beta));

    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    const BLAS_double_complex* B_casted = (const BLAS_double_complex*) B;
    BLAS_double_complex* C_casted = (BLAS_double_complex*) C;

    //BLAS_status status = BLAS_zgemm(*((BLAS_handle*)handle), operation_new_api(transa), operation_new_api(transb),
    BLAS_status status = BLAS_zgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_zgemm\n");
    }
  }

  void BLAS_cgemm_elpa_wrapper (BLAS_handle cudaHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));
    BLAS_float_complex beta_casted = *((BLAS_float_complex*)(&beta));

    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    const BLAS_float_complex* B_casted = (const BLAS_float_complex*) B;
    BLAS_float_complex* C_casted = (BLAS_float_complex*) C;

    //BLAS_status status =  BLAS_cgemm(*((BLAS_handle*)handle), operation_new_api(transa), operation_new_api(transb),
    BLAS_status status =  BLAS_cgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_cgemm\n");
    }
  }


  // todo: new CUBLAS API diverged from standard BLAS api for these functions
  // todo: it provides out-of-place (and apparently more efficient) implementation
  // todo: by passing B twice (in place of C as well), we should fall back to in-place algorithm


  void BLAS_dcopy_elpa_wrapper (BLAS_handle cudaHandle, int n, double *x, int incx, double *y, int incy){

    //BLAS_status status = BLAS_dcopy(*((BLAS_handle*)handle), n, x, incx, y, incy);
    BLAS_status status = BLAS_dcopy(cudaHandle, n, x, incx, y, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_dcopy\n");
    }
  }

  void BLAS_scopy_elpa_wrapper (BLAS_handle cudaHandle, int n, float *x, int incx, float *y, int incy){

    //BLAS_status status = BLAS_scopy(*((BLAS_handle*)handle), n, x, incx, y, incy);
    BLAS_status status = BLAS_scopy(cudaHandle, n, x, incx, y, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_scopy\n");
    }
  }

  void cublasZcopy_elpa_wrapper (BLAS_handle cudaHandle, int n, double _Complex *x, int incx, double _Complex *y, int incy){
    const BLAS_double_complex* X_casted = (const BLAS_double_complex*) x;
          BLAS_double_complex* Y_casted = (      BLAS_double_complex*) y;

    //BLAS_status status = cublasZcopy(*((BLAS_handle*)handle), n, X_casted, incx, Y_casted, incy);
    BLAS_status status = cublasZcopy(cudaHandle, n, X_casted, incx, Y_casted, incy);
    if (status != BLAS_status_success) {
       printf("error when calling cublasZcopy\n");
    }
  }

  void BLAS_ccopy_elpa_wrapper (BLAS_handle cudaHandle, int n, float _Complex *x, int incx, float _Complex *y, int incy){
    const BLAS_float_complex* X_casted = (const BLAS_float_complex*) x;
          BLAS_float_complex* Y_casted = (      BLAS_float_complex*) y;

    //BLAS_status status = BLAS_ccopy(handle, n, X_casted, incx, Y_casted, incy);
    BLAS_status status = BLAS_ccopy(cudaHandle, n, X_casted, incx, Y_casted, incy);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_ccopy\n");
    }
  }

  void BLAS_dtrsm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){

    //BLAS_status status = BLAS_dtrsm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_dtrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                                        diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_dtrsm\n");
    }
  }

  void BLAS_strsm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){

    //BLAS_status status = BLAS_strsm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_strsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                                        diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_strsm\n");
    }
  }

  void BLAS_ztrsm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    BLAS_double_complex* B_casted = (BLAS_double_complex*) B;

    //BLAS_status status = BLAS_ztrsm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_ztrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_ztrsm\n");
    }
  }

  void BLAS_ctrsm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));

    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    BLAS_float_complex* B_casted = (BLAS_float_complex*) B;

    //BLAS_status status = BLAS_ctrsm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_ctrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_ctrsm\n");
    }
  }


  void BLAS_dtrmm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){

    //BLAS_status status = BLAS_dtrmm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_dtrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_dtrmm\n");
    }
  }

  void BLAS_strmm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){

    //BLAS_status status = BLAS_strmm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_strmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_strmm\n");
    }
  }

  void BLAS_ztrmm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    BLAS_double_complex* B_casted = (BLAS_double_complex*) B;

    //BLAS_status status = BLAS_ztrmm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_ztrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_ztrmm\n");
    }
  }

  void BLAS_ctrmm_elpa_wrapper (BLAS_handle cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));

    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    BLAS_float_complex* B_casted = (BLAS_float_complex*) B;

    //BLAS_status status = BLAS_ctrmm(*((BLAS_handle*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    BLAS_status status = BLAS_ctrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
    if (status != BLAS_status_success) {
       printf("error when calling BLAS_ctrmm\n");
    }
  }


}
#endif /* WITH_NVIDIA_GPU_VERSION */
