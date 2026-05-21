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

// Unit test for gpu_set_one_complex<T> from src/elpa1/GPU/tridiag_gpu.h.
// Tests both cuDoubleComplex and cuFloatComplex instantiations.

#include <stdio.h>
#include <stdlib.h>

#include "config-f90.h"

#ifdef WITH_UNIT_TESTS

#ifdef WITH_NVIDIA_GPU_VERSION

#include <stdint.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <type_traits>
#if defined(WANT_HALF_PRECISION_REAL) || defined(WANT_HALF_PRECISION_COMPLEX)
#include <cuda_fp16.h>
#endif

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"
#include "../../../../src/elpa1/GPU/tridiag_gpu.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Returns 0 on pass, 1 on failure.
template <typename T>
static int run_test(const char *type_name, double expected_re, double expected_im)
{
  T *dev_ptr = nullptr;
  T  host_val;
  cudaStream_t stream;

  CUDA_CHECK(cudaMalloc((void **)&dev_ptr, sizeof(T)));
  CUDA_CHECK(cudaMemset(dev_ptr, 0, sizeof(T)));
  CUDA_CHECK(cudaStreamCreate(&stream));

  gpu_set_one_complex<T>(dev_ptr, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(&host_val, dev_ptr, sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dev_ptr));

  double re = (double)host_val.x;
  double im = (double)host_val.y;
  int passed = (re == expected_re) && (im == expected_im);

  printf("  gpu_set_one_complex<%s>: got (%.1f, %.1f)  expected (%.1f, %.1f)  [%s]\n",
         type_name, re, im, expected_re, expected_im,
         passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}

#ifdef WANT_HALF_PRECISION_COMPLEX
static int run_test_half2(void)
{
  __half2 *dev_ptr = nullptr;
  __half2  host_val;
  cudaStream_t stream;

  CUDA_CHECK(cudaMalloc((void **)&dev_ptr, sizeof(__half2)));
  CUDA_CHECK(cudaMemset(dev_ptr, 0, sizeof(__half2)));
  CUDA_CHECK(cudaStreamCreate(&stream));

  gpu_set_one_complex<__half2>(dev_ptr, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(&host_val, dev_ptr, sizeof(__half2), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dev_ptr));

  float re = __half2float(host_val.x);
  float im = __half2float(host_val.y);
  int passed = (re == 1.0f) && (im == 0.0f);

  printf("  gpu_set_one_complex<__half2>: got (%.1f, %.1f)  expected (1.0, 0.0)  [%s]\n",
         re, im, passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
#endif /* WANT_HALF_PRECISION_COMPLEX */

int main(void)
{
  int failures = 0;

  printf("=== Unit tests for gpu_set_one_complex kernel (CUDA) ===\n\n");
  printf("Testing gpu_set_one_complex:\n");
  failures += run_test<cuDoubleComplex>("cuDoubleComplex", 1.0, 0.0);
  failures += run_test<cuFloatComplex> ("cuFloatComplex",  1.0, 0.0);
#ifdef WANT_HALF_PRECISION_COMPLEX
  failures += run_test_half2();
#endif

  printf("\n=== Summary: %d failure(s) ===\n", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
  fprintf(stderr, "Error: this test should only be called when the ELPA build is with NVIDIA GPU version enabled\n");
  abort();
}

#endif /* WITH_NVIDIA_GPU_VERSION */

#endif /* WITH_UNIT_TESTS */
