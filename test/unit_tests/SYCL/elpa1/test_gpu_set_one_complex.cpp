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

#include "config-f90.h"

#ifdef WITH_UNIT_TESTS

#ifdef WITH_SYCL_GPU_VERSION

#include <stdint.h>
#include <complex>
#include <sycl/sycl.hpp>

#include "../../../../src/GPU/common_device_functions.h"
#include "../../../../src/GPU/gpu_to_cuda_and_hip_interface.h"

// Self-contained kernel (no getQueueOrDefault / syclCommon.hpp dependency).
template <typename T>
static void gpu_set_one_complex_kernel(T *a_dev) {
  using S = typename std::conditional<std::is_same<T, gpuFloatComplex>::value,
                                      float, double>::type;
  *a_dev = elpaDeviceNumberFromRealImag<T>(S(1.0), S(0.0));
}

// Returns 0 on pass, 1 on failure.
template <typename T>
static int run_test(sycl::queue &q, const char *type_name)
{
  T *dev_ptr = sycl::malloc_device<T>(1, q);
  T  zero_val{};
  q.memcpy(dev_ptr, &zero_val, sizeof(T)).wait();

  q.single_task([=]() {
    gpu_set_one_complex_kernel(dev_ptr);
  }).wait();

  T host_val{};
  q.memcpy(&host_val, dev_ptr, sizeof(T)).wait();
  sycl::free(dev_ptr, q);

  double re = (double)host_val.real();
  double im = (double)host_val.imag();
  int passed = (re == 1.0) && (im == 0.0);

  printf("  gpu_set_one_complex<%s>: got (%.1f, %.1f)  expected (1.0, 0.0)  [%s]\n",
         type_name, re, im, passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}

int main(void)
{
  sycl::queue q(sycl::default_selector_v);
  int failures = 0;

  printf("Testing gpu_set_one_complex (SYCL):\n");
  failures += run_test<gpuDoubleComplex>(q, "gpuDoubleComplex");
  failures += run_test<gpuFloatComplex> (q, "gpuFloatComplex");

  if (failures == 0)
    printf("All tests passed.\n");
  else
    printf("%d test(s) FAILED.\n", failures);

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else

int main(void)
{
  fprintf(stderr, "Error: this test should only be called when the ELPA build is with SYCL GPU version enabled\n");
  abort();
}

#endif /* WITH_SYCL_GPU_VERSION */

#endif /* WITH_UNIT_TESTS */
