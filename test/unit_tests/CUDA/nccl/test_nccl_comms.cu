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

#include "config-f90.h"

#ifdef WITH_UNIT_TESTS

#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_NVIDIA_NCCL

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "nccl.h"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                             \
              __FILE__, __LINE__, cudaGetErrorString(_e));                      \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                           \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t _r = (call);                                                   \
    if (_r != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL error at %s:%d: %s\n",                             \
              __FILE__, __LINE__, ncclGetErrorString(_r));                      \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                           \
  } while (0)

// Number of scalar elements per test buffer
static const int N = 4;

// -----------------------------------------------------------------------
// Allreduce (sum) of N doubles.
// Each rank contributes 1.0; expected result is 2.0 on both ranks.
// -----------------------------------------------------------------------
static int test_allreduce_double(ncclComm_t comm, cudaStream_t stream, int rank)
{
  double *d_send, *d_recv;
  double  h_send[N], h_recv[N];
  for (int i = 0; i < N; i++) h_send[i] = 1.0;

  CUDA_CHECK(cudaMalloc(&d_send, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_recv, N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_send, h_send, N * sizeof(double), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclAllReduce(d_send, d_recv, N, ncclDouble, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, N * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_recv[i] == 2.0);
  if (rank == 0)
    printf("  AllReduce double      (N=%d, sum): got %.1f  expected 2.0  [%s]\n",
           N, h_recv[0], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Allreduce (sum) of N floats.
// -----------------------------------------------------------------------
static int test_allreduce_float(ncclComm_t comm, cudaStream_t stream, int rank)
{
  float *d_send, *d_recv;
  float  h_send[N], h_recv[N];
  for (int i = 0; i < N; i++) h_send[i] = 1.0f;

  CUDA_CHECK(cudaMalloc(&d_send, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_recv, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_send, h_send, N * sizeof(float), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclAllReduce(d_send, d_recv, N, ncclFloat, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_recv[i] == 2.0f);
  if (rank == 0)
    printf("  AllReduce float       (N=%d, sum): got %.1f  expected 2.0  [%s]\n",
           N, (double)h_recv[0], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Allreduce (sum) of N half-precision reals.
// -----------------------------------------------------------------------
static int test_allreduce_half(ncclComm_t comm, cudaStream_t stream, int rank)
{
  __half *d_send, *d_recv;
  __half  h_send[N], h_recv[N];
  for (int i = 0; i < N; i++) h_send[i] = __float2half(1.0f);

  CUDA_CHECK(cudaMalloc(&d_send, N * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_recv, N * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_send, h_send, N * sizeof(__half), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclAllReduce(d_send, d_recv, N, ncclFloat16, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, N * sizeof(__half), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (__half2float(h_recv[i]) == 2.0f);
  if (rank == 0)
    printf("  AllReduce half        (N=%d, sum): got %.1f  expected 2.0  [%s]\n",
           N, (double)__half2float(h_recv[0]), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Allreduce (sum) of N double-complex numbers, sent as 2*N ncclDouble.
// Each element: real=1.0, imag=0.5 -> expected real=2.0, imag=1.0.
// -----------------------------------------------------------------------
static int test_allreduce_double_complex(ncclComm_t comm, cudaStream_t stream, int rank)
{
  double *d_send, *d_recv;
  double  h_send[2 * N], h_recv[2 * N];
  for (int i = 0; i < N; i++) { h_send[2*i] = 1.0; h_send[2*i+1] = 0.5; }

  CUDA_CHECK(cudaMalloc(&d_send, 2 * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_recv, 2 * N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_send, h_send, 2 * N * sizeof(double), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclAllReduce(d_send, d_recv, 2 * N, ncclDouble, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, 2 * N * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_recv[2*i] == 2.0) && (h_recv[2*i+1] == 1.0);
  if (rank == 0)
    printf("  AllReduce dbl-complex (N=%d, sum): got (%.1f,%.1f)  expected (2.0,1.0)  [%s]\n",
           N, h_recv[0], h_recv[1], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Allreduce (sum) of N float-complex numbers, sent as 2*N ncclFloat.
// Each element: real=1.0, imag=0.5 -> expected real=2.0, imag=1.0.
// -----------------------------------------------------------------------
static int test_allreduce_float_complex(ncclComm_t comm, cudaStream_t stream, int rank)
{
  float *d_send, *d_recv;
  float  h_send[2 * N], h_recv[2 * N];
  for (int i = 0; i < N; i++) { h_send[2*i] = 1.0f; h_send[2*i+1] = 0.5f; }

  CUDA_CHECK(cudaMalloc(&d_send, 2 * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_recv, 2 * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_send, h_send, 2 * N * sizeof(float), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclAllReduce(d_send, d_recv, 2 * N, ncclFloat, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, 2 * N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_recv[2*i] == 2.0f) && (h_recv[2*i+1] == 1.0f);
  if (rank == 0)
    printf("  AllReduce flt-complex (N=%d, sum): got (%.1f,%.1f)  expected (2.0,1.0)  [%s]\n",
           N, (double)h_recv[0], (double)h_recv[1], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Allreduce (sum) of N half-complex numbers, sent as 2*N ncclFloat16.
// Each element: real=1.0, imag=0.5 -> expected real=2.0, imag=1.0.
// -----------------------------------------------------------------------
static int test_allreduce_half_complex(ncclComm_t comm, cudaStream_t stream, int rank)
{
  __half *d_send, *d_recv;
  __half  h_send[2 * N], h_recv[2 * N];
  for (int i = 0; i < N; i++) {
    h_send[2*i]   = __float2half(1.0f);
    h_send[2*i+1] = __float2half(0.5f);
  }

  CUDA_CHECK(cudaMalloc(&d_send, 2 * N * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_recv, 2 * N * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_send, h_send, 2 * N * sizeof(__half), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclAllReduce(d_send, d_recv, 2 * N, ncclFloat16, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, 2 * N * sizeof(__half), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));

  int ok = 1;
  for (int i = 0; i < N; i++)
    ok &= (__half2float(h_recv[2*i]) == 2.0f) && (__half2float(h_recv[2*i+1]) == 1.0f);
  if (rank == 0)
    printf("  AllReduce hlf-complex (N=%d, sum): got (%.1f,%.1f)  expected (2.0,1.0)  [%s]\n",
           N, (double)__half2float(h_recv[0]), (double)__half2float(h_recv[1]),
           ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Bcast of N doubles from root=0.
// Rank 0: 5.0; Rank 1: 0.0 -> expected 5.0 on both.
// -----------------------------------------------------------------------
static int test_bcast_double(ncclComm_t comm, cudaStream_t stream, int rank)
{
  double *d_buf;
  double  h_buf[N];
  for (int i = 0; i < N; i++) h_buf[i] = (rank == 0) ? 5.0 : 0.0;

  CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, N * sizeof(double), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclBroadcast(d_buf, d_buf, N, ncclDouble, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_buf[i] == 5.0);
  if (rank == 0)
    printf("  Bcast double          (N=%d, root=0): got %.1f  expected 5.0  [%s]\n",
           N, h_buf[0], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Bcast of N floats from root=0.
// -----------------------------------------------------------------------
static int test_bcast_float(ncclComm_t comm, cudaStream_t stream, int rank)
{
  float *d_buf;
  float  h_buf[N];
  for (int i = 0; i < N; i++) h_buf[i] = (rank == 0) ? 5.0f : 0.0f;

  CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, N * sizeof(float), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclBroadcast(d_buf, d_buf, N, ncclFloat, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_buf[i] == 5.0f);
  if (rank == 0)
    printf("  Bcast float           (N=%d, root=0): got %.1f  expected 5.0  [%s]\n",
           N, (double)h_buf[0], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Bcast of N half-precision reals from root=0.
// -----------------------------------------------------------------------
static int test_bcast_half(ncclComm_t comm, cudaStream_t stream, int rank)
{
  __half *d_buf;
  __half  h_buf[N];
  for (int i = 0; i < N; i++)
    h_buf[i] = (rank == 0) ? __float2half(5.0f) : __float2half(0.0f);

  CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, N * sizeof(__half), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclBroadcast(d_buf, d_buf, N, ncclFloat16, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(__half), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (__half2float(h_buf[i]) == 5.0f);
  if (rank == 0)
    printf("  Bcast half            (N=%d, root=0): got %.1f  expected 5.0  [%s]\n",
           N, (double)__half2float(h_buf[0]), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Bcast of N double-complex numbers (as 2*N ncclDouble) from root=0.
// Rank 0: (5.0,3.0); Rank 1: (0.0,0.0) -> expected (5.0,3.0).
// -----------------------------------------------------------------------
static int test_bcast_double_complex(ncclComm_t comm, cudaStream_t stream, int rank)
{
  double *d_buf;
  double  h_buf[2 * N];
  for (int i = 0; i < N; i++) {
    h_buf[2*i]   = (rank == 0) ? 5.0 : 0.0;
    h_buf[2*i+1] = (rank == 0) ? 3.0 : 0.0;
  }

  CUDA_CHECK(cudaMalloc(&d_buf, 2 * N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, 2 * N * sizeof(double), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclBroadcast(d_buf, d_buf, 2 * N, ncclDouble, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, 2 * N * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_buf[2*i] == 5.0) && (h_buf[2*i+1] == 3.0);
  if (rank == 0)
    printf("  Bcast dbl-complex     (N=%d, root=0): got (%.1f,%.1f)  expected (5.0,3.0)  [%s]\n",
           N, h_buf[0], h_buf[1], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Bcast of N float-complex numbers (as 2*N ncclFloat) from root=0.
// -----------------------------------------------------------------------
static int test_bcast_float_complex(ncclComm_t comm, cudaStream_t stream, int rank)
{
  float *d_buf;
  float  h_buf[2 * N];
  for (int i = 0; i < N; i++) {
    h_buf[2*i]   = (rank == 0) ? 5.0f : 0.0f;
    h_buf[2*i+1] = (rank == 0) ? 3.0f : 0.0f;
  }

  CUDA_CHECK(cudaMalloc(&d_buf, 2 * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, 2 * N * sizeof(float), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclBroadcast(d_buf, d_buf, 2 * N, ncclFloat, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, 2 * N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));

  int ok = 1;
  for (int i = 0; i < N; i++) ok &= (h_buf[2*i] == 5.0f) && (h_buf[2*i+1] == 3.0f);
  if (rank == 0)
    printf("  Bcast flt-complex     (N=%d, root=0): got (%.1f,%.1f)  expected (5.0,3.0)  [%s]\n",
           N, (double)h_buf[0], (double)h_buf[1], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Bcast of N half-complex numbers (as 2*N ncclFloat16) from root=0.
// -----------------------------------------------------------------------
static int test_bcast_half_complex(ncclComm_t comm, cudaStream_t stream, int rank)
{
  __half *d_buf;
  __half  h_buf[2 * N];
  for (int i = 0; i < N; i++) {
    h_buf[2*i]   = (rank == 0) ? __float2half(5.0f) : __float2half(0.0f);
    h_buf[2*i+1] = (rank == 0) ? __float2half(3.0f) : __float2half(0.0f);
  }

  CUDA_CHECK(cudaMalloc(&d_buf, 2 * N * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, 2 * N * sizeof(__half), cudaMemcpyHostToDevice));

  NCCL_CHECK(ncclBroadcast(d_buf, d_buf, 2 * N, ncclFloat16, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_buf, d_buf, 2 * N * sizeof(__half), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));

  int ok = 1;
  for (int i = 0; i < N; i++)
    ok &= (__half2float(h_buf[2*i]) == 5.0f) && (__half2float(h_buf[2*i+1]) == 3.0f);
  if (rank == 0)
    printf("  Bcast hlf-complex     (N=%d, root=0): got (%.1f,%.1f)  expected (5.0,3.0)  [%s]\n",
           N, (double)__half2float(h_buf[0]), (double)__half2float(h_buf[1]),
           ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0)
      fprintf(stderr, "Error: this test must be run with exactly 2 MPI tasks (got %d)\n", size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int gpu_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
  if (gpu_count == 0) {
    if (rank == 0) fprintf(stderr, "Error: no CUDA-capable GPU found\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  CUDA_CHECK(cudaSetDevice(rank % gpu_count));

  // When multiple ranks share the same physical GPU (e.g. only one GPU
  // on the node), NCCL 2.18+ requires explicit opt-in.
  if (gpu_count < size)
    setenv("NCCL_MULTI_RANK_GPU_ENABLE", "1", 0);

  // Rank 0 generates the ncclUniqueId and distributes it via MPI_Bcast.
  ncclUniqueId nccl_id;
  if (rank == 0)
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, size, nccl_id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int failures = 0;

  if (rank == 0) {
    printf("=== Unit tests for NCCL Allreduce and Bcast (2 MPI tasks) ===\n\n");
    printf("Testing ncclAllReduce (sum):\n");
  }

  failures += test_allreduce_double       (comm, stream, rank);
  failures += test_allreduce_float        (comm, stream, rank);
  failures += test_allreduce_half         (comm, stream, rank);
  failures += test_allreduce_double_complex(comm, stream, rank);
  failures += test_allreduce_float_complex (comm, stream, rank);
  failures += test_allreduce_half_complex  (comm, stream, rank);

  if (rank == 0) printf("\nTesting ncclBroadcast (root=0):\n");

  failures += test_bcast_double       (comm, stream, rank);
  failures += test_bcast_float        (comm, stream, rank);
  failures += test_bcast_half         (comm, stream, rank);
  failures += test_bcast_double_complex(comm, stream, rank);
  failures += test_bcast_float_complex (comm, stream, rank);
  failures += test_bcast_half_complex  (comm, stream, rank);

  // Aggregate failure count across all ranks so both ranks exit with the same code.
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));

  if (rank == 0)
    printf("\n=== Summary: %d failure(s) ===\n", failures);

  MPI_Finalize();
  return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else /* !(WITH_NVIDIA_GPU_VERSION && WITH_NVIDIA_NCCL) */

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  fprintf(stderr,
          "Error: this test should only be called when the ELPA build "
          "is configured with NVIDIA GPU and NCCL enabled\n");
  abort();
}

#endif /* WITH_NVIDIA_NCCL */
#endif /* WITH_NVIDIA_GPU_VERSION */

#endif /* WITH_UNIT_TESTS */
