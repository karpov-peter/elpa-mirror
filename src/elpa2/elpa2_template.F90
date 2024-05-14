#if 0
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "elpa/elpa_simd_constants.h"
#include "../general/error_checking.inc"

#ifdef DEVICE_POINTER
#ifdef ACTIVATE_SKEW
 function elpa_solve_skew_evp_&
#else
 function elpa_solve_evp_&
#endif /* ACTIVATE_SKEW */
  &MATH_DATATYPE&
  &_&
  &2stage_d_ptr_&
  &PRECISION&
  &_impl (obj, &
   aDevExtern, &
   evDevExtern, &
   qDevExtern) result(success)
#else /* DEVICE_POINTER */
#ifdef ACTIVATE_SKEW
 function elpa_solve_skew_evp_&
#else
 function elpa_solve_evp_&
#endif /* ACTIVATE_SKEW */
  &MATH_DATATYPE&
  &_&
  &2stage_a_h_a_&
  &PRECISION&
  &_impl (obj, &
   aExtern, &
   evExtern, &
   qExtern) result(success)
#endif /* DEVICE_POINTER */

   !use matrix_plot
   use elpa_abstract_impl
   use elpa_utilities
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use cuda_functions
   use hip_functions
   use elpa_gpu
   use elpa1_gpu
   use mod_check_for_gpu
   use elpa_omp
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
   use simd_kernel
#endif
#ifdef REDISTRIBUTE_MATRIX
   use elpa_scalapack_interfaces
#endif
   use solve_tridi
#ifdef HAVE_AFFINITY_CHECKING
   use thread_affinity
#endif
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  use elpa_ccl_gpu
#endif
   use mod_query_gpu_usage
   use elpa2_workload
   use, intrinsic :: iso_c_binding
   implicit none
#include "../general/precision_kinds.F90"
   class(elpa_abstract_impl_t), intent(inout)                         :: obj
   logical                                                            :: useGPU
   logical                                                            :: isSkewsymmetric
#if REALCASE == 1
   logical                                                            :: useQR
   logical                                                            :: useQRActual
#endif
   integer(kind=c_int)                                                :: kernel, kernelByUser
   logical                                                            :: userHasSetKernel

#ifdef DEVICE_POINTER
   type(c_ptr)                                                        :: evDevExtern
#ifdef REDISTRIBUTE_MATRIX
   real(kind=REAL_DATATYPE), allocatable                              :: evExtern(:)
#endif /* REDISTRIBUTE_MATRIX */
#else /* DEVICE_POINTER */
   real(kind=REAL_DATATYPE), target, intent(out)                      :: evExtern(obj%na)
#endif /* DEVICE_POINTER */
   
   real(kind=REAL_DATATYPE), pointer                                  :: ev(:)
   integer(kind=ik)                                                   :: negative_or_positive
#ifdef DEVICE_POINTER
   
   type(c_ptr)                                                        :: aDevExtern
   type(c_ptr), optional                                              :: qDevExtern
   
   
#ifdef REDISTRIBUTE_MATRIX
   MATH_DATATYPE(kind=rck), allocatable                               :: aExtern(:,:)
   MATH_DATATYPE(kind=rck), allocatable                               :: qExtern(:,:)
#endif /* REDISTRIBUTE_MATRIX */
         
   
#else /* DEVICE_POINTER */

   
!#ifdef REDISTRIBUTE_MATRIX

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout), target         :: aExtern(obj%local_nrows,*)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, intent(out), target :: qExtern(obj%local_nrows,*)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout), target         :: aExtern(obj%local_nrows,obj%local_ncols)
#ifdef ACTIVATE_SKEW
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: qExtern(obj%local_nrows,2*obj%local_ncols)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: qExtern(obj%local_nrows,obj%local_ncols)
#endif
#endif

#endif /* DEVICE_POINTER */

   MATH_DATATYPE(kind=rck), pointer                                   :: a(:,:)
   MATH_DATATYPE(kind=rck), pointer                                   :: q(:,:)

   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: hh_trans(:,:)

   integer(kind=c_int)                                                :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols
   integer(kind=MPI_KIND)                                             :: my_peMPI, n_pesMPI, my_prowMPI, my_pcolMPI, &
                                                                         np_rowsMPI, np_colsMPI, mpierr
   integer(kind=c_int)                                                :: nbw, num_blocks
#if COMPLEXCASE == 1
   integer(kind=c_int)                                                :: l_cols_nev, l_rows, l_cols
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: tmat(:,:,:)
   real(kind=C_DATATYPE_KIND), allocatable                            :: e(:)
#if COMPLEXCASE == 1
   real(kind=C_DATATYPE_KIND), allocatable                            :: q_real(:,:)
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable, target           :: q_dummy(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer                       :: q_actual(:,:)


   integer(kind=c_int)                                                :: i
   logical                                                            :: success, successGPU
   integer(kind=c_int)                                                :: success_int
   logical                                                            :: wantDebug
   integer(kind=c_int)                                                :: istat, gpu, skewsymmetric, debug, qr
   character(200)                                                     :: errorMessage
   logical                                                            :: do_useGPU, do_useGPU_bandred, &
                                                                         do_useGPU_tridiag_band, do_useGPU_solve_tridi, &
                                                                         do_useGPU_trans_ev_tridi_to_band, &
                                                                         do_useGPU_trans_ev_band_to_full
   integer(kind=c_int)                                                :: numberOfGPUDevices
   integer(kind=c_intptr_t), parameter                                :: size_of_datatype = size_of_&
                                                                                            &PRECISION&
                                                                                            &_&
                                                                                            &MATH_DATATYPE
   integer(kind=c_intptr_t), parameter                                :: size_of_real_datatype = size_of_&
                                                                                                 &PRECISION&
                                                                                                 &_real
   integer(kind=ik)                                                   :: na, nev, nblk, matrixCols, &
                                                                         mpi_comm_rows, mpi_comm_cols,        &
                                                                         mpi_comm_all, check_pd, error, matrixRows
   real(kind=C_DATATYPE_KIND)                                         :: thres_pd

#ifdef REDISTRIBUTE_MATRIX
   integer(kind=ik)                                :: nblkInternal, matrixOrder
   character(len=1)                                :: layoutInternal, layoutExternal
   integer(kind=c_int)                             :: external_blacs_ctxt
   integer(kind=BLAS_KIND)                         :: external_blacs_ctxtBLAS
   integer(kind=BLAS_KIND)                         :: np_rowsInternal, np_colsInternal, my_prowInternal, my_pcolInternal
   integer(kind=BLAS_KIND)                         :: np_rowsExt, np_colsExt, my_prowExt, my_pcolExt
   integer(kind=BLAS_KIND)                         :: sc_descInternal(1:9), sc_desc(1:9), sc_descExt(1:9)
   integer(kind=BLAS_KIND)                         :: na_rowsInternal, na_colsInternal, info_, blacs_ctxtInternal
   integer(kind=BLAS_KIND)                         :: na_rowsExt, na_colsExt
   integer(kind=ik)                                :: mpi_comm_rowsInternal, mpi_comm_colsInternal
   integer(kind=MPI_KIND)                          :: mpi_comm_rowsMPIInternal, mpi_comm_colsMPIInternal
   character(len=1), parameter                     :: matrixLayouts(2) = [ 'C', 'R' ]

   MATH_DATATYPE(kind=rck), pointer                       :: aIntern(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer           :: qIntern(:,:)
   real(kind=REAL_DATATYPE), pointer                                  :: evIntern(:)
#else
   MATH_DATATYPE(kind=rck), pointer                                   :: aIntern(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer                       :: qIntern(:,:)
   real(kind=REAL_DATATYPE), pointer                                  :: evIntern(:)
#endif

   logical                                                            :: do_bandred, do_tridiag, do_solve_tridi,  &
                                                                         do_trans_to_band, do_trans_to_full
   logical                                                            :: good_nblk_gpu

   integer(kind=ik)                                                   :: nrThreads, limitThreads
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
   integer(kind=c_int)                                                :: simdSetAvailable(NUMBER_OF_INSTR)
#endif
   integer(kind=ik)                                                   :: global_index
   logical                                                            :: reDistributeMatrix, doRedistributeMatrix
   integer(kind=ik)                                                   :: pinningInfo

   integer(kind=MPI_KIND)                                             :: bcast_request1, bcast_request2, allreduce_request1, &
                                                                         allreduce_request2, allreduce_request3, &
                                                                         allreduce_request4, allreduce_request5, &
                                                                         allreduce_request6, allreduce_request7, &
                                                                         allreduce_request8
   logical                                                            :: useNonBlockingCollectivesAll
   integer(kind=ik)                                                   :: gpu_old, gpu_new
   integer(kind=ik)                                                   :: non_blocking_collectives_all
#ifdef WITH_GPU_STREAMS
   integer(kind=c_intptr_t)                        :: my_stream
#endif
   integer(kind=c_intptr_t)                         :: a_devIntern, ev_devIntern
   integer(kind=c_intptr_t)                         :: q_real_devIntern, q_devIntern

   integer(kind=c_intptr_t)                         :: q_dev, ev_dev, q_dev_actual, q_dev_dummy, &
                                                       q_dev_real, e_dev, q_part2_dev

   integer(kind=c_intptr_t)                        :: num
   integer(kind=c_intptr_t)                        :: tmat_dev, hh_trans_dev
   integer(kind=c_intptr_t)                        :: a_dev
   integer(kind=c_int)                             :: hh_trans_size
   integer(kind=ik), allocatable                   :: limits(:)
   integer(kind=ik)                                :: nx, n, local_size
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
integer(kind=c_intptr_t)                           :: ccl_comm_all
#endif

#if REALCASE == 1
#undef GPU_KERNEL
#undef GPU_KERNEL2
#define GPU_KERNEL ELPA_2STAGE_REAL_NVIDIA_GPU

#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
#define GPU_KERNEL2 ELPA_2STAGE_REAL_NVIDIA_SM80_GPU
#endif

#undef DEFAULT_KERNEL
#undef KERNEL_STRING

#define KERNEL_STRING "real_kernel"

#ifdef WITH_NVIDIA_GPU_VERSION
#undef GPU_KERNEL
#undef GPU_KERNEL2
#define GPU_KERNEL ELPA_2STAGE_REAL_NVIDIA_GPU
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
#define GPU_KERNEL2 ELPA_2STAGE_REAL_NVIDIA_SM80_GPU
#endif
#endif /* WITH_NVIDIA_GPU_VERSION */

#ifdef WITH_AMD_GPU_VERSION
#undef GPU_KERNEL
#define GPU_KERNEL ELPA_2STAGE_REAL_AMD_GPU
#endif /* WITH_AMD_GPU_VERSION */

#ifdef WITH_SYCL_GPU_VERSION
#undef GPU_KERNEL
#define GPU_KERNEL ELPA_2STAGE_REAL_INTEL_GPU_SYCL
#endif /* WITH_SYCL_GPU_VERSION */

#define DEFAULT_KERNEL ELPA_2STAGE_REAL_DEFAULT

#endif /* REALCASE */

#if COMPLEXCASE == 1
#undef GPU_KERNEL
#define GPU_KERNEL ELPA_2STAGE_COMPLEX_NVIDIA_GPU
#undef DEFAULT_KERNEL
#undef KERNEL_STRING

#define KERNEL_STRING "complex_kernel"

#ifdef WITH_NVIDIA_GPU_VERSION
#undef GPU_KERNEL
#define GPU_KERNEL ELPA_2STAGE_COMPLEX_NVIDIA_GPU
#endif /* WITH_NVIDIA_GPU_VERSION */

#ifdef WITH_AMD_GPU_VERSION
#undef GPU_KERNEL
#define GPU_KERNEL ELPA_2STAGE_COMPLEX_AMD_GPU
#endif /* WITH_AMD_GPU_VERSION */

#ifdef WITH_SYCL_GPU_VERSION
#undef GPU_KERNEL
#define GPU_KERNEL ELPA_2STAGE_COMPLEX_INTEL_GPU_SYCL
#endif /* WITH_SYCL_GPU_VERSION */

#define DEFAULT_KERNEL ELPA_2STAGE_COMPLEX_DEFAULT

#endif /* COMPLEXCASE */

   na         = obj%na
   nev        = obj%nev
   nblk       = obj%nblk
   matrixCols = obj%local_ncols
   matrixRows = obj%local_nrows

   do_tridiag       = .true.
   do_bandred       = .true.
   do_solve_tridi   = .true.
   do_trans_to_band = .true.
   do_trans_to_full = .true.

   call obj%get("nbc_all_elpa2_main", non_blocking_collectives_all, error)
   if (error .ne. ELPA_OK) then
     write(error_unit,*) "ELPA2: Problem getting option for non blocking collectives. Aborting..."
#include "./elpa2_aborting_template.F90"
   endif

   if (non_blocking_collectives_all .eq. 1) then
     useNonBlockingCollectivesAll = .true.
   else
     useNonBlockingCollectivesAll = .false.
   endif

   ! skew?  
#ifdef ACTIVATE_SKEW
   isSkewsymmetric = .true.
#else
   isSkewsymmetric = .false.
#endif
   ! eigenvalues or eigenvectors?
#ifndef DEVICE_POINTER
   if (present(qExtern)) then
#else
   if (present(qDevExtern)) then
#endif
     obj%eigenvalues_only = .false.
   else
     obj%eigenvalues_only = .true.
   endif
   if (nev == 0) then
     nev = 1
     obj%eigenvalues_only = .true.
   endif


#ifdef ACTIVATE_SKEW
    call obj%timer%start("elpa_solve_skew_evp_&
#else
    call obj%timer%start("elpa_solve_evp_&
#endif
    &MATH_DATATYPE&
    &_2stage_&
    &PRECISION&
    &")

    call obj%get("debug",debug, error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA2: Problem getting option for debug settings. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif

    wantDebug = debug == 1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! GPU 1
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    ! check legacy GPU setings
#ifdef ACTIVATE_SKEW
    if (.not.(query_gpu_usage(obj, "ELPA2_SKEW", useGPU))) then
      call obj%timer%stop("elpa_solve_skew_evp_&
      &MATH_DATATYPE&
      &_1stage_&
      &PRECISION&
      &")
#else
    if (.not.(query_gpu_usage(obj, "ELPA2", useGPU))) then
      call obj%timer%stop("elpa_solve_evp_&
      &MATH_DATATYPE&
      &_1stage_&
      &PRECISION&
      &")
#endif
      write(error_unit,*) "ELPA2: Problem getting options for GPU. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif
#endif /* defined(WITH_NVIDIA_GPU_VERSION) ... */

    do_useGPU = .false.


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! KERNEL 1
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! get the kernel and check whether it has been set by the user
    if (obj%is_set(KERNEL_STRING) == 1) then
      userHasSetKernel = .true.
    else
      userHasSetKernel = .false.
    endif

    ! after this point you may _never_ call get for the kernel again, or set the
    ! kernel !! The reason is that we might have to adjust the kernel, which is
    ! passed as a variable
    call obj%get(KERNEL_STRING, kernel, error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA2: Problem getting option for kernel settings. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif

    ! to implement a possibiltiy to set this
    useNonBlockingCollectivesAll = .false.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! MPI
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    call obj%get("mpi_comm_rows",mpi_comm_rows, error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA2: Problem getting option for mpi_comm_rows. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif
    call obj%get("mpi_comm_cols",mpi_comm_cols,error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA2: Problem getting option for mpi_comm_cols. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif
    call obj%get("mpi_comm_parent",mpi_comm_all,error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA2: Problem getting option for mpi_comm_parent. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif

    my_pe    = obj%mpi_setup%myRank_comm_parent
    my_prow = obj%mpi_setup%myRank_comm_rows
    my_pcol = obj%mpi_setup%myRank_comm_cols

    np_rows = obj%mpi_setup%nRanks_comm_rows
    np_cols = obj%mpi_setup%nRanks_comm_cols
    n_pes   = obj%mpi_setup%nRanks_comm_parent

    ! openmp setting
#include "../helpers/elpa_openmp_settings_template.F90"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! GPU 2
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (useGPU) then
      call obj%timer%start("check_for_gpu")
      if (check_for_gpu(obj, my_pe, numberOfGPUDevices, wantDebug=wantDebug)) then
        do_useGPU = .true.
        ! set the neccessary parameters
        call set_gpu_parameters()

      else
        write(error_unit,*) "ELPA2: GPUs are requested but not detected! Aborting..."
        call obj%timer%stop("check_for_gpu")
#include "./elpa2_aborting_template.F90"
      endif
      call obj%timer%stop("check_for_gpu")

      if (nblk*(max(np_rows,np_cols)-1) >= na) then
        if (my_pe .eq. 0) then
          write(error_unit,*) "ELPA: Warning, block size too large for this matrix size and process grid!"
          write(error_unit,*) "Choose a smaller block size if possible.",nblk*(max(np_rows,np_cols)-1),na
          write(error_unit,*) "Disabling GPU usage! "
        endif

        do_useGPU = .false.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! KERNEL 2
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
        if (kernel .eq. GPU_KERNEL .or. kernel .eq. GPU_KERNEL2) then
#else
        if (kernel .eq. GPU_KERNEL) then
#endif
#else /* REALCASE == 1 */
        if (kernel .eq. GPU_KERNEL) then
#endif /* REALCASE == 1 */
          if (userHasSetKernel) then
            ! user fixed inconsistent input.
            ! sadly, we do have to abort
#if COMPLECASE == 1
            if (my_pe .eq. 0) then
              write(error_unit,*) "You set (fixed) the kernel to GPU, but GPUs cannot be used."
              write(error_unit,*) "Either adapt the block size or the process grid, or do not set the GPU kernel! Aborting..."
            endif
#else
            if (my_pe .eq. 0) then
              write(error_unit,*) "You set (fixed) the kernel to GPU, but GPUs cannot be used."
              write(error_unit,*) "Either adapt the block size or the process grid, or do not set the GPU kernel! Aborting..."
            endif
#endif
            stop 1
          else
            ! here we should set the default kernel
            kernel = DEFAULT_KERNEL
          endif
        endif
      endif

    endif ! useGPU

#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
#ifdef SINGLE_PRECISION_REAL
    if (useGPU) then
      if (kernel .eq. ELPA_2STAGE_REAL_NVIDIA_SM80_GPU) then
        if (my_pe .eq. 0) then
          write(error_unit,*) "You set (fixed) the kernel to GPU, but GPUs cannot be used."
        endif
      endif
    endif ! useGPU
#endif
#endif
#endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! BANDWIDTH 1
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (obj%is_set("bandwidth") == 1) then
      ! bandwidth is set. That means, that the inputed matrix is actually banded and thus the
      ! first step of ELPA2 should be skipped
      call obj%get("bandwidth",nbw,error)
      if (nbw == 0) then
        if (wantDebug) then
          write(error_unit,*) "Specified bandwidth = 0; ELPA refuses to solve the eigenvalue problem ", &
                              "for a diagonal matrix! This is too simple"
          endif
          write(error_unit,*) "Specified bandwidth = 0; ELPA refuses to solve the eigenvalue problem ", &
                 "for a diagonal matrix! This is too simple"
        success = .false.
        return
      endif
      if (mod(nbw, nblk) .ne. 0) then
        ! treat matrix with an effective bandwidth slightly bigger than specified bandwidth
        ! such that effective bandwidth is a multiply of nblk. which is a prerequiste for ELPA
        nbw = nblk * ceiling(real(nbw,kind=c_double)/real(nblk,kind=c_double))

        ! just check that effective bandwidth is NOT larger than matrix size
        if (nbw .gt. na) then
          if (wantDebug) then
            write(error_unit,*) "Specified bandwidth ",nbw," leads internaly to a computed bandwidth ", &
                                "which is larger than the matrix size ",na," ! ELPA will abort! Try to", &
                                "solve your problem by not specifing a bandwidth"
          endif
          print *, "Specified bandwidth ",nbw," leads internaly to a computed bandwidth ", &
                                "which is larger than the matrix size ",na," ! ELPA will abort! Try to", &
                                "solve your problem by not specifing a bandwidth"
          success = .false.
          return
        endif
      endif
      do_bandred       = .false. ! we already have a banded matrix
      do_solve_tridi   = .true.  ! we also have to solve something :-)
      do_trans_to_band = .true.  ! and still we have to backsub to banded
      do_trans_to_full = .false. ! but not to full since we have a banded matrix
    else ! matrix is not banded, determine the intermediate bandwidth for full->banded->tridi
      !first check if the intermediate bandwidth was set by the user
      call obj%get("intermediate_bandwidth", nbw, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "Problem getting option for intermediate_bandwidth. Aborting..."
#include "./elpa2_aborting_template.F90"
      endif

      if(nbw == 0) then
        ! intermediate bandwidth was not specified, select one of the defaults

        ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32
        ! On older systems (IBM Bluegene/P, Intel Nehalem) a value of 32 was optimal.
        ! For Intel(R) Xeon(R) E5 v2 and v3, better use 64 instead of 32!
        ! For IBM Bluegene/Q this is not clear at the moment. We have to keep an eye
        ! on this and maybe allow a run-time optimization here
#if REALCASE == 1
        nbw = (63/nblk+1)*nblk
#elif COMPLEXCASE == 1
        nbw = (31/nblk+1)*nblk
#endif
      else
        ! intermediate bandwidth has been specified by the user, check, whether correctly
        if (mod(nbw, nblk) .ne. 0) then
          print *, "Specified bandwidth ",nbw," has to be mutiple of the blocksize ", nblk, ". Aborting..."
#include "./elpa2_aborting_template.F90"
        endif
      endif !nbw == 0

      num_blocks = (na-1)/nbw + 1
    endif  ! matrix not already banded on input

    do_useGPU_bandred = do_useGPU
    do_useGPU_tridiag_band = do_useGPU  
    do_useGPU_solve_tridi = do_useGPU
    do_useGPU_trans_ev_tridi_to_band = do_useGPU
    do_useGPU_trans_ev_band_to_full = do_useGPU


   reDistributeMatrix = .false.

#if COMPLEXCASE == 1
   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev
#endif

#ifdef REDISTRIBUTE_MATRIX
   ! if a matrix redistribution is done then
   ! - aIntern, qIntern are getting allocated for the new distribution
   ! - nblk, matrixCols, matrixRows, mpi_comm_cols, mpi_comm_rows are getting updated
   ! TODO: make sure that nowhere in ELPA the communicators are getting "getted",
   ! and the variables obj%local_nrows,1:obj%local_ncols are being used
   ! - a points then to aIntern, q points to qIntern
#include "../helpers/elpa_redistribute_template.F90"
#endif /* REDISTRIBUTE_MATRIX */


   allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
   check_allocate("tridiag_band: limits", istat, errorMessage)

   call determine_workload(obj, na, nbw, np_rows, limits)
   hh_trans_size = 0
   nx = na
   do n = 1, (na-1)/nbw+1
     call determine_workload(obj, nx, nbw, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     ! add to number of householder vectors     
     !    ! please note: for nx==1 the one and only HH Vector is 0 and is neither calculated nor send below!
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       hh_trans_size = hh_trans_size + local_size   
     endif
     nx = nx - nbw
   enddo
   deallocate(limits)

#ifndef DEVICE_POINTER
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! no device pointer
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   allocate(e(na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa2_template: e, ", istat, errorMessage)

   allocate(hh_trans(1:nbw,1:hh_trans_size), stat=istat, errmsg=errorMessage)
   check_allocate("elpa2_template: hh_trans", istat, errorMessage)

   if (do_bandred) then
     ! tmat is needed only in full->band and band->full steps, so alocate here
     ! (not allocated for banded matrix on input)
     allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
     check_allocate("elpa2_template: tmat", istat, errorMessage)
   endif
#ifndef REDISTRIBUTE_MATRIX
   aIntern => aExtern(1:matrixRows,1:matrixCols)
   a => aExtern(1:matrixRows,1:matrixCols)
   if (present(qExtern)) then
     if (isSkewsymmetric) then
       qIntern => qExtern(1:matrixRows,1:2*matrixCols)
       q => qExtern(1:matrixRows,1:2*matrixCols)
     else
       qIntern => qExtern(1:matrixRows,1:matrixCols)
       q => qExtern(1:matrixRows,1:matrixCols)
     endif
   endif
   evIntern => evExtern(1:obj%na)
   ev => evExtern(1:obj%na)

   ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
   if (.not.(obj%eigenvalues_only)) then
     q_actual => q(1:matrixRows,1:matrixCols)
   else
     allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa2_template: q_dummy", istat, errorMessage)
     q_actual => q_dummy
   endif

#if COMPLEXCASE == 1
   allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
   check_allocate("elpa2_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */

   if (doRedistributeMatrix) then
     !aIntern => aExtern(1:matrixRows,1:matrixCols)
     a => aIntern(1:matrixRows,1:matrixCols)
     if (present(qExtern)) then
       if (isSkewsymmetric) then
         !qIntern => qExtern(1:matrixRows,1:2*matrixCols)
         q => qIntern(1:matrixRows,1:2*matrixCols)
       else
         !qIntern => qExtern(1:matrixRows,1:matrixCols)
         q => qIntern(1:matrixRows,1:matrixCols)
       endif
     endif
     ! ev never changes
     evIntern => evExtern(1:obj%na)
     ev => evIntern(1:obj%na)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_actual => q(1:matrixRows,1:matrixCols)
     else
       allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
       check_allocate("elpa2_template: q_dummy", istat, errorMessage)
       q_actual => q_dummy
     endif

#if COMPLEXCASE == 1
     allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa2_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     aIntern => aExtern(1:matrixRows,1:matrixCols)
     a => aExtern(1:matrixRows,1:matrixCols)
     if (present(qExtern)) then
       if (isSkewsymmetric) then
         qIntern => qExtern(1:matrixRows,1:2*matrixCols)
         q => qExtern(1:matrixRows,1:2*matrixCols)
       else
         qIntern => qExtern(1:matrixRows,1:matrixCols)
         q => qExtern(1:matrixRows,1:matrixCols)
       endif
     endif
     evIntern => evExtern(1:obj%na)
     ev => evExtern(1:obj%na)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_actual => q(1:matrixRows,1:matrixCols)
     else
       allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
       check_allocate("elpa2_template: q_dummy", istat, errorMessage)
       q_actual => q_dummy
     endif

#if COMPLEXCASE == 1
     allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa2_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */

   if (useGPU) then
     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(e_dev, num)
     check_alloc_gpu("elpa2_template e_dev", successGPU)

     if (do_bandred) then
       num = (nbw*nbw*num_blocks) * size_of_datatype
       successGPU = gpu_malloc(tmat_dev, num)
       check_alloc_gpu("elpa2_template tmat_devIntern", successGPU)
     endif

     num = (nbw*hh_trans_size) * size_of_datatype
     successGPU = gpu_malloc(hh_trans_dev, num)
     check_alloc_gpu("elpa2_template hh_trans_dev", successGPU)


#ifndef REDISTRIBUTE_MATRIX
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     num = (matrixRows* matrixCols) * size_of_datatype
     successGPU = gpu_malloc(a_devIntern, num)
     check_alloc_gpu("elpa2_template a_devIntern", successGPU)

     successGPU = gpu_memcpy(a_devIntern, int(loc(a(1,1)),kind=c_intptr_t), &
                 num, gpuMemcpyHostToDevice)
     check_memcpy_gpu("elpa2_template a -> a_devIntern", successGPU)

     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(ev_devIntern, num)
     check_alloc_gpu("elpa2_template ev_devIntern", successGPU)

     if (present(qExtern)) then
       if (isSkewsymmetric) then
         num = (matrixRows* 2*matrixCols) * size_of_datatype
       else
         num = (matrixRows* matrixCols) * size_of_datatype
       endif
       successGPU = gpu_malloc(q_devIntern, num)
       check_alloc_gpu("elpa2_template q_devIntern", successGPU)
     endif

     ! associate a_dev, q_dev, ev_dev
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(ev_devIntern, ev_dev)
     if (present(qExtern)) then
       q_dev = transfer(q_devIntern, q_dev)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa2_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */


   if (doRedistributeMatrix) then

     !  a_devIntern, q_devIntern already allocated
     !
     !! alloc a_devIntern, q_devIntern, ev_devIntern
     !num = (matrixRows* matrixCols) * size_of_datatype
     !successGPU = gpu_malloc(a_devIntern, num)
     !check_alloc_gpu("elpa2_template a_devIntern", successGPU)

     !successGPU = gpu_memcpy(a_devIntern, int(loc(a(1,1)),kind=c_intptr_t), &
     !            num, gpuMemcpyHostToDevice)
     !check_memcpy_gpu("elpa2_template a -> a_devIntern", successGPU)

     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(ev_devIntern, num)
     check_alloc_gpu("elpa2_template ev_devIntern", successGPU)
     !successGPU = gpu_memcpy(ev_devIntern, int(loc(ev(1)),kind=c_intptr_t), &
     !            num, gpuMemcpyHostToDevice)
     !check_memcpy_gpu("elpa2_template ev -> ev_devIntern", successGPU)
     !if (present(qExtern)) then
     !  if (isSkewsymmetric) then
     !    num = (matrixRows* 2*matrixCols) * size_of_datatype
     !  else
     !    num = (matrixRows* matrixCols) * size_of_datatype
     !  endif
     !  successGPU = gpu_malloc(q_devIntern, num)
     !  check_alloc_gpu("elpa2_template q_devIntern", successGPU)
     !endif
     allocate(aIntern(matrixRows,matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa2_template: aIntern", istat, errorMessage)
     if (isSkewsymmetric) then
       allocate(qIntern(matrixRows,2*matrixCols), stat=istat, errmsg=errorMessage)
     else
       allocate(qIntern(matrixRows,matrixCols), stat=istat, errmsg=errorMessage)
     endif
     check_allocate("elpa2_template: qIntern", istat, errorMessage)

     if (isSkewsymmetric) then
       q => qIntern(1:matrixRows,1:2*matrixCols)
     else
       q => qIntern(1:matrixRows,1:matrixCols)
     endif
     a => aIntern(1:matrixRows,1:matrixCols)

     ! associate a_dev, q_dev, ev_dev
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(ev_devIntern, a_dev)
     if (present(qExtern)) then
       q_dev = transfer(q_devIntern, q_dev)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa2_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     num = (matrixRows* matrixCols) * size_of_datatype
     successGPU = gpu_malloc(a_devIntern, num)
     check_alloc_gpu("elpa2_template a_devIntern", successGPU)

     successGPU = gpu_memcpy(a_devIntern, int(loc(a(1,1)),kind=c_intptr_t), &
                 num, gpuMemcpyHostToDevice)
     check_memcpy_gpu("elpa2_template a -> a_devIntern", successGPU)

     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(ev_devIntern, num)
     check_alloc_gpu("elpa2_template ev_devIntern", successGPU)
     !successGPU = gpu_memcpy(ev_devIntern, int(loc(ev(1)),kind=c_intptr_t), &
     !            num, gpuMemcpyHostToDevice)
     !check_memcpy_gpu("elpa2_template ev -> ev_devIntern", successGPU)
     if (present(qExtern)) then
       if (isSkewsymmetric) then
         num = (matrixRows* 2*matrixCols) * size_of_datatype
       else
         num = (matrixRows* matrixCols) * size_of_datatype
       endif
       successGPU = gpu_malloc(q_devIntern, num)
       check_alloc_gpu("elpa2_template q_devIntern", successGPU)
     endif

     ! associate a_dev, q_dev, ev_dev
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(ev_devIntern, a_dev)
     if (present(qExtern)) then
       q_dev = transfer(q_devIntern, q_dev)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa2_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU
#else /* DEVICE_POINTER */
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   ! DEVICE POINTER
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   allocate(evIntern(na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa2_template: evIntern", istat, errorMessage)
   ev => evIntern(1:na)
   allocate(e(na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa2_template: e", istat, errorMessage)

   if (useGPU) then
     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(e_dev, num)
     check_alloc_gpu("elpa2_template e_dev", successGPU)

     if (do_bandred) then
       num = (nbw*nbw*num_blocks) * size_of_datatype
       successGPU = gpu_malloc(tmat_dev, num)
       check_alloc_gpu("elpa2_template tmat_devIntern", successGPU)
     endif

     num = (nbw*hh_trans_size) * size_of_datatype
     successGPU = gpu_malloc(hh_trans_dev, num)
     check_alloc_gpu("elpa2_template hh_trans_dev", successGPU)

#ifndef REDISTRIBUTE_MATRIX
     ! associate a_dev, q_dev, ev_dev
     a_devIntern = transfer(aDevExtern, a_devIntern)
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(evDevExtern, a_dev)
     ev_devIntern = transfer(evDevExtern, ev_devIntern)
     if (present(qDevExtern)) then
       q_dev = transfer(qDevExtern, q_dev)
       q_devIntern = transfer(qDevExtern, q_devIntern)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa2_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */
     if (doRedistributeMatrix) then
       ! associate a_dev, q_dev, ev_dev
       !a_devIntern = transfer(aDevExtern, a_devIntern)
       a_dev = transfer(a_devIntern, a_dev)
       ev_dev = transfer(evDevExtern, ev_dev)
       ev_devIntern = transfer(evDevExtern, ev_devIntern)
       if (present(qDevExtern)) then
         q_dev = transfer(q_devIntern, q_dev)
       endif

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
         q_dev_actual = transfer(q_dev, q_dev_actual)
       else
         num = (matrixRows* matrixCols) * size_of_datatype
         successGPU = gpu_malloc(q_dev_dummy, num)
         check_alloc_gpu("elpa2_template q_dev_dummy", successGPU)
         q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
       endif

#if COMPLEXCASE == 1
       num = (l_rows* l_cols) * size_of_real_datatype
       successGPU = gpu_malloc(q_dev_real, num)
       check_alloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

     else ! doRedistributeMatrix
       ! associate a_dev, q_dev, ev_dev
       a_devIntern = transfer(aDevExtern, a_devIntern)
       a_dev = transfer(a_devIntern, a_dev)
       ev_dev = transfer(evDevExtern, a_dev)
       ev_devIntern = transfer(evDevExtern, a_devIntern)
       if (present(qDevExtern)) then
         q_devIntern = transfer(qDevExtern, q_devIntern)
         q_dev = transfer(q_devIntern, q_dev)
       endif

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
         q_dev_actual = transfer(q_dev, q_dev_actual)
       else
         num = (matrixRows* matrixCols) * size_of_datatype
         successGPU = gpu_malloc(q_dev_dummy, num)
         check_alloc_gpu("elpa2_template q_dev_dummy", successGPU)
         q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
       endif

#if COMPLEXCASE == 1
       num = (l_rows* l_cols) * size_of_real_datatype
       successGPU = gpu_malloc(q_dev_real, num)
       check_alloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
     endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU

#endif /* DEVICE_POINTER */

! no allocate after here


   call obj%get("output_pinning_information", pinningInfo, error)
   if (error .ne. ELPA_OK) then
     write(error_unit,*) "ELPA2: Problem setting option for debug. Aborting..."
#include "./elpa2_aborting_template.F90"
   endif

#ifdef HAVE_AFFINITY_CHECKING
   if (pinningInfo .eq. 1) then
     call init_thread_affinity(nrThreads)

     call check_thread_affinity()
     if (my_pe .eq. 0) call print_thread_affinity(my_pe)
     call cleanup_thread_affinity()
   endif
#endif

    success = .true.

#ifndef DEVICE_POINTER
   ! special case na = 1
   if (na .eq. 1) then
#if REALCASE == 1
     ev(1) = a(1,1)
#endif
#if COMPLEXCASE == 1
     ev(1) = real(a(1,1))
#endif
     if (.not.(obj%eigenvalues_only)) then
       q(1,1) = ONE
     endif

     ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
     ! store the number of OpenMP threads used in the calling function
     ! restore this at the end of ELPA 2
     call omp_set_num_threads(omp_threads_caller)
#endif

#ifdef ACTIVATE_SKEW
     call obj%timer%stop("elpa_solve_skew_evp_&
#else
     call obj%timer%stop("elpa_solve_evp_&
#endif
     &MATH_DATATYPE&
     &_2stage_&
     &PRECISION&
     &")
     success = .true.
     return
   endif
#endif
   if (nev == 0) then
     nev = 1
     obj%eigenvalues_only = .true.
   endif

#ifdef ACTIVATE_SKEW
    isSkewsymmetric = .true.
#else
    isSkewsymmetric = .false.
#endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! GPU 3
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! only if we want (and can) use GPU in general, look what are the
    ! requirements for individual routines. Implicitly they are all set to 1, so
    ! unless specified otherwise by the user, GPU versions of all individual
    ! routines should be used
    if (do_useGPU) then
      call obj%get("gpu_bandred", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "ELPA2: Problem getting option gpu_bandred settings. Aborting..."
#include "./elpa2_aborting_template.F90"
      endif
      do_useGPU_bandred = (gpu == 1)

      ! not yet ported
      !call obj%get("gpu_tridiag_band", gpu, error)
      !if (error .ne. ELPA_OK) then
      !  print *,"Problem getting option for gpu_tridiag_band settings. Aborting..."
      !  stop 1
      !endif
      !do_useGPU_tridiag_band = (gpu == 1)

      call obj%get("gpu_solve_tridi", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "ELPA2: Problem getting option for gpu_solve_tridi settings. Aborting..."
#include "./elpa2_aborting_template.F90"
      endif
      do_useGPU_solve_tridi = (gpu == 1)

      call obj%get("gpu_trans_ev_tridi_to_band", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "ELPA2: Problem getting option for gpu_trans_ev_tridi_to_band settings. Aborting..."
#include "./elpa2_aborting_template.F90"
      endif
      do_useGPU_trans_ev_tridi_to_band = (gpu == 1)

      call obj%get("gpu_trans_ev_band_to_full", gpu, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "ELPA2: Problem getting option for gpu_trans_ev_band_to_full settings. Aborting..."
#include "./elpa2_aborting_template.F90"
      endif
      do_useGPU_trans_ev_band_to_full = (gpu == 1)
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU_trans_ev_tridi_to_band) then
      ! if the user has set explicitely a kernel before than honour this
      ! otherwise set GPU kernel
      if (userHasSetKernel) then
#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
        if (kernel .ne. GPU_KERNEL .and. kernel .ne. GPU_KERNEL2) then
#else
        if (kernel .ne. GPU_KERNEL) then
#endif
#else /* REALCASE == 1 */
        if (kernel .ne. GPU_KERNEL) then
#endif /* REALCASE == 1 */
          write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is set by the user as non-GPU!"
          write(error_unit,*) "The compute kernel will be executed on CPUs!"
          do_useGPU_trans_ev_tridi_to_band = .false.
          kernel = DEFAULT_KERNEL
        else
          good_nblk_gpu = .false.
        endif
      else ! userHasSetKernel
        !call obj%set(KERNEL_STRING, GPU_KERNEL, error)
        !if (error .ne. ELPA_OK) then
        !  write(error_unit,*) "Cannot set kernel to GPU kernel"
        !  stop 1
        !endif

        if (my_pe .eq. 0) write(error_unit,*) "You requested the GPU version, thus the GPU kernel is activated"
        good_nblk_gpu = .false.

#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
        kernel = GPU_KERNEL2
#else
        kernel = GPU_KERNEL
#endif
#else /* REALCASE == 1 */
        kernel = GPU_KERNEL
#endif /* REALCASE == 1 */
      endif ! userHasSetKernel

      ! Accepted values are 2,4,8,16,...,512
      do i = 1,10
        if (nblk == 2**i) then
          good_nblk_gpu = .true.
          exit
        endif
      enddo

      if (.not. good_nblk_gpu .and. do_useGPU_trans_ev_tridi_to_band) then
         write(error_unit,*) "ELPA: Warning, GPU kernel only works with block size 2^n (n = 1, 2, ..., 10)!"
         write(error_unit,*) "The compute kernel will be executed on CPUs!"
         write(error_unit,*) "We recommend changing the block size to 2^n"
         do_useGPU_trans_ev_tridi_to_band = .false.
         ! should be set
         kernel = DEFAULT_KERNEL
      endif
    endif


#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
#ifdef SINGLE_PRECISION_REAL
    if (useGPU) then
      if (kernel .eq. ELPA_2STAGE_REAL_NVIDIA_SM80_GPU) then
        kernel = ELPA_2STAGE_REAL_NVIDIA_GPU
        if (my_pe .eq. 0) then
          !write(error_unit,*) "Currently no MMA implementation for real single-precision Nvidia SM80 kernel."
          !write(error_unit,*) "Using without MMA."
          write(error_unit,*) "Currently no implementation of real single-precision Nvidia SM80 kernel."
          write(error_unit,*) "Using the standard Nvidia GPU kernel instead."
        endif
      endif
    endif ! useGPU
#endif
#endif
#endif
#if COMPLEXCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
    if (useGPU) then
      if (kernel .eq. ELPA_2STAGE_COMPLEX_NVIDIA_SM80_GPU) then
        kernel = ELPA_2STAGE_REAL_NVIDIA_GPU
        if (my_pe .eq. 0) then
          write(error_unit,*) "Currently no implementation of complex Nvidia SM80 kernel." 
          write(error_unit,*) "Using the standard Nvidia GPU kernel instead."
        endif
      endif
    endif ! useGPU
#endif
#endif



    ! check again, now kernel and do_useGPU_trans_ev_tridi_to_band should be
    ! finally consistent
    if (do_useGPU_trans_ev_tridi_to_band) then
      !call obj%get(KERNEL_STRING, kernel, error)
      !if (error .ne. ELPA_OK) then
      !  write(error_unit,*) "Cannot get kernel to GPU kernel"
      !  stop 1
      !endif
#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
      if (kernel .ne. GPU_KERNEL .and. kernel .ne. GPU_KERNEL2) then
#else
      if (kernel .ne. GPU_KERNEL) then
#endif
#else /* REALCASE == 1 */
      if (kernel .ne. GPU_KERNEL) then
#endif /* REALCASE == 1 */
        ! this should never happen, checking as an assert
        write(error_unit,*) "ELPA: INTERNAL ERROR setting GPU kernel!  Aborting..."
        stop 1
      endif
    else ! do not use GPU
#if REALCASE == 1
#ifdef WITH_REAL_NVIDIA_SM80_GPU_KERNEL
      if (kernel .eq. GPU_KERNEL .or. kernel .eq. GPU_KERNEL2) then
#else
      if (kernel .eq. GPU_KERNEL) then
#endif
#else /* REALCASE == 1 */
      if (kernel .eq. GPU_KERNEL) then
#endif /* REALCASE == 1 */
        ! combination not allowed
        write(error_unit,*) "ELPA: Warning, GPU usage has NOT been requested but compute kernel &
                            &is defined as the GPU kernel!  Setting default kernel"
        kernel = DEFAULT_KERNEL
        !TODO do error handling properly
      endif
    endif

#if REALCASE == 1
#ifdef SINGLE_PRECISION_REAL
    ! special case at the moment NO single precision kernels on POWER 8 -> set GENERIC for now
    if (kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK4 .or. &
        kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK6        ) then
        write(error_unit,*) "ELPA: At the moment there exist no specific SINGLE precision kernels for POWER8"
        write(error_unit,*) "The GENERIC kernel will be used at the moment"
        kernel = ELPA_2STAGE_REAL_GENERIC
    endif
    ! special case at the moment NO single precision kernels on SPARC64 -> set GENERIC for now
    if (kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK4 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK6        ) then
        write(error_unit,*) "ELPA: At the moment there exist no specific SINGLE precision kernels for SPARC64"
        write(error_unit,*) "The GENERIC kernel will be used at the moment"
        kernel = ELPA_2STAGE_REAL_GENERIC
    endif
#endif

#endif /* REALCASE == 1 */

     !! consistency check: is user set kernel still identical with "kernel" or did
     !! we change it above? This is a mess and should be cleaned up
     !call obj%get(KERNEL_STRING,kernelByUser,error)
     !if (error .ne. ELPA_OK) then
     !  print *,"Problem getting option for user kernel settings. Aborting..."
     !  stop 1
     !endif

     !if (kernelByUser .ne. kernel) then
     !  call obj%set(KERNEL_STRING, kernel, error)
     !  if (error .ne. ELPA_OK) then
     !    print *,"Problem setting kernel. Aborting..."
     !    stop 1
     !  endif
     !endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! SIMD
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef HAVE_HETEROGENOUS_CLUSTER_SUPPORT
     ! find a kernel which is supported on all used CPUs
     ! at the moment this works only on Intel CPUs
     simdSetAvailable(:) = 0
     call get_cpuid_set(simdSetAvailable, NUMBER_OF_INSTR)
#ifdef WITH_MPI
     if (useNonBlockingCollectivesAll) then
       call mpi_iallreduce(mpi_in_place, simdSetAvailable, NUMBER_OF_INSTR, MPI_INTEGER, MPI_BAND, int(mpi_comm_all,kind=MPI_KIND), &
       allreduce_request1, mpierr)
       call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
     else
       call mpi_allreduce(mpi_in_place, simdSetAvailable, NUMBER_OF_INSTR, MPI_INTEGER, MPI_BAND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
     endif
#endif

     !! compare user chosen kernel with possible kernels
     !call obj%get(KERNEL_STRING,kernelByUser,error)
     !if (error .ne. ELPA_OK) then
     !  print *,"Problem getting option for user kernel settings. Aborting..."
     !  stop 1
     !endif

     ! map kernel to SIMD Set, and check whether this is set is available on all cores

#if REALCASE == 1
    if (simdSetAvailable(map_real_kernel_to_simd_instruction(kernel)) /= 1) then
#endif
#if COMPLEXCASE == 1
    if (simdSetAvailable(map_complex_kernel_to_simd_instruction(kernel)) /=1) then
#endif

      ! if we are not purely running on Intel CPUs, this feature does not work at the moment
      ! this restriction should be lifted step by step
      if (simdSetAvailable(CPU_MANUFACTURER) /= 1) then
         if (my_pe == 0 ) then
         write(error_unit,*) "You enabled the experimental feature of an heterogenous cluster support."
         write(error_unit,*) "However, this works at the moment only if ELPA is run on (different) Intel CPUs!"
         write(error_unit,*) "ELPA detected also non Intel-CPUs, and will this abort now"
         stop 1
        endif
      else
        if (my_pe == 0 ) then
          write(error_unit,*) "The ELPA 2stage kernel of your choice, cannot be run on all CPUs"
          write(error_unit,*) "ELPA will use another kernel..."
        endif

        ! find best kernel available for supported instruction sets
        do i = NUMBER_OF_INSTR, 2, -1
          if (simdSetAvailable(i) == 1) then
            ! map to "best" kernel with this instruction set
            ! this can be only done for kernels that ELPA has been configured to use
#if REALCASE == 1
            kernel = map_simd_instruction_to_real_kernel(i)
#endif
#if COMPLEXCASE == 1
            kernel = map_simd_instruction_to_complex_kernel(i)
#endif
            if (obj%can_set(KERNEL_STRING, kernel) == ELPA_OK) then
            !  call obj%set(KERNEL_STRING, kernel, error)
            !  if (error .ne. ELPA_OK) then
            !    print *,"Problem setting kernel. Aborting..."
            !    stop 1
            !  endif
            if (my_pe == 0 ) write(error_unit,*) "ELPA decided to use ",elpa_int_value_to_string(KERNEL_STRING, kernel)
              exit
            endif
          endif
        enddo
      endif

    endif
#endif /* HAVE_HETEROGENOUS_CLUSTER_SUPPORT */


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! QR
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#if REALCASE == 1
    call obj%get("qr",qr,error)
    if (error .ne. ELPA_OK) then
      print *,"ELPA2: Problem getting option for qr settings. Aborting..."
#include "./elpa2_aborting_template.F90"
    endif
    if (qr .eq. 1) then
      useQR = .true.
    else
      useQR = .false.
    endif
#endif /* REALCASE == 1 */

#if REALCASE == 1
    useQRActual = .false.
    ! set usage of qr decomposition via API call
    if (useQR) useQRActual = .true.
    if (.not.(useQR)) useQRACtual = .false.

    if (useQRActual) then
      if (mod(na,2) .ne. 0) then
        if (wantDebug) then
          write(error_unit,*) "solve_evp_real_2stage: QR-decomposition: blocksize does not fit with matrixsize"
        endif
        write(error_unit,*) "Do not use QR-decomposition for this matrix and blocksize."
        success = .false.
        return
      endif
    endif
#endif /* REALCASE */

    if (obj%eigenvalues_only) then
      do_trans_to_band  = .false.
      do_trans_to_full  = .false.
    endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! START COMPUTE
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! start the computations in 5 steps
    if (do_bandred) then
      call obj%autotune_timer%start("full_to_band")
      call obj%timer%start("full_to_band")
#ifdef HAVE_LIKWID
      call likwid_markerStartRegion("full_to_band")
#endif
      ! Reduction full -> band
      if (do_useGPU_bandred) then

        call bandred_gpu_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, na, a_dev, &
        matrixRows, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, tmat_dev, &
        wantDebug, success, &
#if REALCASE == 1
        useQRActual, &
#endif
         nrThreads, isSkewsymmetric)

      else
        call bandred_cpu_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, na, a, &
        matrixRows, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, tmat, &
        wantDebug, success, &
#if REALCASE == 1
        useQRActual, &
#endif
         nrThreads, isSkewsymmetric)
      endif

#ifdef HAVE_LIKWID
      call likwid_markerStopRegion("full_to_band")
#endif
      call obj%timer%stop("full_to_band")
      call obj%autotune_timer%stop("full_to_band")
  
      if (success) then
        success_int = 0
      else
        success_int = 1
      endif

#ifdef WITH_MPI
      if (useNonBlockingCollectivesAll) then
        call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
        allreduce_request2, mpierr)
        call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
      else
        call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
      endif
#endif
      if (success_int .eq. 1) then
        write(error_unit,*) "ELPA2: bandred returned an error. Aborting..."
#include "./elpa2_aborting_template.F90"
      endif
    endif

     ! Reduction band -> tridiagonal
     if (do_tridiag) then

       call obj%autotune_timer%start("band_to_tridi")
       call obj%timer%start("band_to_tridi")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("band_to_tridi")
#endif


       if (do_useGPU_tridiag_band) then

         ! hh_trans, hh_trans_dev is allocated WITHIN the tridiag routine
         call tridiag_band_gpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION&
         (obj, na, nbw, nblk, hh_trans_size, a_dev, matrixRows, ev_dev, e_dev, matrixCols, hh_trans_dev,  mpi_comm_rows, &
          mpi_comm_cols, mpi_comm_all, &
          wantDebug, nrThreads, isSkewsymmetric, success)

       else
         call tridiag_band_cpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION&
         (obj, na, nbw, nblk, hh_trans_size, a, matrixRows, ev, e, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, &
         mpi_comm_all, &
          wantDebug, nrThreads, isSkewsymmetric, success)
       endif 

       if (success) then
         success_int = 0
       else
         success_int = 1
       endif

#ifdef WITH_MPI
       if (useNonBlockingCollectivesAll) then
         call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
         allreduce_request3, mpierr)
         call mpi_wait(allreduce_request3, MPI_STATUS_IGNORE, mpierr)
       else
         call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       endif
#endif
       if (success_int .eq. 1) then
         write(error_unit,*) "Error in tridiag_band. Aborting..."
         return
       endif

#ifdef WITH_MPI
       call obj%timer%start("mpi_communication")


       if (useGPU) then
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_all = obj%gpu_setup%ccl_comm_all
        successGPU = ccl_group_start()
        if (.not.successGPU) then
          print *,"Error in setting up nccl_group_start!"
          stop
        endif
        successGPU = ccl_bcast(ev_dev, ev_dev, &
                         int(na,kind=c_size_t), &
#ifdef DOUBLE_PRECISION
                         cclDouble, &
#endif
#ifdef SINGLE_PRECISION
                         cclFloat, &
#endif
                         0_c_int, ccl_comm_all, my_stream)
        successGPU = ccl_bcast(e_dev, e_dev, &
                         int(na,kind=c_size_t), &
#ifdef DOUBLE_PRECISION
                         cclDouble, &
#endif
#ifdef SINGLE_PRECISION
                         cclFloat, &
#endif
                         0_c_int, ccl_comm_all, my_stream)


          if (.not.successGPU) then
            print *,"Error in nccl_reduce"
            stop
          endif
          successGPU = ccl_group_end()
          if (.not.successGPU) then
            print *,"Error in setting up nccl_group_end!"
            stop
          endif

#else
         num = na * size_of_real_datatype
         successGPU = gpu_memcpy(int(loc(ev(1)),kind=c_intptr_t), ev_devIntern, &
                      num, gpuMemcpyDeviceToHost)
         check_memcpy_gpu("elpa2_template ev_devIntern -> ev", successGPU)
         num = na * size_of_real_datatype
         successGPU = gpu_memcpy(int(loc(e(1)),kind=c_intptr_t), e_dev, &
                      num, gpuMemcpyDeviceToHost)
         check_memcpy_gpu("elpa2_template e_dev -> e", successGPU)


         if (useNonBlockingCollectivesAll) then
           call mpi_ibcast(ev, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), &
                   bcast_request1, mpierr)
           call mpi_ibcast(e, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), &
                   bcast_request2, mpierr)

           call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
           call mpi_wait(bcast_request2, MPI_STATUS_IGNORE, mpierr)
         else
           call mpi_bcast(ev, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
           call mpi_bcast(e, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
         endif
         num = na * size_of_real_datatype
         successGPU = gpu_memcpy(ev_devIntern, int(loc(ev(1)),kind=c_intptr_t), &
                      num, gpuMemcpyHostToDevice)
         check_memcpy_gpu("elpa2_template ev -> ev_devIntern:", successGPU)
         num = na * size_of_real_datatype
         successGPU = gpu_memcpy(e_dev, int(loc(e(1)),kind=c_intptr_t), &
                      num, gpuMemcpyHostToDevice)
         check_memcpy_gpu("elpa2_template e -> e_dev:", successGPU)
#endif
       else
         if (useNonBlockingCollectivesAll) then
           call mpi_ibcast(ev, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), &
                   bcast_request1, mpierr)
           call mpi_ibcast(e, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), &
                   bcast_request2, mpierr)

           call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
           call mpi_wait(bcast_request2, MPI_STATUS_IGNORE, mpierr)
         else
           call mpi_bcast(ev, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
           call mpi_bcast(e, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, 0_MPI_KIND, int(mpi_comm_all,kind=MPI_KIND), mpierr)
         endif
       endif
       call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("band_to_tridi")
#endif
       call obj%timer%stop("band_to_tridi")
       call obj%autotune_timer%stop("band_to_tridi")
     endif ! do_tridiag

     ! Solve tridiagonal system
     if (do_solve_tridi) then
       call obj%autotune_timer%start("solve")
       call obj%timer%start("solve")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("solve")
#endif


       if (useGPU) then
         call solve_tridi_gpu_&
         &PRECISION&
         & (obj, na, nev, ev_dev, e_dev,  &
#if REALCASE == 1
           q_dev_actual, matrixRows,          &
#endif
#if COMPLEXCASE == 1
          q_dev_real, l_rows,  &
#endif 
          nblk, matrixCols, mpi_comm_all, mpi_comm_rows, mpi_comm_cols, wantDebug, &
                success, nrThreads)
       else

         call solve_tridi_cpu_&
         &PRECISION &
         (obj, na, nev, ev, e, &
#if REALCASE == 1
         q_actual, matrixRows,   &
#endif
#if COMPLEXCASE == 1
         q_real, l_rows, &
#endif
         nblk, matrixCols, mpi_comm_all, mpi_comm_rows, mpi_comm_cols, wantDebug, success, nrThreads)
       endif

#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("solve")
#endif
       call obj%timer%stop("solve")
       call obj%autotune_timer%stop("solve")
  
       if (success) then
         success_int = 0
       else
         success_int = 1
       endif

#ifdef WITH_MPI
       if (useNonBlockingCollectivesAll) then
         call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
         allreduce_request4, mpierr)
         call mpi_wait(allreduce_request4, MPI_STATUS_IGNORE, mpierr)
       else
         call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       endif
#endif

       if (success_int .eq. 1) then
         write(error_unit,*) "ELPA2: solve returned an error: Aborting..."
#include "./elpa2_aborting_template.F90"
       endif
     endif ! do_solve_tridi



     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     !!!
     !
     ! TODO  check whether needed
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     !deallocate(e, stat=istat, errmsg=errorMessage)
     !check_deallocate("elpa2_template: e", istat, errorMessage)

     if (obj%eigenvalues_only) then
       do_trans_to_band = .false.
       do_trans_to_full = .false.
     else
       call obj%get("check_pd",check_pd,error)
       if (error .ne. ELPA_OK) then
         write(error_unit,*) "Problem getting option for check_pd. Aborting..."
#include "./elpa2_aborting_template.F90"
       endif
       if (check_pd .eq. 1) then
         call obj%get("thres_pd_&
         &PRECISION&
         &",thres_pd,error)
         if (error .ne. ELPA_OK) then
            write(error_unit,*) "Problem getting option for thres_pd_&
            &PRECISION&
            &. Aborting..."
#include "./elpa2_aborting_template.F90"
         endif

         if (useGPU) then
           num = (na) * size_of_real_datatype
           successGPU = gpu_memcpy(int(loc(ev(1)),kind=c_intptr_t), ev_dev, &
                   num, gpuMemcpyDeviceToHost)
           check_memcpy_gpu("elpa2_template ev_dev -> ev", successGPU)
         endif
         check_pd = 0
         do i = 1, na
           if (ev(i) .gt. thres_pd) then
             check_pd = check_pd + 1
           endif
         enddo
         if (check_pd .lt. na) then
           ! not positiv definite => eigenvectors needed
           do_trans_to_band = .true.
           do_trans_to_full = .true.
         else
           do_trans_to_band = .false.
           do_trans_to_full = .false.
         endif
       endif
     endif ! eigenvalues only

#if COMPLEXCASE == 1
     if (do_trans_to_band) then
       if (useGPU) then
#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         call GPU_COPY_REAL_PART_TO_Q_PRECISION_COMPLEX(q_dev, q_dev_real, matrixRows, l_rows, l_cols_nev, my_stream)
#else
         call GPU_COPY_REAL_PART_TO_Q_PRECISION_COMPLEX(q_dev, q_dev_real, matrixRows, l_rows, l_cols_nev)
#endif
       else !useGPU
         ! q must be given thats why from here on we can use q and not q_actual
         q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)
       endif
     endif


     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     !
     ! TODO check deallocation
     !! make sure q_real is deallocated when using check_pd
     !if (allocated(q_real)) then
     !  deallocate(q_real, stat=istat, errmsg=errorMessage)
     !  check_deallocate("elpa2_template: q_real", istat, errorMessage)
     !endif
#endif /* COMPLEXCASE == 1 */

     if (isSkewsymmetric) then
       if (useGPU) then
#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         call GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION_REAL(q_dev, matrixRows, matrixCols, my_stream)
#else
         call GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION_REAL(q_dev, matrixRows, matrixCols)
#endif
         do i = 1, matrixRows
           global_index = np_rows*nblk*((i-1)/nblk) + MOD(i-1,nblk) + MOD(np_rows+my_prow-0, np_rows)*nblk + 1
           if (mod(global_index-1,4) .eq. 0) then
             ! do nothing
           end if
           if (mod(global_index-1,4) .eq. 1) then
             negative_or_positive = 1
#ifdef WITH_GPU_STREAMS
             my_stream = obj%gpu_setup%my_stream
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive, my_stream)
#else
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive)
#endif 
           end if
           if (mod(global_index-1,4) .eq. 2) then
             negative_or_positive = -1
#ifdef WITH_GPU_STREAMS
             my_stream = obj%gpu_setup%my_stream
             call GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION_REAL(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive, my_stream)
#else
             call GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION_REAL(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive)
#endif
           end if
           if (mod(global_index-1,4) .eq. 3) then
             negative_or_positive = -1
#ifdef WITH_GPU_STREAMS
             my_stream = obj%gpu_setup%my_stream
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, i, matrixRows, matrixCols, &
                          negative_or_positive, my_stream)
#else
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, i, matrixRows, matrixCols, &
                          negative_or_positive)
#endif
           end if
         enddo
       else ! useGPU
         ! Extra transformation step for skew-symmetric matrix. Multiplication with diagonal complex matrix D.
         ! This makes the eigenvectors complex.
         ! For now real part of eigenvectors is generated in first half of q, imaginary part in second part.
         q(1:matrixRows, matrixCols+1:2*matrixCols) = 0.0
         do i = 1, matrixRows
!          global_index = indxl2g(i, nblk, my_prow, 0, np_rows)
           global_index = np_rows*nblk*((i-1)/nblk) + MOD(i-1,nblk) + MOD(np_rows+my_prow-0, np_rows)*nblk + 1
           if (mod(global_index-1,4) .eq. 0) then
              ! do nothing
           end if
           if (mod(global_index-1,4) .eq. 1) then
              q(i,matrixCols+1:2*matrixCols) = q(i,1:matrixCols)
              q(i,1:matrixCols) = 0
           end if
           if (mod(global_index-1,4) .eq. 2) then
              q(i,1:matrixCols) = -q(i,1:matrixCols)
           end if
           if (mod(global_index-1,4) .eq. 3) then
              q(i,matrixCols+1:2*matrixCols) = -q(i,1:matrixCols)
              q(i,1:matrixCols) = 0
           end if
         end do
       endif ! useGPU
     endif ! isSkew

!#ifdef WITH_SYCL_GPU_VERSION
!     if (obj%gpu_setup%syclCPU) then
!       print *,"Switching of the GPU trans_ev_tridi due to SYCL CPU",obj%gpu_setup%syclCPU
!       do_useGPU_trans_ev_tridi_to_band =.false.
!       kernel = DEFAULT_KERNEL
!       do_useGPU_trans_ev_band_to_full =.false.
!     endif
!#endif
       ! Backtransform stage 1
     if (do_trans_to_band) then

       !debug
!#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
       !if (gpu_vendor() == OPENMP_OFFLOAD_GPU .or. gpu_vendor() == SYCL_GPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION)
       if (gpu_vendor() == OPENMP_OFFLOAD_GPU) then
         if (do_useGPU_trans_ev_tridi_to_band) then
           do_useGPU_trans_ev_tridi_to_band = .false.
           kernel = DEFAULT_KERNEL
           write(error_unit,*) "Disabling GPU kernel for OPENMP_OFFLOAD_GPU"
         endif
       endif
#endif
       call obj%autotune_timer%start("tridi_to_band")
       call obj%timer%start("tridi_to_band")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("tridi_to_band")
#endif
       ! In the skew-symmetric case this transforms the real part
       if (do_useGPU_trans_ev_tridi_to_band) then
         call trans_ev_tridi_to_band_gpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, na, nev, nblk, nbw, q_dev, &
         matrixRows, matrixCols, hh_trans_dev, hh_trans_size, my_pe, mpi_comm_rows, mpi_comm_cols, &
         wantDebug, &
         nrThreads, success=success, kernel=kernel)
       else
!kernel = DEFAULT_KERNEL
         call trans_ev_tridi_to_band_cpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, na, nev, nblk, nbw, q, &
         matrixRows, matrixCols, hh_trans, hh_trans_size, my_pe, mpi_comm_rows, mpi_comm_cols, &
         wantDebug,  &
         nrThreads, success=success, kernel=kernel)
       endif
#ifdef HAVE_LIKWID
       call likwid_markerStopRegion("tridi_to_band")
#endif
       call obj%timer%stop("tridi_to_band")
       call obj%autotune_timer%stop("tridi_to_band")

       if (success) then
         success_int = 0
       else
         success_int = 1
       endif

#ifdef WITH_MPI
       if (useNonBlockingCollectivesAll) then
         call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
         allreduce_request5, mpierr)
         call mpi_wait(allreduce_request5, MPI_STATUS_IGNORE, mpierr)
       else
         call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       endif
#endif

       if (success_int .eq. 1) then
         write(error_unit,*) "ELPA2: trans_ev_tridi_to_band returned an error. Aborting..."
         return
       endif

     endif ! do_trans_to_band

     ! the array q (currently) always resides on host even when using GPU

     if (do_trans_to_full) then
       call obj%autotune_timer%start("band_to_full")
       call obj%timer%start("band_to_full")
#ifdef HAVE_LIKWID
       call likwid_markerStartRegion("band_to_full")
#endif

       ! Backtransform stage 2
       ! In the skew-symemtric case this transforms the real part
       if (do_useGPU_trans_ev_band_to_full) then
         call trans_ev_band_to_full_gpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, na, nev, nblk, nbw, a_dev, &
         matrixRows, tmat_dev, q_dev,  &
         matrixRows, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols &
#if REALCASE == 1
         , useQRActual  &
#endif
         , success)
       else ! do_useGPU_trans_ev_band_to_full
         call trans_ev_band_to_full_cpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, na, nev, nblk, nbw, a, &
         matrixRows, tmat, q,  &
         matrixRows, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols &
#if REALCASE == 1
         , useQRActual  &
#endif
         , success)
       endif ! do_useGPU_trans_ev_band_to_full
       call obj%timer%stop("band_to_full")
       call obj%autotune_timer%stop("band_to_full")

       if (success) then
         success_int = 0
       else
         success_int = 1
       endif

#ifdef WITH_MPI
       if (useNonBlockingCollectivesAll) then
         call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
         allreduce_request6, mpierr)
         call mpi_wait(allreduce_request6, MPI_STATUS_IGNORE, mpierr)
       else
         call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       endif
#endif
       if (success_int .eq. 1) then
         write(error_unit,*) "Error in trans_ev_band_to_full. Aborting..."
         return
       endif
     endif ! do_trans_to_full
! #ifdef DOUBLE_PRECISION_REAL
!        call prmat(na,useGPU,q(1:matrixRows, 1:matrixCols),q_dev,matrixRows,matrixCols,nblk,my_prow,my_pcol,np_rows,np_cols,'R',1)
! #endif


     !skew symmetric imaginary part for tridi_to_band and band_to_full
     if (isSkewsymmetric) then
       if (do_trans_to_band) then
         call obj%autotune_timer%start("tridi_to_band")
         call obj%timer%start("skew_tridi_to_band")
         ! Transform imaginary part
         ! Transformation of real and imaginary part could also be one call of trans_ev_tridi acting on the n x 2n matrix.
         if (do_useGPU_trans_ev_tridi_to_band) then
           num = matrixRows*matrixCols*size_of_datatype
           successGPU = gpu_malloc(q_part2_dev, num)
           check_alloc_gpu("elpa2_template q_dev", successGPU)
           
          ! copy q_part2(1:matrixRows,1:matrixCols) = q(1:matrixRows, matrixCols+1:2*matrixCols)
#ifdef WITH_GPU_STREAMS
           my_stream = obj%gpu_setup%my_stream 
           call GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, q_part2_dev, matrixRows, matrixCols, &
                                                                my_stream)
#else      
           call GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, q_part2_dev, matrixRows, matrixCols)
#endif 

           call trans_ev_tridi_to_band_gpu_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
           (obj, na, nev, nblk, nbw, q_part2_dev, &
           matrixRows, matrixCols, hh_trans_dev, hh_trans_size, my_pe, mpi_comm_rows, mpi_comm_cols, &
           wantDebug,  &
           nrThreads, success=success, kernel=kernel)
         else
           call trans_ev_tridi_to_band_cpu_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
           (obj, na, nev, nblk, nbw, q(1:matrixRows, matrixCols+1:2*matrixCols), &
           matrixRows, matrixCols, hh_trans, hh_trans_size, my_pe, mpi_comm_rows, mpi_comm_cols, &
           wantDebug, &
           nrThreads, success=success, kernel=kernel)
         endif

         call obj%timer%stop("skew_tridi_to_band")
         call obj%autotune_timer%stop("tridi_to_band")

         if (success) then
           success_int = 0
         else
           success_int = 1
         endif

#ifdef WITH_MPI
         if (useNonBlockingCollectivesAll) then
           call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
           allreduce_request7, mpierr)
           call mpi_wait(allreduce_request7, MPI_STATUS_IGNORE, mpierr)
         else
           call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
         endif
#endif
         if (success_int .eq. 1) then
          write(error_unit,*) "Error in trans_ev_tridi_to_band (imaginary part). Aborting..."
          return
         endif
       endif ! do_trans_tridi_to_band

       if (do_trans_to_full) then
         call obj%autotune_timer%start("band_to_full")
         call obj%timer%start("band_to_full")
         ! Transform imaginary part
         ! Transformation of real and imaginary part could also be one call of trans_ev_band_to_full_ acting on the n x 2n matrix.
         if (do_useGPU_trans_ev_band_to_full) then
           call trans_ev_band_to_full_gpu_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
           (obj, na, nev, nblk, nbw, a_dev, &
           matrixRows, tmat_dev, q_part2_dev,  &
           matrixRows, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
#if REALCASE == 1
           useQRActual, &
#endif
           success)
         else
           call trans_ev_band_to_full_cpu_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
           (obj, na, nev, nblk, nbw, a, &
           matrixRows, tmat, q(1:matrixRows, matrixCols+1:2*matrixCols),  &
           matrixRows, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
#if REALCASE == 1
           useQRActual, &
#endif
           success)
         endif

#ifdef HAVE_LIKWID
         call likwid_markerStopRegion("band_to_full")
#endif
         call obj%timer%stop("band_to_full")
         call obj%autotune_timer%stop("band_to_full")

         if (success) then
           success_int = 0
         else
           success_int = 1
         endif

#ifdef WITH_MPI
         if (useNonBlockingCollectivesAll) then
           call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
           allreduce_request8, mpierr)
           call mpi_wait(allreduce_request8, MPI_STATUS_IGNORE, mpierr)
         else
           call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
         endif
#endif
         if (success_int .eq. 1) then
          write(error_unit,*) "Error in trans_ev_band_to_full (imaginary part). Aborting..."
          return
         endif
       endif ! isSkewSymmetric

       if (isSkewsymmetric) then
         if (useGPU) then
#ifdef WITH_GPU_STREAMS
           my_stream = obj%gpu_setup%my_stream
           call GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, q_part2_dev, matrixRows, matrixCols, &
                                                                 my_stream)
#else
           call GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, q_part2_dev, matrixRows, matrixCols)
#endif
         endif ! useGPU
       endif ! isSKewSymmetric

#ifndef DEVICE_POINTER
      if (useGPU) then
        ! copy back
        if (isSkewsymmetric) then
          num = (matrixRows* 2*matrixCols) * size_of_datatype
        else
          num = (matrixRows* matrixCols) * size_of_datatype
        endif
        successGPU = gpu_memcpy(int(loc(q(1,1)),kind=c_intptr_t), q_dev, &
                    num, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa2_template q_dev -> q", successGPU)
      endif ! useGPU
#endif /* DEVICE_POINTER */

      if (isSkewsymmetric) then
        if (useGPU) then
          successGPU = gpu_free(q_part2_dev)
          check_dealloc_gpu("elpa2_template q_part2_dev", successGPU)
        endif
      endif
    endif !do_full_to_band


#ifndef DEVICE_POINTER
   ! copy back ev in anycase
   if (useGPU) then
      num = (na) * size_of_real_datatype
      successGPU = gpu_memcpy(int(loc(ev(1)),kind=c_intptr_t), ev_dev, &
                 num, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("elpa2_template ev_dev -> ev", successGPU)
   endif
#endif /* DEVICE_POINTER */



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!
! CHECK DEALLOCATES
!     ! We can now deallocate the stored householder vectors
!     deallocate(hh_trans, stat=istat, errmsg=errorMessage)
!     check_deallocate("elpa2_template: hh_trans", istat, errorMessage)
!
!     ! make sure tmat is deallocated when using check_pd
!     if (allocated(tmat)) then
!       deallocate(tmat, stat=istat, errmsg=errorMessage)
!       check_deallocate("elpa2_template: tmat", istat, errorMessage)
!     endif
!
!     if (obj%eigenvalues_only) then
!       deallocate(q_dummy, stat=istat, errmsg=errorMessage)
!       check_deallocate("elpa2_template: q_dummy", istat, errorMessage)
!     endif

     ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
    call omp_set_num_threads(omp_threads_caller)
#endif

#ifdef REDISTRIBUTE_MATRIX
#include "../helpers/elpa_redistribute_back_template.F90"
#endif /* REDISTRIBUTE_MATRIX */


#ifndef DEVICE_POINTER
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! no device pointer
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   deallocate(e, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa2_template: e, ", istat, errorMessage)

   deallocate(hh_trans, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa2_template: hh_trans", istat, errorMessage)

   if (do_bandred) then
     ! tmat is needed only in full->band and band->full steps, so alocate here
     ! (not allocated for banded matrix on input)
     deallocate(tmat, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa2_template: tmat", istat, errorMessage)
   endif
#ifndef REDISTRIBUTE_MATRIX
   nullify(aIntern)
   nullify(a)
   if (present(qExtern)) then
     nullify(qIntern)
     nullify(q)
   endif

   nullify(evIntern)
   nullify(ev)

   ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
   if (.not.(obj%eigenvalues_only)) then
   else
     deallocate(q_dummy, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa2_template: q_dummy", istat, errorMessage)
   endif
   nullify(q_actual)

#if COMPLEXCASE == 1
   deallocate(q_real, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa2_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */

   if (doRedistributeMatrix) then
     nullify(a)
     if (present(qExtern)) then
       nullify(q)
     endif
     ! ev never changes
     nullify(evIntern)
     nullify(ev)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       deallocate(q_dummy, stat=istat, errmsg=errorMessage)
       check_allocate("elpa2_template: q_dummy", istat, errorMessage)
     endif
     nullify(q_actual)

#if COMPLEXCASE == 1
     deallocate(q_real, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa2_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     nullify(aIntern)
     nullify(a)
     if (present(qExtern)) then
       nullify(qIntern)
       nullify(q)
     endif
     nullify(evIntern)
     nullify(ev)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       deallocate(q_dummy, stat=istat, errmsg=errorMessage)
       check_deallocate("elpa2_template: q_dummy", istat, errorMessage)
     endif
     nullify(q_actual)

#if COMPLEXCASE == 1
     deallocate(q_real, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa2_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */

   if (useGPU) then
     successGPU = gpu_free(e_dev)
     check_dealloc_gpu("elpa2_template e_dev", successGPU)

     if (do_bandred) then
       successGPU = gpu_free(tmat_dev)
       check_dealloc_gpu("elpa2_template tmat_devIntern", successGPU)
     endif

     successGPU = gpu_free(hh_trans_dev)
     check_dealloc_gpu("elpa2_template hh_trans_dev", successGPU)


#ifndef REDISTRIBUTE_MATRIX
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     successGPU = gpu_free(a_devIntern)
     check_dealloc_gpu("elpa2_template a_devIntern", successGPU)

     successGPU = gpu_free(ev_devIntern)
     check_dealloc_gpu("elpa2_template ev_devIntern", successGPU)

     if (present(qExtern)) then
       successGPU = gpu_free(q_devIntern)
       check_dealloc_gpu("elpa2_template q_devIntern", successGPU)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa2_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */


   if (doRedistributeMatrix) then

     successGPU = gpu_free(ev_devIntern)
     check_dealloc_gpu("elpa2_template ev_devIntern", successGPU)
     deallocate(aIntern, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa2_template: aIntern", istat, errorMessage)
     deallocate(qIntern, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa2_template: qIntern", istat, errorMessage)

     nullify(q)
     nullify(a)


     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa2_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     successGPU = gpu_free(a_devIntern)
     check_dealloc_gpu("elpa2_template a_devIntern", successGPU)

     successGPU = gpu_free(ev_devIntern)
     check_dealloc_gpu("elpa2_template ev_devIntern", successGPU)
     if (present(qExtern)) then
       successGPU = gpu_free(q_devIntern)
       check_dealloc_gpu("elpa2_template q_devIntern", successGPU)
     endif


     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa2_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU
#else /* DEVICE_POINTER */
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   ! DEVICE POINTER
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   deallocate(evIntern, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa2_template: evIntern", istat, errorMessage)
   nullify(ev)
   deallocate(e, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa2_template: e", istat, errorMessage)

   if (useGPU) then
     successGPU = gpu_free(e_dev)
     check_dealloc_gpu("elpa2_template e_dev", successGPU)

     if (do_bandred) then
       successGPU = gpu_free(tmat_dev)
       check_dealloc_gpu("elpa2_template tmat_devIntern", successGPU)
     endif

     successGPU = gpu_free(hh_trans_dev)
     check_dealloc_gpu("elpa2_template hh_trans_dev", successGPU)

#ifndef REDISTRIBUTE_MATRIX
     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa2_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */
     if (doRedistributeMatrix) then
       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
       else
         successGPU = gpu_free(q_dev_dummy)
         check_dealloc_gpu("elpa2_template q_dev_dummy", successGPU)
       endif

#if COMPLEXCASE == 1
       successGPU = gpu_free(q_dev_real)
       check_dealloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

     else ! doRedistributeMatrix
       ! associate a_dev, q_dev, ev_dev

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
       else
         successGPU = gpu_free(q_dev_dummy)
         check_dealloc_gpu("elpa2_template q_dev_dummy", successGPU)
       endif

#if COMPLEXCASE == 1
       successGPU = gpu_free(q_dev_real)
       check_dealloc_gpu("elpa2_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
     endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU

#endif /* DEVICE_POINTER */

#ifdef ACTIVATE_SKEW
     call obj%timer%stop("elpa_solve_skew_evp_&
#else
     call obj%timer%stop("elpa_solve_evp_&
#endif
     &MATH_DATATYPE&
     &_2stage_&
     &PRECISION&
     &")
1    format(a,f10.3)

   end function

! vim: syntax=fortran
