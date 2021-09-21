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

#include "../general/sanity.F90"
#include "../general/error_checking.inc"
#include "config-f90.h"

  use, intrinsic  :: iso_c_binding
  use precision
  use elpa1_compute
  use elpa_utilities
  use elpa_mpi
  use elpa_abstract_impl
  use elpa_blas_interfaces
  use elpa_gpu
  use cuda_functions


  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)             :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)     :: a(obj%local_nrows,*)
#else
  MATH_DATATYPE(kind=rck)     :: a(obj%local_nrows,obj%local_ncols)
#endif

  integer(kind=ik)             :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)       :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=ik)             :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
  integer(kind=ik)             :: n, nc, i, info, ns, nb
  integer(kind=BLAS_KIND)      :: infoBLAS
  MATH_DATATYPE(kind=rck), allocatable   :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)
  logical                      :: wantDebug
  logical                      :: success
  integer(kind=ik)             :: istat, debug, error
  character(200)               :: errorMessage
!Soheil
  integer(kind=c_intptr_t)                      :: tmat1_dev, tmat2_dev, a_dev
  logical                                       :: successGPU, useIntelGPU
  logical, parameter                            :: useGPU = .TRUE.   ! later must be passed in as inp. arg.
  integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
  character(20)                                 :: gpuString
  integer(kind=c_int)                           :: gpuID
  integer(kind=ik)                              :: max_l_rows, max_l_cols, max_nblk, my_glob_rank, dev_cnt

  useIntelGPU = .false.  
  if(useGPU) then
    gpuString = "_gpu"

    if (gpu_vendor() == INTEL_GPU) then
      useIntelGPU = .true.
    endif
  else
    gpuString = ""
  endif

!END Soheil

  call obj%timer%start("elpa_invert_trm_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &" // gpuString)   !Soheil: added // gpuString

  na         = obj%na
  matrixRows = obj%local_nrows
  nblk       = obj%nblk
  matrixCols = obj%local_ncols

  call obj%get("mpi_comm_rows",mpi_comm_rows,error)
  if (error .ne. ELPA_OK) then
    print *,"Error getting option for mpi_comm_rows. Aborting..."
    stop
  endif
  call obj%get("mpi_comm_cols",mpi_comm_cols,error)
  if (error .ne. ELPA_OK) then
    print *,"Error getting option for mpi_comm_cols. Aborting..."
    stop
  endif

  call obj%get("debug", debug,error)
  if (error .ne. ELPA_OK) then
    print *,"Error getting option for debug. Aborting..."
    stop
  endif
  if (debug == 1) then
    wantDebug = .true.
  else
    wantDebug = .true.
  endif
  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

  my_prow = int(my_prowMPI,kind=c_int)
  np_rows = int(np_rowsMPI,kind=c_int)
  my_pcol = int(my_pcolMPI,kind=c_int)
  np_cols = int(np_colsMPI,kind=c_int)
!Soheil
  call mpi_comm_rank(MPI_COMM_WORLD, my_glob_rank, mpierr)
  if (my_glob_rank == 0)   print *,'Using GPU: ', useGPU

  call obj%timer%stop("mpi_communication")
  success = .true.

  if(useGPU) then
    call set_gpu_parameters()
    dev_cnt = 1   ! 1..4
    gpuID = mod(my_glob_rank, dev_cnt) 
    success = cuda_setdevice(gpuID)
    if (.not.(success)) then
      print *,"invert_trm: Cannot set GPU device. Aborting..."
      stop
    endif

    success = cublas_create(cublasHandle)
    if (.not.(success)) print *, 'Device initialization failed.'

  endif

  l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
  l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a
!Soheil:
  max_l_rows = max(l_rows,1)
  max_l_cols = max(l_cols,1)
  max_nblk   = max(nblk,1)

  allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmp1", istat, errorMessage)

  allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmp2", istat, errorMessage)

  tmp1 = 0
  tmp2 = 0

  allocate(tmat1(l_rows,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmat1", istat, errorMessage)

  allocate(tmat2(nblk,l_cols), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmat2", istat, errorMessage)

  tmat1 = 0
  tmat2 = 0
!Soheil
if (useGPU) then
  successGPU = cuda_malloc(a_dev, matrixRows*matrixCols*size_of_datatype)
  check_alloc_gpu("elpa_invert_trm: a_dev", successGPU)

  successGPU = cuda_malloc(tmat1_dev, max_l_rows*max_nblk*size_of_datatype)  
  check_alloc_gpu("elpa_invert_trm: tmat1_dev", successGPU)

  successGPU = cuda_malloc(tmat2_dev, max_l_cols*max_nblk*size_of_datatype)
  check_alloc_gpu("elpa_invert_trm: tmat2_dev", successGPU)
end if

  ns = ((na-1)/nblk)*nblk + 1

  do n = ns,1,-nblk

    l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
    l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

    nb = nblk
    if (na-n+1 < nblk) nb = na-n+1

    l_rowx = local_index(n+nb, my_prow, np_rows, nblk, +1)
    l_colx = local_index(n+nb, my_pcol, np_cols, nblk, +1)

    if (my_prow==prow(n, nblk, np_rows)) then

      if (my_pcol==pcol(n, nblk, np_cols)) then
        call obj%timer%start("blas")
        call PRECISION_TRTRI('U', 'N', int(nb,kind=BLAS_KIND), a(l_row1,l_col1), int(matrixRows,kind=BLAS_KIND), &
                             infoBLAS)
        info = int(infoBLAS,kind=ik)
        call obj%timer%stop("blas")

        if (info/=0) then
          if (wantDebug) write(error_unit,*) "elpa_invert_trm_&
          &MATH_DATATYPE&

#if REALCASE == 1
          &: Error in DTRTRI"
#endif
#if COMPLEXCASE == 1
          &: Error in ZTRTRI"
#endif

          success = .false.
          call obj%timer%stop("elpa_invert_trm_&
          &MATH_DATATYPE&
          &_&
          &PRECISION&
          &")
          return
        endif

        nc = 0
        do i=1,nb
          tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
          nc = nc+i
        enddo
      endif
#ifdef WITH_MPI
      call obj%timer%start("mpi_communication")
      call MPI_Bcast(tmp1, int(nb*(nb+1)/2,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,       &
                     int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
      call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
      nc = 0
      do i=1,nb
        tmp2(1:i,i) = tmp1(nc+1:nc+i)
        nc = nc+i
      enddo

      call obj%timer%start("blas")
      if (l_cols-l_colx+1>0) &
      call PRECISION_TRMM('L', 'U', 'N', 'N', int(nb,kind=BLAS_KIND), int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, &
                              tmp2, int(ubound(tmp2,dim=1),kind=BLAS_KIND), a(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND))
      call obj%timer%stop("blas")
      if (l_colx<=l_cols)   tmat2(1:nb,l_colx:l_cols) = a(l_row1:l_row1+nb-1,l_colx:l_cols)
      if (my_pcol==pcol(n, nblk, np_cols)) tmat2(1:nb,l_col1:l_col1+nb-1) = tmp2(1:nb,1:nb) ! tmp2 has the lower left triangle 0

    endif

    if (l_row1>1) then
      if (my_pcol==pcol(n, nblk, np_cols)) then
        tmat1(1:l_row1-1,1:nb) = a(1:l_row1-1,l_col1:l_col1+nb-1)
        a(1:l_row1-1,l_col1:l_col1+nb-1) = 0
      endif

      do i=1,nb
#ifdef WITH_MPI
        call obj%timer%start("mpi_communication")
        call MPI_Bcast(tmat1(1,i), int(l_row1-1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                       int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

        call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
      enddo
    endif
#ifdef WITH_MPI
    call obj%timer%start("mpi_communication")
    if (l_cols-l_col1+1>0) &
    call MPI_Bcast(tmat2(1,l_col1), int((l_cols-l_col1+1)*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                   int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

    call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

!Soheil
    if (l_row1>1 .and. l_cols-l_col1+1>0) then
      if (useGPU .and. .not. useIntelGPU) then 
          successGPU = cuda_memcpy(tmat1_dev, int(loc(tmat1), kind=c_intptr_t), & 
                                  l_rows*nblk*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_invert_trm: tmat1_dev", successGPU)

          successGPU = cuda_memcpy(tmat2_dev, int(loc(tmat2), kind=c_intptr_t), & 
                                  l_cols*nblk*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_invert_trm: tmat2_dev", successGPU)

          successGPU = cuda_memcpy(a_dev, int(loc(a), kind=c_intptr_t), & 
                          matrixCols*matrixRows*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_invert_trm: a_dev", successGPU)

          call cublas_PRECISION_GEMM('N', 'N', (l_row1-1), (l_cols-l_col1+1), nb, -ONE, &
                           tmat1_dev, max_l_rows, & 
                           (tmat2_dev+((l_col1-1)*nblk*size_of_datatype)), max_nblk, ONE, &
                           (a_dev+((l_col1-1)*matrixRows*size_of_datatype)), matrixRows )

          successGPU = cuda_memcpy(int(loc(tmat1), kind=c_intptr_t), tmat1_dev, & 
                                  l_rows*nblk*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_invert_trm: tmat1_dev back copy", successGPU)

          successGPU = cuda_memcpy(int(loc(tmat2), kind=c_intptr_t), tmat2_dev,  & 
                                  l_cols*nblk*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_invert_trm: tmat2_dev back copy", successGPU)

          successGPU = cuda_memcpy(int(loc(a), kind=c_intptr_t), a_dev,  & 
                          matrixCols*matrixRows*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_invert_trm: a_dev back copy", successGPU)
      else
         call obj%timer%start("blas")
          call PRECISION_GEMM('N', 'N', int(l_row1-1,kind=BLAS_KIND), int(l_cols-l_col1+1,kind=BLAS_KIND), &
                          int(nb,kind=BLAS_KIND), -ONE, &
                           tmat1, int(ubound(tmat1,dim=1),kind=BLAS_KIND), tmat2(1,l_col1), &
                           int(ubound(tmat2,dim=1),kind=BLAS_KIND), ONE, &
                            a(1,l_col1), int(matrixRows,kind=BLAS_KIND) )
         call obj%timer%stop("blas")
      end if  ! useGPU
    end if   ! l_row1>1 .and. l_cols-l_col1+1>0
!END Soheil
  enddo

  deallocate(tmp1, tmp2, tmat1, tmat2, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_invert_trm: tmp1, tmp2, tmat1, tmat2", istat, errorMessage)

!Soheil: free up GPU memory
  if (useGPU) then
      successGPU = cuda_free(a_dev)
      check_dealloc_gpu("invert_trm: a_dev ", successGPU)

      successGPU = cuda_free(tmat1_dev)
      check_dealloc_gpu("invert_trm: tmat1_dev ", successGPU)

      successGPU = cuda_free(tmat2_dev)
      check_dealloc_gpu("invert_trm: tmat2_dev ", successGPU)
  end if

  call obj%timer%stop("elpa_invert_trm_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &" // gpuString)
