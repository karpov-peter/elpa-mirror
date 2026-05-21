!    Copyright 2026, MPCDF
!
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
! Author: Andreas Marek, MPCDF

#ifdef WANT_HALF_PRECISION_REAL

  interface
    subroutine cuda_copy_double_to_half_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_double_to_half_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_to_half_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_double_to_half_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half_to_double_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half_to_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half_to_double_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half_to_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_to_half_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_float_to_half_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_to_half_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_float_to_half_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half_to_float_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half_to_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half_to_float_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half_to_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  ! Generic interfaces (dispatch on pointer type)

  interface cuda_double_to_half
    module procedure cuda_double_to_half_intptr
    module procedure cuda_double_to_half_cptr
  end interface

  interface cuda_half_to_double
    module procedure cuda_half_to_double_intptr
    module procedure cuda_half_to_double_cptr
  end interface

  interface cuda_float_to_half
    module procedure cuda_float_to_half_intptr
    module procedure cuda_float_to_half_cptr
  end interface

  interface cuda_half_to_float
    module procedure cuda_half_to_float_intptr
    module procedure cuda_half_to_float_cptr
  end interface

#endif /* WANT_HALF_PRECISION_REAL */

#ifdef WANT_HALF_PRECISION_COMPLEX

  interface
    subroutine cuda_copy_double_complex_to_half2_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_double_complex_to_half2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_complex_to_half2_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_double_complex_to_half2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half2_to_double_complex_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half2_to_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half2_to_double_complex_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half2_to_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_single_complex_to_half2_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_single_complex_to_half2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_single_complex_to_half2_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_single_complex_to_half2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half2_to_single_complex_intptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half2_to_single_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: src
      integer(kind=C_intptr_T), value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_half2_to_single_complex_cptr_c(src, dst, n, my_stream) &
               bind(C, name="cuda_copy_half2_to_single_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),              value  :: src
      type(c_ptr),              value  :: dst
      integer(kind=C_INT),      value  :: n
      integer(kind=C_intptr_T), value  :: my_stream
    end subroutine
  end interface

  ! Generic interfaces

  interface cuda_double_complex_to_half2
    module procedure cuda_double_complex_to_half2_intptr
    module procedure cuda_double_complex_to_half2_cptr
  end interface

  interface cuda_half2_to_double_complex
    module procedure cuda_half2_to_double_complex_intptr
    module procedure cuda_half2_to_double_complex_cptr
  end interface

  interface cuda_single_complex_to_half2
    module procedure cuda_single_complex_to_half2_intptr
    module procedure cuda_single_complex_to_half2_cptr
  end interface

  interface cuda_half2_to_single_complex
    module procedure cuda_half2_to_single_complex_intptr
    module procedure cuda_half2_to_single_complex_cptr
  end interface

#endif /* WANT_HALF_PRECISION_COMPLEX */

  contains

! =========================================================================
! Thin Fortran wrappers
! =========================================================================

#ifdef WANT_HALF_PRECISION_REAL

  subroutine cuda_double_to_half_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_double_to_half_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_double_to_half_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_double_to_half_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half_to_double_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half_to_double_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half_to_double_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half_to_double_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_float_to_half_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_float_to_half_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_float_to_half_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_float_to_half_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half_to_float_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half_to_float_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half_to_float_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half_to_float_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

#endif /* WANT_HALF_PRECISION_REAL */

#ifdef WANT_HALF_PRECISION_COMPLEX

  subroutine cuda_double_complex_to_half2_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_double_complex_to_half2_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_double_complex_to_half2_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_double_complex_to_half2_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half2_to_double_complex_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half2_to_double_complex_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half2_to_double_complex_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half2_to_double_complex_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_single_complex_to_half2_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_single_complex_to_half2_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_single_complex_to_half2_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_single_complex_to_half2_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half2_to_single_complex_intptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=C_intptr_T) :: src, dst, my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half2_to_single_complex_intptr_c(src, dst, n, my_stream)
#endif
  end subroutine

  subroutine cuda_half2_to_single_complex_cptr(src, dst, n, my_stream)
    use, intrinsic :: iso_c_binding
    implicit none
    type(c_ptr)              :: src, dst
    integer(kind=C_intptr_T) :: my_stream
    integer(kind=C_INT)      :: n
#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_half2_to_single_complex_cptr_c(src, dst, n, my_stream)
#endif
  end subroutine

#endif /* WANT_HALF_PRECISION_COMPLEX */
