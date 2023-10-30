
#include "config-f90.h"
module solve_tridi
  use precision
  implicit none
  private

  public :: solve_tridi_double
  public :: solve_tridi_dptr_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: solve_tridi_single
  public :: solve_tridi_dptr_single
#endif

  contains

! real double precision first
#undef DEVICE_POINTER
#define DOUBLE_PRECISION_REAL
#define REALCASE
#define DOUBLE_PRECISION
#define PRECISION_AND_SUFFIX double
#include "../general/precision_macros.h"
#include "./solve_tridi_template.F90"
#undef DOUBLE_PRECISION_REAL
#undef REALCASE
#undef DOUBLE_PRECISION
#undef PRECISION_AND_SUFFIX

! real double precision DEVICE_POINTER
#define DEVICE_POINTER
#define DOUBLE_PRECISION_REAL
#define REALCASE
#define DOUBLE_PRECISION
#define PRECISION_AND_SUFFIX double
#include "../general/precision_macros.h"
#include "./solve_tridi_template.F90"
#undef DOUBLE_PRECISION_REAL
#undef REALCASE
#undef DOUBLE_PRECISION
#undef PRECISION_AND_SUFFIX
#undef DEVICE_POINTER


#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#undef DEVICE_POINTER
#define SINGLE_PRECISION_REAL
#define REALCASE
#define SINGLE_PRECISION
#define PRECISION_AND_SUFFIX single
#include "../general/precision_macros.h"
#include "./solve_tridi_template.F90"
#undef SINGLE_PRECISION_REAL
#undef REALCASE
#undef SINGLE_PRECISION
#undef PRECISION_AND_SUFFIX

! real single precision DEVICE_POINTER
#define DEVICE_POINTER
#define SINGLE_PRECISION_REAL
#define REALCASE
#define SINGLE_PRECISION
#define PRECISION_AND_SUFFIX single
#include "../general/precision_macros.h"
#include "./solve_tridi_template.F90"
#undef SINGLE_PRECISION_REAL
#undef REALCASE
#undef SINGLE_PRECISION
#undef PRECISION_AND_SUFFIX
#undef DEVICE_POINTER

#endif

end module
