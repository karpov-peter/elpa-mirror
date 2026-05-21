#ifdef REALCASE
#ifdef COMPLEXCASE
#error Cannot define both REALCASE and COMPLEXCASE
#endif
#endif

#ifndef REALCASE
#ifndef COMPLEXCASE
#error Must define one of REALCASE or COMPLEXCASE
#endif
#endif

#ifdef SINGLE_PRECISION
#ifdef DOUBLE_PRECISION
#error Cannot define both SINGLE_PRECISION and DOUBLE_PRECISION
#endif
#ifdef HALF_PRECISION
#error Cannot define both SINGLE_PRECISION and HALF_PRECISION
#endif
#endif

#ifdef DOUBLE_PRECISION
#ifdef HALF_PRECISION
#error Cannot define both DOUBLE_PRECISION and HALF_PRECISION
#endif
#endif

#ifndef SINGLE_PRECISION
#ifndef DOUBLE_PRECISION
#ifndef HALF_PRECISION
#error Must define one of SINGLE_PRECISION, DOUBLE_PRECISION, or HALF_PRECISION
#endif
#endif
#endif
