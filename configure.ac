#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT(CalibRig, 0.1, cyril.pichard@xxxxxx.xxx)
#AC_CONFIG_SRCDIR([src/main.cpp])
#AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE

AC_SEARCH_LIBS(glutInit, glut)
AC_SEARCH_LIBS(X11, X11)
AC_SEARCH_LIBS(XNVCtrl, XNVCtrl)

PKG_CHECK_MODULES(OpenCV, opencv)

# Checks for programs.
AC_PROG_CXX
AC_PROG_MAKE_SET

AC_ARG_WITH([nvsdi-include-path],
  [AS_HELP_STRING([--with-nvsdi-include-path],
    [location of the NVSDI headers, defaults to /home/cyril/Installs/SDIsdk3.5/linux/include])],
  [SDI_CFLAGS="-I$withval"],
  [SDI_CFLAGS='-I/home/cyril/Installs/SDIsdk3.5/sdk/linux/include'])
AC_SUBST([SDI_CFLAGS])

AC_ARG_WITH([nvsdi-lib-path],
  [AS_HELP_STRING([--with-nvsdi-include-path],
    [location of the NVSDI libs, defaults to /home/cyril/Installs/sdk/linux/lib32])],
  [SDI_LIBS="-L$withvali -lXNVCtrl"],
  [SDI_LIBS='-L/home/cyril/Installs/SDIsdk3.5/sdk/linux/lib32 -lXNVCtrl'])
AC_SUBST([SDI_LIBS])

AC_ARG_WITH([cuda-include-path],
  [AS_HELP_STRING([--with-cuda-include-path],
    [location of the NVSDI headers, defaults to /usr/local/cuda/include])],
  [CUDA_CFLAGS="-I$withval"],
  [CUDA_CFLAGS='-I/usr/local/cuda/include'])
AC_SUBST([CUDA_CFLAGS])

AC_ARG_WITH([cuda-lib-path],
  [AS_HELP_STRING([--with-cuda-include-path],
    [location of the NVSDI libs, defaults to /usr/local/cuda/lib32])],
  [CUDA_LIBS="-L$withvali"],
  [CUDA_LIBS='-L/usr/local/cuda/lib -lcuda -lcudart'])
AC_SUBST([CUDA_LIBS])

AC_ARG_WITH([cuda-toolkit-include-path],
  [AS_HELP_STRING([--with-cuda-toolkit-include-path],
    [location of the NVSDI headers, defaults to /usr/local/cuda/include])],
  [CUDATOOLKIT_CFLAGS="-I$withval"],
  [CUDATOOLKIT_CFLAGS='-I/home/cyril/Installs/NVIDIA_GPU_Computing_SDK/C/common/inc'])
AC_SUBST([CUDATOOLKIT_CFLAGS])

AC_ARG_WITH([cuda-toolkit-lib-path],
  [AS_HELP_STRING([--with-cuda-toolkit-lib-path],
    [ defaults to ])],
  [CUDATOOLKIT_LIBS="-L$withvali"],
  [CUDATOOLKIT_LIBS='-L/home/cyril/Installs/NVIDIA_GPU_Computing_SDK/C/common/lib/linux'])
AC_SUBST([CUDATOOLKIT_LIBS])


AC_ARG_WITH([cudpp-toolkit-include-path],
  [AS_HELP_STRING([--with-cudpp-toolkit-include-path],
    [location of the NVSDI headers, defaults to /usr/local/cudpp/include])],
  [CUDPPTOOLKIT_CFLAGS="-I$withval"],
  [CUDPPTOOLKIT_CFLAGS='-I/home/cyril/Installs/cudpp_src_1.1.1/cudpp/include'])
AC_SUBST([CUDPPTOOLKIT_CFLAGS])

AC_ARG_WITH([cudpp-toolkit-lib-path],
  [AS_HELP_STRING([--with-cudpp-toolkit-lib-path],
    [ defaults to ])],
  [CUDPPTOOLKIT_LIBS="-L$withvali"],
  [CUDPPTOOLKIT_LIBS='/home/cyril/Installs/cudpp_src_1.1.1/lib/libcudpp_i686.a '])
AC_SUBST([CUDPPTOOLKIT_LIBS])


AC_ARG_WITH([boost-include-path],
  [AS_HELP_STRING([--with-boost-include-path],
    [location of the boost libraries])],
  [BOOST_CFLAGS="-I$withval"],
  [BOOST_CFLAGS='-I/home/cyril/Installs/boost_1_45_0/'])
AC_SUBST([BOOST_CFLAGS])

AC_ARG_WITH([boost-lib-path],
  [AS_HELP_STRING([--with-boost-lib-path],
    [location of the boost libs])],
  [BOOST_LIBS="-L$withvali"],
  [BOOST_LIBS='-L/usr/local/lib/ -lboost_thread -lboost_program_options -lboost_system'])
AC_SUBST([BOOST_LIBS])

AC_ARG_WITH([tbb-lib-path],
  [AS_HELP_STRING([--with-tbb-include-path],
    [location of the tbb libs])],
  [TBB_LIBS="-L$withvali"],
  [TBB_LIBS='-L/home/cyril/Installs/tbb22_20090809oss/build/linux_ia32_gcc_cc4.3.2_libc2.7_kernel2.6.26_release/ -ltbb'])
AC_SUBST([TBB_LIBS])



AC_CONFIG_FILES([
	Makefile
	src/Makefile
	tests/debug-cuda/Makefile
	])

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AC_OUTPUT
