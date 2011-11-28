NVCC = nvcc
SUFFIXES = .cu
.cu.$(OBJEXT):
	$(NVCC) -g -G -Xcompiler -DBOOST_UNORDERED_NO_HAS_MOVE_ASSIGN -arch=sm_23 -I../../src/ -I./ @BOOST_CFLAGS@ @OpenCV_CFLAGS@ @CUDATOOLKIT_CFLAGS@ @CUDPPTOOLKIT_CFLAGS@ -c -o  $@ $<

AM_CPPFLAGS = -g -I../../src @SDI_CFLAGS@ @OpenCV_CFLAGS@ @CUDA_CFLAGS@ @CUDATOOLKIT_CFLAGS@ @CUDPPTOOLKIT_CFLAGS@ @BOOST_CFLAGS@ -DGL_GLEXT_PROTOTYPES -DGLX_GLXEXT_PROTOTYPES

bin_PROGRAMS=debugcuda
debugcuda_LDADD = @SDI_LIBS@ @OpenCV_LIBS@ @CUDA_LIBS@ @CUDATOOLKIT_LIBS@ @CUDPPTOOLKIT_LIBS@ @BOOST_LIBS@ @TBB_LIBS@
debugcuda_LDFLAGS = -Wl,-rpath,./ 
debugcuda_SOURCES = \
main.cpp \
../../src/CudaUtils.h \
../../src/CudaUtils.cpp\
../../src/ImageGL.h \
../../src/ImageGL.cpp \
../../src/Deformation.h\
../../src/SurfGPUData.h\
../../src/SurfNonMax.h\
../../src/SurfNonMax.cu\
../../src/SurfHessian.h\
../../src/SurfHessian.cu\
../../src/SurfDescriptor.h\
../../src/SurfDescriptor.cu\
../../src/SurfCollectPoints.h\
../../src/SurfCollectPoints.cu\
../../src/SurfMatching.h\
../../src/SurfMatching.cu\
../../src/SurfMatchingC.cpp\
../../src/ImageProcessing.h\
../../src/ImageProcessing.cpp\
../../src/CudaImageProcessing.h\
../../src/CudaImageProcessing.cu\
../../src/SurfMatchingExperimental.h\
../../src/SurfMatchingExperimental.cu

