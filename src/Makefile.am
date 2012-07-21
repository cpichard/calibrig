NVCC = nvcc
SUFFIXES = .cu
.cu.$(OBJEXT):
	$(NVCC) -g -G -Xcompiler -DBOOST_UNORDERED_NO_HAS_MOVE_ASSIGN -arch=sm_23 -I./ @BOOST_CFLAGS@ @OpenCV_CFLAGS@ @CUDATOOLKIT_CFLAGS@ @CUDPPTOOLKIT_CFLAGS@ -c -o  $@ $<

AM_CPPFLAGS = -g @SDI_CFLAGS@ @OpenCV_CFLAGS@ @CUDA_CFLAGS@ @CUDATOOLKIT_CFLAGS@ @CUDPPTOOLKIT_CFLAGS@ @BOOST_CFLAGS@ -DGL_GLEXT_PROTOTYPES -DGLX_GLXEXT_PROTOTYPES

bin_PROGRAMS=calibrig
calibrig_LDADD = @SDI_LIBS@ @OpenCV_LIBS@ @CUDA_LIBS@ @CUDATOOLKIT_LIBS@ @CUDPPTOOLKIT_LIBS@ @BOOST_LIBS@ @TBB_LIBS@
calibrig_LDFLAGS = -Wl,-rpath,./
calibrig_SOURCES = \
main.cpp \
NvSDIin.h NvSDIin.cpp \
NvSDIutils.h NvSDIutils.cpp \
Monitor.h Monitor.cpp\
StereoAnalyzer.h StereoAnalyzer.cpp\
StereoAnalyzerCPU.h StereoAnalyzerCPU.cpp\
StereoAnalyzerGPU.h StereoAnalyzerGPU.cpp\
ScreenLayout.h ScreenLayout.cpp\
QuadViewScreen.h QuadViewScreen.cpp\
DiffScreen.h DiffScreen.cpp\
CudaUtils.h CudaUtils.cpp\
Utils.h Utils.cpp\
ImageGL.h ImageGL.cpp\
Grabber.h Grabber.cpp\
MainWindow.h MainWindow.cpp\
AnalysisResult.h\
AnalysisResult.cpp\
AnalyzerFunctor.h\
AnalyzerFunctorCPU.h\
NetworkServer.h\
NetworkServer.cpp\
ComputationData.h\
ComputationData.cpp\
ComputationDataCPU.h\
ComputationDataCPU.cpp\
ComputationDataGPU.h\
ComputationDataGPU.cpp\
SurfGPUData.h\
SurfNonMax.h\
SurfNonMax.cu\
SurfHessian.h\
SurfHessian.cu\
SurfDescriptor.h\
SurfDescriptor.cu\
SurfCollectPoints.h\
SurfCollectPoints.cu\
SurfMatching.h\
SurfMatching.cu\
SurfMatchingC.cpp\
VertexBufferObject.h\
VertexBufferObject.cpp\
CommandStack.h\
CommandStack.cpp\
ImageProcessing.h\
ImageProcessing.cpp\
CudaImageProcessing.h\
CudaImageProcessing.cu\
SurfMatchingExperimental.h\
SurfMatchingExperimental.cu\
HistogramScreen.h\
HistogramScreen.cpp\
AnaglyphScreen.h\
AnaglyphScreen.cpp\
MixScreen.h\
MixScreen.cpp

