
#include "CudaUtils.h"
#include "ImageGL.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <assert.h>
#include <fstream>
#include <string>
#include <vector>
#include <cv.h>
#include <highgui.h>

#include "CudaImageProcessing.h"
#include "ImageProcessing.h"
#include "SurfHessian.h"
#include "SurfNonMax.h"
#include "SurfDescriptor.h"
#include "SurfMatching.h"
#include "SurfDescriptor.h"
#include "SurfCollectPoints.h"

bool initCuda(CUcontext & cuContext)
{
    // Initialize Cuda
    CUresult cerr;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
        fprintf(stderr, "Sorry, no CUDA device found");
        return false;
	}

	int selectedDevice = 0;
	if (selectedDevice >= deviceCount)
	{
        fprintf(stderr, "Choose device ID between 0 and %d\n", deviceCount-1);
        return false;
	}

	// Initialize the CUDA device 
    CUdevice cuDevice;
	cerr = cuDeviceGet(&cuDevice,selectedDevice);
	checkError(cerr);

	cerr = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST|CU_CTX_BLOCKING_SYNC, cuDevice);
	checkError(cerr);
}

void computeDescriptorsLane(void *pixels, int depth, int width, int height, VertexBufferObject &points, DescriptorData  &descriptors)
{
    unsigned char *srcBuffer;
    unsigned char *dstBuffer;
    cudaMalloc( (void**) &srcBuffer, sizeof(unsigned char)*depth*width*height);
    cudaMemcpy( srcBuffer, pixels, sizeof(unsigned char)*depth*width*height, cudaMemcpyHostToDevice );
    cudaMalloc((void**) &dstBuffer, sizeof(unsigned char)*depth*width*height*2);
    // convert to RGB
    cudaYCbYCrToY( (uchar4*)dstBuffer, (uchar4*)srcBuffer, width*2, height);
    std::cout << "Passed cudaYCbYCrToY" << std::endl;

    // convert to 1 plane float cuda
    //float *yBuffer;
    //cudaMalloc((void**)&yBuffer, sizeof(float)*width*2*height);
    
    UInt2 imgSize(1920,1080);
    CudaImageBuffer<float> m_satImage;
    allocBuffer(m_satImage, imgSize);
    
    cudaRGBAtoCuda((float*)m_satImage, (uchar4*)dstBuffer, width*2, height, width*2); 
    std::cout << "Passed cudaRGBAtoCuda" << std::endl;
    
    convertToIntegral(m_satImage);
    std::cout << "Passed convertToIntegral" << std::endl;
    
    CudaImageBuffer<float> m_hesImage;
    allocBuffer(m_hesImage, imgSize);
    HessianData     m_hessianData;
    m_hessianData.allocImages(imgSize);
    computeHessianDet( m_satImage, m_hesImage, m_hessianData );
    std::cout << "Passed computeHessianDet" << std::endl;

    computeNonMaxSuppression( m_hesImage, m_hessianData );
    std::cout << "Passed computeNonMaxSuppression" << std::endl;

    collectHessianPoints( m_hessianData, descriptors);
    std::cout << "Passed collectHessianPoint" << std::endl;
    
    computeDescriptors( m_satImage, descriptors);
    std::cout << "Passed computeDescriptors" << std::endl;

    //collectPoints( descriptors, points, imgSize );
    
    // free memory
    m_hessianData.freeImages();
    releaseBuffer(m_hesImage); 
    releaseBuffer(m_satImage);
    cudaFree(srcBuffer);
    cudaFree(dstBuffer);
}

void readTestImage(const std::string &fileName, void *buffer, size_t bufferSize)
{
    FILE *ptr_fp = fopen(fileName.c_str(), "rb");
    if(ptr_fp==NULL)
        return;
    if( fread(buffer, bufferSize, 1, ptr_fp) != 1)
    {
       std::cout << "read buffer ok" << std::endl;  
    }

    fclose(ptr_fp);
}

void readTestsFiles(std::vector<std::string> &testFiles, const std::string &fileName)
{
    std::ifstream ifs( fileName.c_str() );
    std::string temp;

    while( getline( ifs, temp ) )
    {    
        testFiles.push_back( temp );    
    }
}

int main(int argc, char *argv[])
{
    // Read command line files 
    
    // Init cuda
    CUcontext cuContext;
    initCuda(cuContext); 

    // Image buffer allocation
    unsigned int depth =4;
    unsigned int width=960;
    unsigned int height=1080;
    size_t bufferSize = sizeof(unsigned char)*width*height*depth;
    unsigned char *imgRight = (unsigned char*)malloc(bufferSize);
    unsigned char *imgLeft= (unsigned char*)malloc(bufferSize);

    // Read the list of test files
    std::vector<std::string> testsFiles;
    readTestsFiles(testsFiles, "./tests.txt");
   
    // Launch tests 
    for(int i=0; i<testsFiles.size(); i++)
    {
        std::stringstream testImage1;
        testImage1 << "./" << testsFiles[i] << "_1.dat";
        std::stringstream testImage2;
        testImage2 << "./" << testsFiles[i] << "_2.dat";
        
        readTestImage( testImage1.str(), imgRight, bufferSize);
        readTestImage( testImage2.str(), imgLeft, bufferSize);
        
        VertexBufferObject rightPoints;
        VertexBufferObject leftPoints;
        DescriptorData  rightDescriptors;
        DescriptorData  leftDescriptors;
        computeDescriptorsLane(imgRight, depth, width, height, rightPoints, rightDescriptors);
        computeDescriptorsLane(imgLeft, depth, width, height, leftPoints, leftDescriptors);
        
        UInt2 imgSize(1920,1080);

        vector<CvPoint2D32f> leftMatchedPts;
        vector<CvPoint2D32f> rightMatchedPts;
        leftMatchedPts.reserve(10000);
        rightMatchedPts.reserve(10000);
        computeMatching( leftDescriptors, rightDescriptors, leftMatchedPts, rightMatchedPts, imgSize);
    }
    
    // finalize cuda
    CUresult cerr = cuCtxDestroy(cuContext);
    checkError(cerr);
    return 0;    
}
