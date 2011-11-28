
#include "CudaUtils.h"
#include "ImageGL.h"

#include <cstdio>
#include <assert.h>

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

int main()
{
    // initialize Cuda
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

	//Initialize the CUDA device 
    CUdevice cuDevice;
	cerr = cuDeviceGet(&cuDevice,selectedDevice);
	checkError(cerr);

    CUcontext cuContext;
	cerr = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST|CU_CTX_BLOCKING_SYNC, cuDevice);
	checkError(cerr);
    
    // Open 1080i 422 image
    const char *filename="./test.tif"; 
    IplImage* img=0; 
    //img=cvLoadImage(filename);
    img=cvCreateImage(cvSize(960,1080),IPL_DEPTH_8U,4); 
    // Copy to cuda buffer
    if(img)
    {
        std::cout << "image ok" << std::endl;
        unsigned char *srcBuffer;
        unsigned char *dstBuffer;
        unsigned int depth =4;
        unsigned int width=960;
        unsigned int height=1080;

        cudaMalloc( (void**) &srcBuffer, sizeof(unsigned char)*depth*width*height);
        cudaMemcpy( srcBuffer, img->imageData, sizeof(unsigned char)*depth*width*height, cudaMemcpyHostToDevice );
        if(img)
        {
            cvReleaseImage(&img);    
        }
        
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

        DescriptorData m_descriptors;

        collectHessianPoints( m_hessianData, m_descriptors);
        std::cout << "Passed collectHessianPoint" << std::endl;
        
        computeDescriptors( m_satImage, m_descriptors);
        std::cout << "Passed computeDescriptors" << std::endl;
        
        // free memory
        m_hessianData.freeImages();
        releaseBuffer(m_hesImage); 
        releaseBuffer(m_satImage);
        cudaFree(srcBuffer);
        cudaFree(dstBuffer);
    }
    // finalize cuda
    cerr = cuCtxDestroy(cuContext);
    checkError(cerr);
    return 0;    
}
