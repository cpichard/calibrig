#include "GrabberTest.h"
#include "ImageProcessing.h"

#include <fstream>
void readTestImage(const std::string &fileName, void *buffer, size_t bufferSize);
void uploadImageBuffer( unsigned char *buffer, unsigned int width, unsigned int heigth, unsigned int depth, ImageGL &dst );

GrabberTest::GrabberTest( Display *dpy, HGPUNV *gpu, GLXContext ctx )
: Grabber(dpy, gpu, ctx)
{}

GrabberTest::~GrabberTest(){}

bool GrabberTest::init()
{
    // Default values
    unsigned int depth =4;
    unsigned int width=960;
    unsigned int height=1080;

    // Allocation of stream buffer
    UInt2 imgSize(width,height);
    m_videoSize= UInt2(2*width,height);
    allocBufferAndTexture(m_stream1CaptureHandle, imgSize);
    allocBufferAndTexture(m_stream2CaptureHandle, imgSize);

    // Alloc memory buffers 
    size_t bufferSize = sizeof(unsigned char)*width*height*depth;
    unsigned char *imgRight = (unsigned char*)malloc(bufferSize);
    unsigned char *imgLeft= (unsigned char*)malloc(bufferSize);

    //std::string testFile="snapshot_cbox2_021312153017";
    //std::string testFile="snapshot_kali_030612075916"; 
    //std::string testFile="snapshot_kali_031112173101"; 
    std::string testFile="snapshot_cbox2_040612125803";
    std::stringstream testImage1;
    testImage1 << "./" << testFile << "_1.dat";
    std::stringstream testImage2;
    testImage2 << "./" << testFile << "_2.dat";
    readTestImage( testImage1.str(), imgRight, bufferSize);
    readTestImage( testImage2.str(), imgLeft, bufferSize);

    uploadImageBuffer( imgRight, width, height, depth, m_stream1CaptureHandle );
    uploadImageBuffer( imgLeft, width, height, depth, m_stream2CaptureHandle );

    SetSize(m_stream1CaptureHandle, m_videoSize);
    SetSize(m_stream2CaptureHandle, m_videoSize);

    free(imgRight);
    free(imgLeft);
    return true;
}
bool GrabberTest::captureVideo()
{
    return true;    
}

void GrabberTest::shutdown()
{
    releaseBufferAndTexture(m_stream1CaptureHandle);
    releaseBufferAndTexture(m_stream2CaptureHandle);    
}


void uploadImageBuffer( unsigned char *buffer, unsigned int width, unsigned int heigth, unsigned int depth, ImageGL &dst )
{
    CudaDevicePtrWrapper<ImageGL,uchar4*> outDevicePtr(dst);
    cudaMemcpy( outDevicePtr, buffer, sizeof(unsigned char)*depth*width*heigth, cudaMemcpyHostToDevice );
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

