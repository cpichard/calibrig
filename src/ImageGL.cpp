#include "ImageGL.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>
#include <cuda_runtime.h>
#include <cudpp.h>

#include "CudaUtils.h"

#include "ImageProcessing.h"

// 1 plane float buffer
bool allocBuffer( PixelBufferObject &img, UInt2 &imgSize )
{
   if( Size(img) != imgSize && imgSize != UInt2(0,0) )
   {
        // Allocate monitor PBO 
        glGenBuffersARB( 1, &(img.m_bufId) );
        glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, img.m_bufId );
        glBufferDataARB( GL_PIXEL_UNPACK_BUFFER_ARB, Width(imgSize) * Height(imgSize) * sizeof(float), NULL, GL_STREAM_COPY);

        // Register buffer for use with cuda
        //CUresult cerr = cuGLRegisterBufferObject( img.m_bufId );
        //checkError(cerr);
        glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

        SetSize(img, imgSize);
        return true;
   }

   return false;
}

bool releaseBuffer( PixelBufferObject &img )
{
    if( Size(img) != UInt2(0,0) )
    {
        SetSize(img,UInt2(0,0));

        // Unregister buffer from cuda
        //cuGLUnregisterBufferObject( img.m_bufId );

        // Release PBO
        glDeleteBuffers(1, &img.m_bufId );
        img.m_bufId = 0;
        return true;
    }
    return false;
}

// TODO : separate allocation buffer and texture
static
bool allocBufferAndTexture( GLuint &buf, GLuint &tex, unsigned int width, unsigned int height, unsigned int depth )
{
    // Texture 
    glGenTextures( 1, &tex );
    GLenum err = glGetError(); assert( err == GL_NO_ERROR );
	glBindTexture( GL_TEXTURE_RECTANGLE_NV, tex );
	glTexParameterf( GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D( GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, width, height,0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture( GL_TEXTURE_RECTANGLE_NV, 0 );

    // Allocate monitor PBO ( pixel buffer object )
    glGenBuffersARB( 1, &buf );
	glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, buf );
	glBufferDataARB( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * depth, NULL, GL_STREAM_COPY );

    // Register buffer for use with cuda
    //CUresult cerr = cuGLRegisterBufferObject( buf );
    //checkError(cerr);
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

    return true;
}

bool allocBufferAndTexture( ImageGL &img, UInt2 &imgSize )
{
   if( Size(img) != imgSize && imgSize != UInt2(0,0) )
   {
       if( allocBufferAndTexture( BufId(img), TexId(img), Width(imgSize), Height(imgSize), 4 ) )
       {
            SetSize(img, imgSize);
            return true;
       }
   }
   return false;
}

static bool releaseBufferAndTexture( GLuint &buf, GLuint &tex )
{
    // Unregister buffer from cuda
    //cuGLUnregisterBufferObject( buf );

    //std::cout << "Released " << tex << std::endl;
    // Release PBO
    glDeleteBuffers(1, &buf );
    buf = 0;
    
    // Release Texture
    glDeleteTextures(1, &tex );
    tex = 0;

    return true;
}

bool releaseBufferAndTexture( ImageGL &img )
{
    if( Size(img) != UInt2(0,0) )
    {
        SetSize(img, UInt2(0,0));
        return releaseBufferAndTexture( BufId(img), TexId(img) );
    }
    return true;
}

// Copy buffer of imageGL to its texture
bool copyBufToTex( ImageGL &imageGL )
{
    // Copy cuda buffer m_monitorBufId in texture m_monitorTexId
    glBindTexture( GL_TEXTURE_RECTANGLE_NV, TexId(imageGL) );
    assert(glGetError() == GL_NO_ERROR);
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, BufId(imageGL) );
    assert(glGetError() == GL_NO_ERROR);
    glTexSubImage2D( GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, Width(imageGL), Height(imageGL), GL_RGBA, GL_UNSIGNED_BYTE, 0);
    assert(glGetError() == GL_NO_ERROR);
   	glBindTexture( GL_TEXTURE_RECTANGLE_NV, 0 );
    assert(glGetError() == GL_NO_ERROR);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    assert(glGetError() == GL_NO_ERROR);

    return true;
}



