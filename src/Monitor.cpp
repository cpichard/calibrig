#include "Monitor.h"
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>

#include "CudaUtils.h"
#include "Utils.h"
#include "ImageProcessing.h"
#include "CudaImageProcessing.h"


Monitor::Monitor( )
:  m_scaledVideoWidth(0),
   m_scaledVideoHeight(0),
   m_imageGL()
{
    // Compute m_scaledVideoWidth and m_scaledVideoHeight
    m_scaledVideoWidth = 1.f;
    m_scaledVideoHeight = 1.f;
}

Monitor::~Monitor()
{
    freeImage();
}

void Monitor::resizeImage( UInt2 &imgSize )
{
    freeImage();
    allocImage( imgSize );
}

void Monitor::allocImage( UInt2 &imgSize )
{
    allocBufferAndTexture( m_imageGL, imgSize );
}

void Monitor::freeImage()
{
    releaseBufferAndTexture( m_imageGL );
}


void Monitor::updateWithSDIVideo( ImageGL &img )
{
    if( convertYCbYCrToRGB( img, m_imageGL ) )
    {
        copyBufToTex( m_imageGL );
    }
}

void Monitor::updateWithImageBuffer( unsigned char *buffer, unsigned int width, unsigned int heigth, unsigned int depth )
{
    if( depth == 1 )
    {
        copyImageBuffer(buffer, width, heigth, depth, m_imageGL );
        
        //
        copyBufToTex(m_imageGL);
    }
}

void Monitor::updateWithImageBuffer( ImageGL &img, unsigned int depth )
{
    // Copy passed image
    copyImageBuffer(img, m_imageGL);

    //
    copyBufToTex(m_imageGL);
}

void Monitor::updateWithSDIVideoDiff( ImageGL &imgA, ImageGL &imgB )
{
    
    diffImageBufferYCbYCr( imgA, imgB, m_imageGL );

    //
    copyBufToTex(m_imageGL);
}

void Monitor::drawGL()
{
    if( TexId(m_imageGL) == 0 )
    {
        return;
    }

    // Enable texture mapping
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    assert(glGetError() == GL_NO_ERROR);
    
    // Draw textured quad in graphics window.
    glBindTexture( GL_TEXTURE_RECTANGLE_NV, TexId(m_imageGL) );
    glBegin(GL_QUADS);
        glTexCoord2i(0, 0);
        glVertex2f(-m_scaledVideoWidth, -m_scaledVideoHeight);
        glTexCoord2i(0, Height(m_imageGL));
        glVertex2f(-m_scaledVideoWidth, m_scaledVideoHeight);
        glTexCoord2i(Width(m_imageGL), Height(m_imageGL));
        glVertex2f(m_scaledVideoWidth, m_scaledVideoHeight);
        glTexCoord2i(Width(m_imageGL), 0);
        glVertex2f(m_scaledVideoWidth, -m_scaledVideoHeight);
    glEnd();
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);
    assert(glGetError() == GL_NO_ERROR);

    // Disable texture mapping
    glDisable(GL_TEXTURE_RECTANGLE_NV);
    assert(glGetError() == GL_NO_ERROR);
}
