#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "cv.h"
#include "highgui.h"

#include "ImageGL.h"

class Monitor
{
public:
    Monitor();
    ~Monitor();

    void resizeImage( UInt2 &imgSize );
    void allocImage( UInt2 &imgSize );
    void freeImage();

    void updateWithSDIVideo( ImageGL &img );
    void updateWithImageBuffer( unsigned char *buffer, unsigned int width, unsigned int height, unsigned int depth );
    void updateWithImageBuffer( ImageGL &img, unsigned int depth );
    void updateWithSDIVideoDiff( ImageGL &imgA, ImageGL &imgB );
    void drawGL();

    inline UInt2 & imageSize() { return Size(m_imageGL); }

private:

    GLfloat m_scaledVideoWidth;
    GLfloat m_scaledVideoHeight;

    ImageGL m_imageGL;
};

#endif