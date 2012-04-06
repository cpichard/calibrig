#ifndef __ANAGLYPHSCREEN_H__
#define __ANAGLYPHSCREEN_H__

#include "ScreenLayout.h"
#include "Monitor.h"

class AnaglyphScreen : public ScreenLayout
{
public:
    AnaglyphScreen( Display *dpy, UInt2 winSize );
    ~AnaglyphScreen();
    
    void draw();
    void updateResult();
    void nextFrame();
    
    void resizeImage( UInt2 &winSize );
    void allocImage( UInt2 &winSize );
    void freeImage();
    
private:
    Monitor m_mon;

    // Warp buffer
    ImageGL m_anaglyphImg;
    ImageGL m_leftImg;
    ImageGL m_rightImg;

};

#endif // __ANAGLYPHSCREEN_H__
