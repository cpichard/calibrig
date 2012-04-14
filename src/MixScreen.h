#ifndef __MIXSCREEN_H__
#define __MIXSCREEN_H__

#include "ScreenLayout.h"
#include "Monitor.h"

class MixScreen : public ScreenLayout
{
public:
    MixScreen( Display *dpy, UInt2 winSize );
    ~MixScreen();
    
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

#endif // __MIXSCREEN_H__
