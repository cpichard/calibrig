#ifndef __DIFFSCREEN_H__
#define __DIFFSCREEN_H__

#include "ScreenLayout.h"
#include "Monitor.h"

class DiffScreen : public ScreenLayout
{
public:
    DiffScreen( Display *dpy, UInt2 winSize );
    ~DiffScreen();
    
    void draw();
    void updateResult();
    void nextFrame();
    
    void resizeImage( UInt2 &winSize );
    void allocImage( UInt2 &winSize );
    void freeImage();
    
private:
    Monitor m_mon;

    // Warp buffer
    ImageGL m_warpedImg;
    ImageGL m_leftImg;
    ImageGL m_rightImg;

};

#endif // __DIFFSCREEN_H__