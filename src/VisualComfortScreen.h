#ifndef __VISUALCONFORTSCREEN_H__
#define __VISUALCONFORTSCREEN_H__

#include "ScreenLayout.h"
#include "Monitor.h"


class VisualComfortScreen : public ScreenLayout
{
public:
    VisualComfortScreen( Display *dpy, UInt2 winSize );
    ~VisualComfortScreen();

    void draw();
    void updateResult();
    void nextFrame();
    
    void resizeImage( UInt2 &winSize );
    void allocImage( UInt2 &winSize );
    void freeImage();

private:    
    Monitor m_mon;

    // Warp buffer
    ImageGL m_leftImg;
    ImageGL m_rightImg;
    ImageGL m_warpedRightImg;
    ImageGL m_warpedLeftImg;
};


#endif//__VISUALCONFORTSCREEN_H__
