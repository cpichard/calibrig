#ifndef __QUADVIEWSCREEN_H__
#define __QUADVIEWSCREEN_H__

#include "ScreenLayout.h"

#include "Monitor.h"

class QuadViewScreen : public ScreenLayout
{
public:
    QuadViewScreen( Display *dpy, UInt2 winSize );
    void draw();
    void resizeImage( UInt2 &winSize );
    void updateResult();
    void nextFrame();

private:
    Monitor m_mon1;
    Monitor m_mon2;
    Monitor m_mon3;
    Monitor m_mon4;
};

#endif //__QUADVIEWSCREEN_H__