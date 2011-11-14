#ifndef __HISTOGRAMSCREEN_H__
#define __HISTOGRAMSCREEN_H__

#include <ScreenLayout.h>

class HistogramScreen : public ScreenLayout
{

public:
    HistogramScreen( Display *dpy, UInt2 winSize );
    ~HistogramScreen();

    virtual void draw();
    virtual void resizeImage( UInt2 &imgSize );
    virtual void updateResult();
    virtual void nextFrame();
};

#endif//__HISTOGRAMSCREEN_H__
