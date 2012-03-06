#ifndef __GRABBERTEST_H__
#define __GRABBERTEST_H__

#include "Grabber.h"

class GrabberTest : public Grabber
{
public:
    GrabberTest( Display *dpy, HGPUNV *gpu, GLXContext ctx );
    ~GrabberTest();

    virtual bool init();
    virtual bool captureVideo();
    virtual void shutdown();
};


#endif//__GRABBERTEST_H__
