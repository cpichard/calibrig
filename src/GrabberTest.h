#ifndef __GRABBERTEST_H__
#define __GRABBERTEST_H__

#include <vector>
#include <string>
#include "Grabber.h"

class GrabberTest : public Grabber
{
public:
    GrabberTest( Display *dpy, HGPUNV *gpu, GLXContext ctx );
    ~GrabberTest();

    virtual bool init();
    virtual bool captureVideo();
    virtual void shutdown();
    
    int m_lastIndex;
    unsigned char *m_imgRight;
    unsigned char *m_imgLeft;
    std::vector<std::string> m_testImages;
};


#endif//__GRABBERTEST_H__
