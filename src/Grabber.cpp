#include "Grabber.h"

#include "ImageProcessing.h"

Grabber::Grabber( Display *dpy, HGPUNV *gpu, GLXContext ctx )
: m_dpy(dpy)
, m_gpu(gpu)
, m_ctx(ctx)
, m_videoSize(0,0)
, m_stream1CaptureHandle()
, m_stream2CaptureHandle()
{}


Grabber::~Grabber(){}

void Grabber::saveImages()
{
    // Build filenames
    char dateAndTime[16];
    time_t tt = time(NULL);
    struct tm *localTime = localtime (&tt);
    strftime(dateAndTime, 16, "%m%d%y%H%M%S", localTime);
    char hostname[64];
    gethostname(hostname, 64);
    
    std::stringstream d1, d2;
    d1 << "./snapshot_" << hostname << "_" << dateAndTime << "_1.dat";
    d2 << "./snapshot_" << hostname << "_" << dateAndTime << "_2.dat";
    
    // Format machine date time stream   
    saveGrabbedImage(m_stream1CaptureHandle, d1.str());
    saveGrabbedImage(m_stream2CaptureHandle, d2.str());  
    
    // Save format : TODO : store values in NVSDIin.cpp
    //std::stringstream t1, t2 ;
    //t1 << "./snapshot_" << hostname << "_" << dateAndTime << "_1.txt";
    //t2 << "./snapshot_" << hostname << "_" << dateAndTime << "_2.txt";
    //decodeSignalFormat
    //decodeComponentSampling
    //decodeBitsPerComponent
    //decodeColorSpace
}



