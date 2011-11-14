#ifndef __SCREENLAYOUT_H__
#define __SCREENLAYOUT_H__

#include "Utils.h"
#include "NvSDIin.h"
#include "StereoAnalyzer.h"

//! A screen layout is like a screen page of the application
//! There are 3 pages for the moment : Quad view, diff view and histogram
class ScreenLayout
{
public:
    ScreenLayout( Display *dpy, UInt2 winSize );
    virtual ~ScreenLayout();
    
    virtual void draw()=0;
    virtual void resizeWindow( UInt2 winSize ){ m_winSize = winSize; }
    virtual void resizeImage( UInt2 &imgSize )=0;
    virtual void updateResult()=0;
    virtual void nextFrame()=0;

    inline unsigned int winWidth() const {return Width(m_winSize);}
    inline unsigned int winHeight() const {return Height(m_winSize);}

    inline static void setResult( ComputationData *result ){m_analysisResult=result;}

    inline static void setVideoStreams( ImageGL stream1, ImageGL stream2 ){m_stream1=stream1; m_stream2=stream2;};

private:
    static void initFonts(Display *dpy);
    
protected:

    static void drawText( float x, float y, const std::string &text );

    // Display
    Display             *m_dpy;

    // Windows size
    static UInt2        m_winSize;

    // Common to all screens
    static GLuint           m_fontDisplayList;
    static ComputationData   *m_analysisResult;
    static ImageGL          m_stream1;
    static ImageGL          m_stream2;
};

#endif //__SCREENLAYOUT_H__