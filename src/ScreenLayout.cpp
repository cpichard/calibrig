#include "ScreenLayout.h"
#include "Utils.h"


GLuint ScreenLayout::m_fontDisplayList = 0;
ImageGL ScreenLayout::m_stream1;
ImageGL ScreenLayout::m_stream2;
UInt2 ScreenLayout::m_winSize(0,0);


ComputationData *ScreenLayout::m_analysisResult = NULL;

ScreenLayout::ScreenLayout( Display *dpy,  UInt2 winSize )
:   m_dpy(dpy)
{
    m_winSize = winSize;
    if( m_fontDisplayList == 0 )
    {
        initFonts(m_dpy);
    }
}

ScreenLayout::~ScreenLayout()
{
// TODO : delete fonts when the last ScreenLayout end
// TODO : or change archi
//    if( m_font )
//    {
//        XFreeFont( m_dpy, m_font);
//    }
}

// Static function
void ScreenLayout::initFonts( Display *dpy )
{
    // Init fonts
    XFontStruct *font = XLoadQueryFont(dpy, "-*-lucida-*-r-normal-*-24-*-*-*-*-152-*-*");
    m_fontDisplayList = glGenLists(96);
    glXUseXFont(font->fid, ' ', 96, m_fontDisplayList);
    XFreeFont(dpy,font);
}


void ScreenLayout::drawText( float x, float y, const std::string &text )
{
    glPushAttrib(GL_LIST_BIT);
        glListBase(m_fontDisplayList - ' ');
        glRasterPos3f(x, y, 0.f);
        glCallLists(text.size(), GL_BYTE, text.c_str());
    glPopAttrib();

}