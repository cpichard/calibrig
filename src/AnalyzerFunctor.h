
#ifndef __ANALYZERFUNCTOR_H__
#define __ANALYZERFUNCTOR_H__

#include "StereoAnalyzer.h"


// TODO rename analyzer
struct AnalyzerFunctor
{
    AnalyzerFunctor( StereoAnalyzer &analyzer, 
                        CUcontext &cuContext,
                        Display *dpy,
                        GLXContext &ctx,
                        Window &win )
    : m_analyzer(analyzer)
    , m_running(true)
    , m_cuContext(cuContext)
    , m_dpy(dpy)
    , m_win(win)
    , m_ctx(ctx)
    {}

    void stop(){m_running=false;}

    void operator()()
    {
      // Create a second OpenGL context
      XVisualInfo *vinfo; 
      XSetWindowAttributes swattr; 
      int attrList[20]; 
      int indx=0; 
      Window wid; 
      GLXContext util_glctx;     
      attrList[indx] = GLX_USE_GL; 
      indx++; 
      attrList[indx] = GLX_DEPTH_SIZE; 
      indx++; 
      attrList[indx] = 1; 
      indx++; 
      attrList[indx] = GLX_RGBA; 
      indx++; 
      attrList[indx] = GLX_RED_SIZE; 
      indx++; 
      attrList[indx] = 1; 
      indx++; 
      attrList[indx] = GLX_GREEN_SIZE; 
      indx++; attrList[indx] = 1; 
      indx++; 
      attrList[indx] = GLX_BLUE_SIZE; 
      indx++; attrList[indx] = 1; 
      indx++;     
      attrList[indx] = None;     
      vinfo = glXChooseVisual(m_dpy, DefaultScreen(m_dpy), attrList);     
      if (vinfo == NULL) { printf ("ERROR: Can't open window\n"); exit (1);     }     
      swattr.colormap=XCreateColormap (m_dpy ,RootWindow (m_dpy,vinfo->screen), vinfo->visual, AllocNone);     
      swattr.background_pixel = BlackPixel (m_dpy, vinfo->screen);     
      swattr.border_pixel = BlackPixel (m_dpy, vinfo->screen);
      wid = XCreateWindow(m_dpy,RootWindow(m_dpy, vinfo->screen),           
                            30, 30, 1, 1, 0, vinfo->depth, CopyFromParent,           
                            vinfo->visual,CWBackPixel | CWBorderPixel | CWColormap, &swattr);     
      util_glctx = glXCreateContext(m_dpy, vinfo, m_ctx, True);     
      if (util_glctx == NULL) { printf("glXCreateContext failed \n"); return;     }     
      if (!glXMakeCurrent(m_dpy, wid, util_glctx)) 
      { 
        printf("glXMakeCurrent failed \n"); 
        return;
      }  
        
        // Set the cuda context to this thread 
        cuCtxSetCurrent(m_cuContext);
        checkLastError();
        while( m_running == true )
        {
            if( m_analyzer.imagesAreNew() == false )
            {
                sleep(0.1);
            }
            else
            if( m_analyzer.try_lock() )
            {
                m_analyzer.analyse();
                m_analyzer.unlock();
            }
        }
    }

    StereoAnalyzer &m_analyzer;
    bool            m_running;
    CUcontext       m_cuContext;
    Display *m_dpy;
    GLXContext &m_ctx;
    Window &m_win;
};

#endif 

