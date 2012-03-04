
#ifndef __ANALYZERFUNCTOR_H__
#define __ANALYZERFUNCTOR_H__

#include "StereoAnalyzer.h"
#include "MainWindow.h"

// TODO rename analyzer
struct AnalyzerFunctor
{
    AnalyzerFunctor( StereoAnalyzer &analyzer, 
                        CUcontext &cuContext,
                        Display *dpy,
                        GLXContext &ctx
                        )
    : m_analyzer(analyzer)
    , m_running(true)
    , m_cuContext(cuContext)
    , m_dpy(dpy)
    , m_ctx(ctx)
    {}

    void stop(){m_running=false;}

    // Function launch by the thread
    void operator()()
    {
        // Create an OpenGL context for this thread
        GLXContext ctx;
        Window win = createWindow(m_dpy, 0, ctx, 1, 1, m_ctx);
        
        // Make this new GL context current
        glXMakeCurrent(m_dpy, win, ctx);
       
        // TODO : make the cuda stream current to the thread ?
        //
        
        // Set the cuda context to this thread 
        cuCtxSetCurrent(m_cuContext);
        checkLastError();

        // Let's roll
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
};

#endif 

