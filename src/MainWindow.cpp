#include "MainWindow.h"

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstdio>


Window createMainWindow( Display *dpy, GLXContext &ctx, int xscreen, unsigned int windowWidth, unsigned int windowHeight )
{
    // Create a window
    Window win = createWindow( dpy, xscreen, ctx, windowWidth, windowHeight, 0 );
    
    // Map window.
    XMapWindow(dpy, win);

    // Make OpenGL rendering context current.
    if(!(glXMakeCurrent(dpy, win, ctx)))
    {
        std::cerr << "glXMakeCurrent failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Don't lock the capture/draw loop to the graphics vsync.
    glXSwapIntervalSGI(0);

    return win;
}

// Return the more suitable frame buffer config
GLXFBConfig findFrameBufferConfig(Display *dpy, int xscreen)
{
    GLXFBConfig *configs, config;
    int numConfigs;
    int configList[] = {
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_DOUBLEBUFFER, GL_TRUE,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_FLOAT_COMPONENTS_NV, GL_FALSE,
        None 
    };

    // Find required framebuffer configuration
    configs = glXChooseFBConfig(dpy, xscreen, configList, &numConfigs);
    if(!configs) 
    {
        std::cerr << "Unable to find a matching FBConfig." << std::endl; 
        exit(EXIT_FAILURE);
    }

    // Find framebuffer config with the required number of color bits.
    int i;
    for(i = 0; i < numConfigs; i++)
    {
        int attr;
        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_RED_SIZE, &attr))
        {
            std::cerr << "glXGetFBConfigAttrib(GLX_RED_SIZE) failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if(attr != 8)
            continue;

        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_GREEN_SIZE, &attr))
        {
            printf("glXGetFBConfigAttrib(GLX_GREEN_SIZE) failed!\n");
            exit(EXIT_FAILURE);
        }
        if(attr != 8)
            continue;

        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_BLUE_SIZE, &attr))
        {
            std::cerr << "glXGetFBConfigAttrib(GLX_BLUE_SIZE) failed!" << std::endl;
            exit(EXIT_FAILURE);
        }

        if(attr != 8)
            continue;

        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_ALPHA_SIZE, &attr)) 
        {
            std::cerr << "glXGetFBConfigAttrib(GLX_ALPHA_SIZE) failed" << std::endl;
            exit(EXIT_FAILURE);
        }

        if(attr != 8)
            continue;
        break;
    }

    if(i == numConfigs) 
    {
        std::cerr << "No FBConfigs found" << std::endl;
        exit(EXIT_FAILURE);
    }

    config = configs[i];

    // Don't need the config list anymore so free it.
    XFree(configs);
    configs = NULL;

    return config;
}

// Create an X Window
Window createWindow(  Display *dpy
                    , int xscreen
                    , GLXContext &ctx
                    , int windowWidth, int windowHeight
                    , GLXContext sharedCtx)
{
    // Find frame buffer config
    GLXFBConfig config = findFrameBufferConfig( dpy, xscreen );

    // Create an OpenGL rendering context for the onscreen window.
    ctx = glXCreateNewContext(dpy, config, GLX_RGBA_TYPE, sharedCtx, GL_TRUE);

    // Get visual from FB config.
    XVisualInfo *vi = glXGetVisualFromFBConfig(dpy, config);
    if(vi) 
    {
        std::cout << "Using visual " << vi->visualid << std::endl;
        std::cout << "Depth = " << vi->depth << std::endl;
    } 
    else 
    {
        std::cerr << "Couldn't find visual for onscreen window." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create color map.
    Colormap cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    if(!cmap)  
    {
        std::cerr << "XCreateColormap failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create window.
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.background_pixel = 1;
    swa.override_redirect = true;
    swa.event_mask  = ExposureMask 
                    | StructureNotifyMask 
                    | SubstructureNotifyMask
                    | KeyPressMask 
                    | KeyReleaseMask 
                    | ButtonPressMask 
                    | ButtonReleaseMask 
                    | PointerMotionMask 
                    | PropertyChangeMask
                    | ClientMessage;

    unsigned long mask  = CWBackPixel 
                        | CWBorderPixel 
                        | CWColormap 
                        | CWEventMask;

    Window win = XCreateWindow( dpy, RootWindow(dpy, vi->screen), 
                                0, 0, windowWidth, windowHeight, 0,
                                vi->depth, InputOutput, vi->visual,
                                mask, &swa);

    return win;
}
