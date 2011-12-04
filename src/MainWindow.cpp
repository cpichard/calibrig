#include "MainWindow.h"

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstdio>

// Create
static Bool WaitForNotify(Display * d, XEvent * e, char *arg)
{
    return (e->type == MapNotify) && (e->xmap.window == (Window) arg);
}

Window createMainWindow( Display *dpy, GLXContext &ctx, int xscreen, unsigned int windowWidth, unsigned int windowHeight )
{
    XVisualInfo *vi, VI;
    GLXFBConfig *configs, config;
    XEvent event;
    XSetWindowAttributes swa;

    unsigned long mask;
    int numConfigs;
    int config_list[] = {GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_DOUBLEBUFFER, GL_TRUE,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_FLOAT_COMPONENTS_NV, GL_FALSE,
        None
    };

    // Find required framebuffer configuration
    configs = glXChooseFBConfig(dpy, xscreen, config_list, &numConfigs);
    if(!configs) {
        fprintf(stderr, "Unable to find a matching FBConfig.\n");
        exit(1);
    }

    int i;
    // Find an FBconfig with the required number of color bits.
    for(i = 0; i < numConfigs; i++) {
        int attr;
        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_RED_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_RED_SIZE) failed!\n");
            exit(1);
        }
        if(attr != 8)
            continue;

        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_GREEN_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_GREEN_SIZE) failed!\n");
            exit(1);
        }
        if(attr != 8)
            continue;

        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_BLUE_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_BLUE_SIZE) failed!\n");
            exit(1);
        }

        if(attr != 8)
            continue;

        if(glXGetFBConfigAttrib(dpy, configs[i], GLX_ALPHA_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_ALPHA_SIZE) failed\n");
            exit(1);
        }

        if(attr != 8)
            continue;
        break;
    }

    if(i == numConfigs) {
        printf("No FBConfigs found\n");
        exit(1);
    }

    config = configs[i];

    // Don't need the config list anymore so free it.
    XFree(configs);
    configs = NULL;


    // Create an OpenGL rendering context for the onscreen window.
    ctx = glXCreateNewContext(dpy, config, GLX_RGBA_TYPE, 0, GL_TRUE);

    // Get visual from FB config.
    vi = &VI;
    if(vi = glXGetVisualFromFBConfig(dpy, config)) {
        printf("Using visual %0x\n", vi->visualid);
        printf("Depth = %d\n", vi->depth);
    } else {
        printf("Couldn't find visual for onscreen window.\n");
        exit(1);
    }

    Colormap cmap;
    // Create color map.
    if(!(cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),
        vi->visual, AllocNone))) {
        fprintf(stderr, "XCreateColormap failed!\n");
        exit(1);
    }

    // Create window.
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

    mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask;
    Window win = XCreateWindow(dpy, RootWindow(dpy, vi->screen),
    0, 0, windowWidth, windowHeight, 0,
    vi->depth, InputOutput, vi->visual,
    mask, &swa);

    // Map window.
    XMapWindow(dpy, win);
    //XIfEvent(dpy, &event, WaitForNotify, (char *) win);

    // Set window colormap.
    XSetWMColormapWindows(dpy, win, &win, 1);

    // Make OpenGL rendering context current.
    if(!(glXMakeCurrent(dpy, win, ctx)))
    {
        fprintf(stderr, "glXMakeCurrent failed!\n");
        exit(1);
    }

    // Don't lock the capture/draw loop to the graphics vsync.
    glXSwapIntervalSGI(0);
    XFlush(dpy);
    return win;
}
