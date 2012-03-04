#ifndef __GRAPHICSYSTEMX11_H__
#define __GRAPHICSYSTEMX11_H__

#include "NvSDIutils.h"
#include "Utils.h"

#define MAX_GPUS 4

class GraphicSystemX11
{
public:
    GraphicSystemX11(UInt2 windowSize);
    virtual ~GraphicSystemX11();

    // Check if the system is compatible
    bool checkSystem();
    // Look fo gpus
    bool scanGPUS();

    Display *m_display; /// X11 Display
    Atom m_wmDeleteMessage; /// Register interest in the close window message 
    
    // GLX version
    int m_glxMajorVersion;
    int m_glxMinorVersion;

    // Number of gpu found
    int num_gpus;
    HGPUNV m_gpuList[MAX_GPUS];
    HGPUNV *m_gpu; 
    Window m_mainWin; 
    GLXContext m_glxContext;
    // Display manager ?

    // Window manager ?

};

#endif//__GRAPHICSYSTEMX11_H__
