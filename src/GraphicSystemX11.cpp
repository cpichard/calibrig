#include "GraphicSystemX11.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "NVCtrlLib.h"
#include "NVCtrl.h"
#include <iostream>

#include "MainWindow.h"

GraphicSystemX11::GraphicSystemX11(UInt2 windowSize)
: m_display(NULL)
, m_mainWin(NULL)
, m_glxContext(NULL)
, m_gpu(NULL)
{
    m_display = XOpenDisplay(NULL);
    if( m_display == NULL )
    {
        std::cerr << "Couldn't find X11 display - existing" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Register interest in the close window message
    m_wmDeleteMessage = XInternAtom(m_display, "WM_DELETE_WINDOW", False);
    // TODO XSetWMProtocols(m_display, mainWin, &wmDeleteMessage, 1);

    checkSystem();
    
    // Grab the first GPU for now for DVP
    m_gpu = &(m_gpuList[0]);
    
    // Create window
    // gs.m_display -> GraphicRenderContext ? gs.m_display + cuda + gl ?yy
    m_mainWin= createMainWindow( m_display, m_glxContext, m_gpu->deviceXScreen, Width(windowSize), Height(windowSize) );

    // Register interest in the close window message
    XSetWMProtocols(m_display, m_mainWin, &m_wmDeleteMessage, 1);
}

GraphicSystemX11::~GraphicSystemX11()
{
    if( m_display )
    {
        XCloseDisplay(m_display);    
    }
}


bool GraphicSystemX11::checkSystem()
{
    // Query GLX version
    glXQueryVersion( m_display, &m_glxMajorVersion, &m_glxMinorVersion );
    std::cout << "glX-Version " << m_glxMajorVersion << "." << m_glxMinorVersion << std::endl;

    // Scan the systems for GPUs
    if( scanGPUS() == false )
    {
        std::cerr << "Error: no gpu found on this machine. Exiting" << std::endl;        
        exit(EXIT_FAILURE);
    }
}


//assuming there is one XServer running in the system
bool GraphicSystemX11::scanGPUS()
{
    HGPUNV gpuDevice;
    int num_gpus, num_screens;
    int gpu, screen;
    int mask;
    int *pData;
    int len, j;
    char *str=NULL;

    /* Get the number of gpus in the system */
    bool ret = XNVCTRLQueryTargetCount(m_display, NV_CTRL_TARGET_TYPE_GPU, &num_gpus);
    if(!ret) 
    {
        std::cerr << "Failed to query number of gpus" << std::endl;
        return false;
    }
    std::cout << "number of GPUs: " << num_gpus << std::endl;

    int num_gpusWithXScreen = 0;
    for(gpu = 0; gpu < num_gpus; gpu++)
    {
        printf("GPU %d information:\n", gpu);
        /* GPU name */
        ret = XNVCTRLQueryTargetStringAttribute(m_display,
            NV_CTRL_TARGET_TYPE_GPU,
            gpu, // target_id
            0, // display_mask
            NV_CTRL_STRING_PRODUCT_NAME,
            &str);
        if(!ret) 
        {
            fprintf(stderr, "Failed to query gpu product name\n");
            return 1;
        }
        printf("   Product Name                    : %s\n", str);
        /* X Screens driven by this GPU */
        ret = XNVCTRLQueryTargetBinaryData
            (m_display,
            NV_CTRL_TARGET_TYPE_GPU,
            gpu, // target_id
            0, // display_mask
            NV_CTRL_BINARY_DATA_XSCREENS_USING_GPU,
            (unsigned char **) &pData,
            &len);
        if(!ret) 
        {
            fprintf(stderr, "Failed to query list of X Screens\n");
            exit(EXIT_FAILURE);
        }
        printf("   Number of X Screens on GPU %d    : %d\n", gpu, pData[0]);
        //only return GPUs that have XScreens
        if(pData[0]) 
        {
            gpuDevice.deviceXScreen = pData[1]; //chose the first screen
            strcpy(gpuDevice.deviceName, str);
            m_gpuList[gpu] = gpuDevice;
            num_gpusWithXScreen++;
        }
        free(str);
        str=NULL;
        XFree(pData);
    }
    return num_gpusWithXScreen;
}


