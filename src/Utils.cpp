#include "Utils.h"
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cudaGL.h>
#include "CudaUtils.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <iostream>
#include "NvSDIutils.h"
#include "NvSDIin.h"

bool checkSystem( Display *dpy )
{
    // GLX
    int glxMajorVersion, glxMinorVersion;
    glXQueryVersion( dpy, &glxMajorVersion, &glxMinorVersion );
    std::cout << "glX-Version " << glxMajorVersion << "." << glxMinorVersion << std::endl;

    //scan the systems for GPUs
    HGPUNV gpuList[MAX_GPUS];
    int	num_gpus = ScanHW(dpy,gpuList);
    if(num_gpus < 1)
        return false;

    return true;
}
