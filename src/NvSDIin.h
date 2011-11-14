#ifndef __NVSDIIN_H__
#define __NVSDIIN_H__
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>

#include <GL/gl.h>
#include <GL/glext.h>
//#include <GL/glew.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include "NVCtrlLib.h"
#include "NVCtrl.h"
#include "NvSDIutils.h"

#if defined __cplusplus
extern "C" {
#endif

// Definitions
#define FALSE 0
#define TRUE 1
#define MAX_VIDEO_STREAMS 4
#define NUM_QUERIES 5

    /* NV_video_capture */
#define GL_VIDEO_BUFFER_NV                                  0x9020
#define GL_VIDEO_BUFFER_BINDING_NV                          0x9021
#define GL_FIELD_UPPER_NV                                   0x9022
#define GL_FIELD_LOWER_NV                                   0x9023
#define GL_NUM_VIDEO_CAPTURE_STREAMS_NV                     0x9024
#define GL_NEXT_VIDEO_CAPTURE_BUFFER_STATUS_NV              0x9025
#define GL_VIDEO_CAPTURE_TO_422_SUPPORTED_NV                0x9026
#define GL_LAST_VIDEO_CAPTURE_STATUS_NV                     0x9027
#define GL_VIDEO_BUFFER_PITCH_NV                            0x9028
#define GL_VIDEO_COLOR_CONVERSION_MATRIX_NV                 0x9029
#define GL_VIDEO_COLOR_CONVERSION_MAX_NV                    0x902A
#define GL_VIDEO_COLOR_CONVERSION_MIN_NV                    0x902B
#define GL_VIDEO_COLOR_CONVERSION_OFFSET_NV                 0x902C
#define GL_VIDEO_BUFFER_INTERNAL_FORMAT_NV                  0x902D
#define GL_PARTIAL_SUCCESS_NV                               0x902E
#define GL_SUCCESS_NV                                       0x902F
#define GL_FAILURE_NV                                       0x9030
#define GL_YCBYCR8_422_NV                                   0x9031
#define GL_YCBAYCRA8_4224_NV                                0x9032
#define GL_Z6Y10Z6CB10Z6Y10Z6CR10_422_NV                    0x9033
#define GL_Z6Y10Z6CB10Z6A10Z6Y10Z6CR10Z6A10_4224_NV         0x9034
#define GL_Z4Y12Z4CB12Z4Y12Z4CR12_422_NV                    0x9035
#define GL_Z4Y12Z4CB12Z4A12Z4Y12Z4CR12Z4A12_4224_NV         0x9036
#define GL_Z4Y12Z4CB12Z4CR12_444_NV                         0x9037
#define GL_VIDEO_CAPTURE_FRAME_WIDTH_NV                     0x9038
#define GL_VIDEO_CAPTURE_FRAME_HEIGHT_NV                    0x9039
#define GL_VIDEO_CAPTURE_FIELD_UPPER_HEIGHT_NV              0x903A
#define GL_VIDEO_CAPTURE_FIELD_LOWER_HEIGHT_NV              0x903B

#ifndef GLX_NV_video_capture
typedef struct GLXVideoCaptureDeviceNVRec *GLXVideoCaptureDeviceNV;
#define GLX_DEVICE_ID_NV                 0x20CD
#define GLX_UNIQUE_ID_NV                 0x20CE
#define GLX_NUM_VIDEO_CAPTURE_SLOTS_NV   0x20CF
#endif

typedef
enum eCaptureType {
    TEXTURE_FRAME,
    BUFFER_FRAME,
} eCaptureType;

typedef
struct CaptureOptions {
    CaptureOptions();
    CaptureOptions(int screen);
    int xscreen; //index to the GPU list
    eCaptureType captureType;
    bool bDualLink;
    bool bChromaExpansion;
    int sampling;
    int bitsPerComponent;
    GLint bufferInternalFormat; // number of components in a pixel
    GLint textureInternalFormat; // number of components in a pixel
    GLint pixelFormat; // pixel format
    GLfloat *cscMat;
    GLfloat *cscOffset;
    GLfloat *cscMin;
    GLfloat *cscMax;
} CaptureOptions;

/*
 **	CNvSDIin - class that allows to capture one or multiple streams into
 ** a texture/video buffer object
 **
 */

#define MAX_GPUS 4
#define MEASURE_PERFORMANCE

typedef
class CNvSDIin
{
protected:
    Display *dpy;
    GLuint m_videoSlot; // Video slot number
    GLint m_hGVI; // Global GVI identifier
    unsigned int m_numStreams;
    unsigned int m_videoFormat;
    int m_videoWidth;
    int m_videoHeight;
    int m_windowWidth;
    int m_windowHeight;
    GLuint m_vidBuf[MAX_VIDEO_STREAMS]; // Video capture buffers
    GLint m_bufPitch[MAX_VIDEO_STREAMS]; // Buffer pitch
    GLuint m_vidTex[MAX_VIDEO_STREAMS]; // Video capture textures
    GLfloat m_cscMat[4][4];
    GLfloat m_cscMax[4];
    GLfloat m_cscMin[4];
    GLfloat m_cscOffset[4];
    GLXVideoCaptureDeviceNV m_hCaptureDevice; // SDI capture device handle.
    bool m_bCaptureStarted; // Set to true when glBeginVideoCaptureNV had successfuly completed
    GLuint m_captureTimeQuery;
    CaptureOptions m_captureOptions;
    int m_numActiveJacks;
    int m_numJacks;
    
public:
    float m_gviTime;
    float m_gpuTime;

    CNvSDIin();
    ~CNvSDIin();

    void setCaptureOptions( Display *display, const CaptureOptions &options );

    bool initCaptureDeviceNVCtrl();
    bool destroyCaptureDeviceNVCtrl();

    GLboolean initCaptureDeviceGL();
    GLboolean initInputDeviceGL();
    GLboolean destroyCaptureDeviceGL();

    bool startCapture();
    GLenum capture(GLuint *sequenceNum, GLuint64EXT *captureTime);
    bool endCapture();

    inline GLXVideoCaptureDeviceNV getHandle() {
        return m_hCaptureDevice;
    };

    inline unsigned int getWidth() {
        return m_videoWidth;
    };

    inline unsigned int getHeight() {
        return m_videoHeight;
    };

    inline GLuint getTextureObjectHandle(int objInd) {
        return m_vidTex[objInd];
    };

    inline GLuint getBufferObjectHandle(int objInd) {
        return m_vidBuf[objInd];
    };

    inline int getBufferObjectPitch(int objInd) {
        return m_bufPitch[objInd];
    };

    inline unsigned int getNumStreams() {
        return m_numStreams;
    }

    inline unsigned int getVideoFormat() {
        return m_videoFormat;
    }
}
CNvSDIin;

#if defined __cplusplus
} /* extern "C" */
#endif

#endif //__NVSDIIN_H__

