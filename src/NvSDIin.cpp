#include "NvSDIin.h"


#if defined __cplusplus
extern "C" {
#endif


#ifndef GL_ARB_sync
#define GL_ARB_sync
GLAPI void glGetInteger64v(GLenum pname, GLint64EXT *params);
#endif


#ifndef GL_NV_video_capture
#define GL_NV_video_capture 1
GLAPI void  glBeginVideoCaptureNV (GLuint video_capture_slot);
GLAPI void  glBindVideoCaptureStreamBufferNV (GLuint video_capture_slot, GLuint stream, GLenum frame_region, GLintptr offset);
GLAPI void  glBindVideoCaptureStreamTextureNV (GLuint video_capture_slot, GLuint stream, GLenum frame_region, GLenum target, GLuint texture);
GLAPI void  glEndVideoCaptureNV (GLuint video_capture_slot);
GLAPI void  glGetVideoCaptureivNV (GLuint video_capture_slot, GLenum pname, GLint *params);
GLAPI void  glGetVideoCaptureStreamivNV (GLuint video_capture_slot, GLuint stream, GLenum pname, GLint *params);
GLAPI void  glGetVideoCaptureStreamuivNV (GLuint video_capture_slot, GLuint stream, GLenum pname, GLuint *params);
GLAPI void  glGetVideoCaptureStreamfvNV (GLuint video_capture_slot, GLuint stream, GLenum pname, GLfloat *params);
GLAPI void  glGetVideoCaptureStreamdvNV (GLuint video_capture_slot, GLuint stream, GLenum pname, GLdouble *params);
GLAPI void  glPresentFrameKeyedBuffersNV (GLuint video_slot, GLuint64EXT minPresentTime, GLuint beginPresentTimeId, GLuint presentDurationId, GLenum type, GLenum target0, GLuint fill0, GLintptr fill0Offset, GLuint key0, GLintptr key0Offset, GLenum target1, GLuint fill1, GLintptr fill1Offset, GLuint key1, GLintptr key1Offset);
GLAPI void  glPresentFrameDualFillBuffersNV (GLuint video_slot, GLuint64EXT minPresentTime, GLuint beginPresentTimeId, GLuint presentDurationId, GLenum type, GLenum target0, GLuint fill0, GLintptr fill0Offset, GLenum target1, GLuint fill1, GLintptr fill1Offset, GLenum target2, GLuint fill2, GLintptr fill2Offset, GLenum target3, GLuint fill3, GLintptr fill3Offset);
GLAPI void  glPresentVideoParameterivNV (GLuint video_slot, GLenum pname, const GLint *params);
GLAPI GLenum  glVideoCaptureNV (GLuint video_capture_slot, GLuint *sequence_num, GLuint64EXT *capture_time);
GLAPI void  glVideoCaptureStreamParameterivNV (GLuint video_capture_slot, GLuint stream, GLenum pname, const GLint *params);
GLAPI void  glVideoCaptureStreamParameteruivNV (GLuint video_capture_slot, GLuint stream, GLenum pname, const GLuint *params);
GLAPI void  glVideoCaptureStreamParameterfvNV (GLuint video_capture_slot, GLuint stream, GLenum pname, const GLfloat *params);
GLAPI void  glVideoCaptureStreamParameterdvNV (GLuint video_capture_slot, GLuint stream, GLenum pname, const GLdouble *params);
#endif


CaptureOptions::CaptureOptions()
{}

CaptureOptions::CaptureOptions( int screen )
{
    GLfloat mat[4][4];
    float scale = 1.0f;
    //GLfloat scmax[] = {5000, 5000, 5000, 5000};
    GLfloat scmax[] = {256, 256, 256, 256};
    GLfloat scmin[] = {0, 0, 0, 0};
    // Initialize matrix to the identity.
    mat[0][0] = scale; mat[0][1] = 0; mat[0][2] = 0; mat[0][3] = 0;
    mat[1][0] = 0; mat[1][1] = scale; mat[1][2] = 0; mat[1][3] = 0;
    mat[2][0] = 0; mat[2][1] = 0; mat[2][2] = scale; mat[2][3] = 0;
    mat[3][0] = 0; mat[3][1] = 0; mat[3][2] = 0; mat[3][3] = scale;
    GLfloat offset[] = {0, 0, 0, 0};
    mat[0][0] = 1.164f *scale;
    mat[0][1] = 1.164f *scale;
    mat[0][2] = 1.164f *scale;
    mat[0][3] = 0;

    mat[1][0] = 0;
    mat[1][1] = -0.392f *scale;
    mat[1][2] = 2.017f *scale;
    mat[1][3] = 0;

    mat[2][0] = 1.596f *scale;
    mat[2][1] = -0.813f *scale;
    mat[2][2] = 0.f;
    mat[2][3] = 0;

    mat[3][0] = 0;
    mat[3][1] = 0;
    mat[3][2] = 0;
    mat[3][3] = 1;

    offset[0] =-0.87f;
    offset[1] = 0.53026f;
    offset[2] = -1.08f;
    offset[3] = 0;
    
	cscMax = scmax;
	cscMin = scmin;
	cscMat = (float*)mat;
	cscOffset = offset;
	captureType = BUFFER_FRAME;
	bufferInternalFormat =  GL_YCBYCR8_422_NV;
	bitsPerComponent = NV_CTRL_GVI_BITS_PER_COMPONENT_8;
	sampling = NV_CTRL_GVI_COMPONENT_SAMPLING_422;
	xscreen = screen;
	bDualLink = false;
	bChromaExpansion = false;

}

CNvSDIin::CNvSDIin()
:m_numActiveJacks(0), m_numJacks(0)
{
  m_videoSlot = 1;
  // Setup CSC for each stream.
  float scale = 1.0f;
  
  m_cscMax[0] = 5000; m_cscMax[1] = 5000; 
  m_cscMax[2] = 5000; m_cscMax[3]= 5000;
  m_cscMin[0] = 0; m_cscMin[1] = 0; 
  m_cscMin[2]= 0; m_cscMin[3] = 0;
  
  // Initialize matrix to the identity.
  m_cscMat[0][0] = scale; m_cscMat[0][1] = 0; 
  m_cscMat[0][2] = 0; m_cscMat[0][3] = 0;
  m_cscMat[1][0] = 0; m_cscMat[1][1] = scale; 
  m_cscMat[1][2] = 0; m_cscMat[1][3] = 0;
  m_cscMat[2][0] = 0; m_cscMat[2][1] = 0; 
  m_cscMat[2][2] = scale; m_cscMat[2][3] = 0;
  m_cscMat[3][0] = 0; m_cscMat[3][1] = 0; 
  m_cscMat[3][2] = 0; m_cscMat[3][3] = scale;
  
  m_captureOptions.bufferInternalFormat = GL_RGBA8;
  m_captureOptions.textureInternalFormat = GL_RGBA8;
  m_captureOptions.pixelFormat = GL_RGBA;	
  m_captureOptions.bDualLink = false;
  m_captureOptions.bChromaExpansion = true;
  m_captureOptions.bitsPerComponent = NV_CTRL_GVI_BITS_PER_COMPONENT_8;
  m_captureOptions.sampling = NV_CTRL_GVI_COMPONENT_SAMPLING_422;
  m_captureOptions.captureType = TEXTURE_FRAME;
  m_captureOptions.xscreen = 0;
  m_bCaptureStarted = false;
}


CNvSDIin::~CNvSDIin()
{
}

void CNvSDIin::setCaptureOptions(Display *display, const CaptureOptions &captureOptions)	
{
  dpy = display;
  if(captureOptions.cscMat)
    memcpy(m_cscMat,captureOptions.cscMat,sizeof(GLfloat)*16);
  if(captureOptions.cscOffset)
    memcpy(m_cscOffset,captureOptions.cscOffset,sizeof(GLfloat)*4);	
  if(captureOptions.cscMin)
    memcpy(m_cscMin,captureOptions.cscMin,sizeof(GLfloat)*4);	
  if(captureOptions.cscMax)
    memcpy(m_cscMax,captureOptions.cscMax,sizeof(GLfloat)*4);	
  m_captureOptions.bufferInternalFormat = captureOptions.bufferInternalFormat;
  m_captureOptions.textureInternalFormat = captureOptions.textureInternalFormat;
  m_captureOptions.pixelFormat = captureOptions.pixelFormat;	
  
  m_captureOptions.bDualLink = captureOptions.bDualLink;
  m_captureOptions.bitsPerComponent = captureOptions.bitsPerComponent;
  m_captureOptions.bChromaExpansion = captureOptions.bChromaExpansion;
  m_captureOptions.sampling = captureOptions.sampling;
  
  m_captureOptions.captureType = captureOptions.captureType;
  
  m_captureOptions.xscreen = captureOptions.xscreen;

}

GLboolean CNvSDIin::initCaptureDeviceGL()
{
    GLXVideoCaptureDeviceNV *VideoInDevices;
    unsigned int *VideoOutDevices;
    int numDevices = 0;
    //
    // I n i t i a l i z e  V i d e o  C a p t u r e  D e v i c e
    //
    // Enumerate available video capture devices
    VideoInDevices = glXEnumerateVideoCaptureDevicesNV( dpy, m_captureOptions.xscreen, &numDevices );

    if( !VideoInDevices || numDevices <= 0)
    {
        printf("No video capture devices found.\n");
        return GL_FALSE;
    }

    if( m_numStreams == 0 )
    {
        printf("No stream found.\n");
        return GL_FALSE;
    }
    
    // Choose first device found.  Free device list.
    m_hCaptureDevice = VideoInDevices[0];
    XFree(VideoInDevices);

    // Lock video capture device.
    glXLockVideoCaptureDeviceNV(dpy, m_hCaptureDevice);
  
  
    // Bind video capture device to the current OpenGL rendering context.
    if( Success != glXBindVideoCaptureDeviceNV(dpy, m_videoSlot, m_hCaptureDevice))
    {
        printf("Could not bind video input device\n");
        return GL_FALSE;
    }

    for (int i=0; i < m_numStreams; i++)
    {
        // Set the buffer object capture data format for each stream.
        glVideoCaptureStreamParameterfvNV(m_videoSlot, i,
                          GL_VIDEO_COLOR_CONVERSION_MATRIX_NV,
                          &m_cscMat[0][0]);

        assert(glGetError() == GL_NO_ERROR);

        glVideoCaptureStreamParameterfvNV(m_videoSlot, i,
                          GL_VIDEO_COLOR_CONVERSION_MAX_NV,&m_cscMax[0]);

        assert(glGetError() == GL_NO_ERROR);

        glVideoCaptureStreamParameterfvNV(m_videoSlot, i,
                          GL_VIDEO_COLOR_CONVERSION_MIN_NV,&m_cscMin[0]);

        assert(glGetError() == GL_NO_ERROR);

        glVideoCaptureStreamParameterfvNV(m_videoSlot, i,
                          GL_VIDEO_COLOR_CONVERSION_OFFSET_NV,&m_cscOffset[0]);

        assert(glGetError() == GL_NO_ERROR);
    }


    if( m_captureOptions.captureType == BUFFER_FRAME )
    {
        // Create video buffer objects for each stream
        glGenBuffersARB(m_numStreams, m_vidBuf);
        assert(glGetError() == GL_NO_ERROR);

        for (int i = 0; i < m_numStreams; i++)
        {
            // Set the buffer object capture data format.
            glVideoCaptureStreamParameterivNV(m_videoSlot, i,
                          GL_VIDEO_BUFFER_INTERNAL_FORMAT_NV,
                          &m_captureOptions.bufferInternalFormat);

            assert(glGetError() == GL_NO_ERROR);

            // Get the video buffer pitch
            glGetVideoCaptureStreamivNV(m_videoSlot, i, GL_VIDEO_BUFFER_PITCH_NV,
                        &m_bufPitch[i]);
            
            assert(glGetError() == GL_NO_ERROR);

            // Bind the buffer
            glBindBufferARB(GL_VIDEO_BUFFER_NV, m_vidBuf[i]);

            // Allocate required space in video capture buffer
            glBufferDataARB(GL_VIDEO_BUFFER_NV, m_bufPitch[i] * m_videoHeight,
                NULL, GL_STREAM_READ_ARB);


            // Bind the buffer to the video capture device.
            glBindVideoCaptureStreamBufferNV(m_videoSlot, i, GL_FRAME_NV, 0);
        }
    }  
    else //capture to textures
    {
        // Create video capture texture object(s)
        glGenTextures(m_numStreams, m_vidTex);
        assert(glGetError() == GL_NO_ERROR);

        // Bind the input capture textures for each stream
        for (int i = 0; i < m_numStreams; i++)
        {
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, m_vidTex[i]);
            assert(glGetError() == GL_NO_ERROR);

            glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            assert(glGetError() == GL_NO_ERROR);

            // Get frame size. - This can be done here in OpenGL.
            // Of this can be done with NV Control as it is in initVideoIn().
            glGetVideoCaptureStreamivNV(m_videoSlot, i,
                        GL_VIDEO_CAPTURE_FRAME_WIDTH_NV,
                        &m_videoWidth);
            assert(glGetError() == GL_NO_ERROR);

            glGetVideoCaptureStreamivNV(m_videoSlot, i,
                        GL_VIDEO_CAPTURE_FRAME_HEIGHT_NV,
                        &m_videoHeight);
            assert(glGetError() == GL_NO_ERROR);

            // Initialize texture.
            glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, m_captureOptions.textureInternalFormat, m_videoWidth, m_videoHeight,
                 0, m_captureOptions.pixelFormat, GL_UNSIGNED_BYTE, NULL);
            assert(glGetError() == GL_NO_ERROR);


            glBindVideoCaptureStreamTextureNV(m_videoSlot, i, GL_FRAME_NV,
                          GL_TEXTURE_RECTANGLE_NV, m_vidTex[i]);
            assert(glGetError() == GL_NO_ERROR);
      }
    }
  return GL_TRUE;
}


bool CNvSDIin::startCapture()
{
  // Start video capture
  glBeginVideoCaptureNV(m_videoSlot);
  glGenQueries(1,&m_captureTimeQuery);
  GLenum err = glGetError();
  //assert(err == GL_NO_ERROR);
  if(err == GL_NO_ERROR)
    m_bCaptureStarted = true;
  else
    {
      printf("Got OGL error: 0x%x - couldn't start capture\n",err);
      m_bCaptureStarted = false;
    }
  return m_bCaptureStarted;
}



GLenum CNvSDIin::capture(GLuint *sequenceNum, GLuint64EXT *captureTime)
{	
	GLenum ret;
 	// Make OpenGL context current.
	GLuint64EXT captureTimeStart;
	GLuint64EXT captureTimeEnd;	

	// Capture the video to a buffer object	
	glBeginQuery(GL_TIME_ELAPSED_EXT,m_captureTimeQuery);		
	glGetInteger64v(GL_CURRENT_TIME_NV,(GLint64EXT *)&captureTimeStart);

	ret = glVideoCaptureNV(m_videoSlot, sequenceNum, captureTime);		

	glGetInteger64v(GL_CURRENT_TIME_NV,(GLint64EXT *)&captureTimeEnd);
	glEndQuery(GL_TIME_ELAPSED_EXT);	
	m_gviTime = (captureTimeEnd - *captureTime)*.000000001;
	GLuint64EXT timeElapsed;
	glGetQueryObjectui64vEXT(m_captureTimeQuery, GL_QUERY_RESULT, &timeElapsed);
	m_gpuTime = timeElapsed*.000000001;	
    
    // Drop frames if the frame rate is less than 2 frames per second 
    float frameRate = 2.f;
    float captureLatency=m_gviTime;
    //int frameNumber=0;
    while(captureLatency>1.5/frameRate)
    {
	    ret = glVideoCaptureNV(m_videoSlot, sequenceNum, &captureTimeStart);	
	    glGetInteger64v(GL_CURRENT_TIME_NV,(GLint64EXT *)&captureTimeEnd);
        captureLatency = (captureTimeEnd-captureTimeStart)*0.000000001;
        //printf("Dropping frame %d\n", frameNumber);
        //frameNumber++;
    }
	return ret;	
}

bool CNvSDIin::endCapture()
{
  if(m_bCaptureStarted)
    {
      //assert(glGetError() == GL_NO_ERROR);
      glEndVideoCaptureNV(m_videoSlot);
      //assert(glGetError() == GL_NO_ERROR);
      glDeleteQueries(1,&m_captureTimeQuery);
      //assert(glGetError() == GL_NO_ERROR);
      m_bCaptureStarted = false;
    }
  return true;
}


bool CNvSDIin::destroyCaptureDeviceNVCtrl()
{
  return true;
}


GLboolean CNvSDIin::destroyCaptureDeviceGL()
{
  //glXMakeCurrent(dpy,NULL,NULL);
  //glXDestroyContext(dpy,ctx);
  // Release video capture device.
  glXReleaseVideoCaptureDeviceNV(dpy, m_hCaptureDevice);
  if(m_captureOptions.captureType == BUFFER_FRAME)
    {	
      // Delete video buffer objects
      glDeleteBuffersARB(m_numStreams, m_vidBuf);
    }
  else
    {
      // Delete video texture objects
      glDeleteTextures(m_numStreams, m_vidTex);
    }
  return GL_TRUE;
}

bool CNvSDIin::initCaptureDeviceNVCtrl()
{
  int major, minor;
  int numVideoIn;

  // Check NV-CONTROL X availability.
  if (!XNVCTRLIsNvScreen(dpy, m_captureOptions.xscreen)) {
    fprintf(stderr, "The NV-CONTROL X not available on screen "
	    "%d of '%s'.\n", m_captureOptions.xscreen, XDisplayName(NULL));
    return GL_FALSE;
  }

  // Query NV-CONTROL X version.
  if (!XNVCTRLQueryVersion(dpy, &major, &minor)) {
    fprintf(stderr, "The NV-CONTROL X extension does not exist on '%s'.\n",
	    XDisplayName(NULL));
    return GL_FALSE;
  }

  // Print NV-CONTROL X version information.
  printf("\n");
  printf("Using NV-CONTROL extension %d.%d on %s\n", 
	 major, minor, XDisplayName(NULL));

  // Query number of SDI video capture devices in the system.
  if (!XNVCTRLQueryTargetCount(dpy, NV_CTRL_TARGET_TYPE_GVI, &numVideoIn)) {
    fprintf(stderr, "No video capture devices available.\n");
    return GL_FALSE;
  }

  // Choose the first available GVI device (target_id = 0).
  // Print number of video capture devices available.
  fprintf(stderr, "Number of video capture devices available: %d\n",
	  numVideoIn);

  // Query the GVI identifier.
  if (!XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI,
				   0, 0, NV_CTRL_GVI_GLOBAL_IDENTIFIER,
				   &m_hGVI)) {
    fprintf(stderr, "Unable to query global GVI identifier.\n");
  }

  // Query the number of available jacks on the SDI video capture device
  // For now, simply query the first video capture device found.
  if (!XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI,
                                   0, 0, NV_CTRL_GVI_NUM_JACKS, 
                                   &m_numJacks)){
    fprintf(stderr, "Unable to query available jacks on video capture device.\n");
    return GL_FALSE;
  }

  // Print number of available video jacks found on the capture device
  fprintf(stderr, "Number of available jacks: %d\n", m_numJacks);
  
  return GL_TRUE;
}


GLboolean CNvSDIin::initInputDeviceGL()
{
    // Determine the number of active jacks.  Active jacks are defined
    // as those detecting a video input signal other than
    // NV_CTRL_GVIO_VIDEO_FORMAT_NONE.
    m_numActiveJacks = 0;

    bool activeJacks[4] = {0, 0, 0, 0};  // Assume a maximum of 4 for now.
    int value = NV_CTRL_GVIO_VIDEO_FORMAT_NONE;
    for (int i = 0; i < m_numJacks; i++)
    {
    // Query signal format
    if (!XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI,
                0, i, NV_CTRL_GVIO_DETECTED_VIDEO_FORMAT,
                     &value))
    {
      fprintf(stderr, "Jack %d : Cannot detect input video format\n", i+1);
    }
    else
    {
      // Assumes all jacks have same video format.
      if (value != NV_CTRL_GVIO_VIDEO_FORMAT_NONE) {
        fprintf(stderr, "Jack %d : %s\n", i + 1, decodeSignalFormat(value));
        m_numActiveJacks++;
        m_videoFormat = value;
        activeJacks[i] = true;
      } // if
    } // else
    } // for

    // Set the number of streams to the number of active jacks.
    m_numStreams = m_numActiveJacks;

    int numLinks = 1;
    if(m_captureOptions.bDualLink)
    {
    m_numStreams >>= 1;
    numLinks = 2;
    }

    // Print number of active video jacks found on the capture device
    fprintf(stderr, "Number of active jacks: %d\n", m_numActiveJacks);
    // Configure all active jacks as single link stream.
    //
    // In the case of 3G, a single stream is composed of
    // two channels jackX.0 and jack X.1. 3G signals must
    // be connected to physical jacks 1 and 3, logical
    // jacks 0 and 2.
    //
    // In the case of SD and HD, each single link stream
    // utilizes only  a single channel.
    char instr[255];
    char *outstr;
    switch(m_videoFormat) {
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_60_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_50_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_98_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_3G_LEVEL_B_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_3G_LEVEL_B_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_50_00_3G_LEVEL_A_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_A_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_60_00_3G_LEVEL_A_SMPTE274:
    // Only two streams of 3G Level A are supported.  Inputs must
    // be connected on physical jacks 1 and 3 or logical jacks 0 and 2.
    switch (m_numStreams) {
        case 1:
            if (!activeJacks[0] && !activeJacks[2]) {
                printf("WARNING: 3G input supported on jacks 1 and 3 only!\n");
                printf("WARNING: Please connect 3G inputs to jacks 1 or 3.\n");
            }
            if (activeJacks[0]) {
                strcpy(instr, "stream=0, link0=jack0.0, link1=jack0.1");
            }
            if (activeJacks[2]) {
                strcpy(instr, "stream=0, link0=jack2.0, link1=jack2.1");
            }
            break;
        case 2:
            if (!activeJacks[0] || ! activeJacks[2]) {
                printf("WARNING: 3G input supported on jacks 1 and 3 only!\n");
                printf("WARNING: Please connect 3G inputs connected to jacks 1 and 3.\n");
            }
            strcpy(instr, "stream=0, link0=jack0.0, link1=jack0.1; stream=1, link0=jack2.0, link1=jack2.1");
            break;
        default:
           fprintf(stderr, "Illegal number of streams specified.\n");
        } // switch
    break;
    default:
    switch (m_numStreams) {
        case 1:
            if(numLinks == 2) //dual link
                strcpy(instr, "stream=0, link0=jack0, link1=jack1");
            else
                strcpy(instr, "stream=0, link0=jack0");
            break;
        case 2:
            if(numLinks == 2)//dual link
                strcpy(instr, "stream=0, link0=jack0, link1=jack1; stream=1, link0=jack2, link1=jack3");
            else
                strcpy(instr, "stream=0, link0=jack0; stream=1, link0=jack1");
            break;
        case 3:
          strcpy(instr,
             "stream=0, link0=jack0; stream=1, link0=jack1; stream=2, link0=jack2");
          break;
        case 4:
          strcpy(instr,
             "stream=0, link0=jack0; stream=1, link0=jack1; stream=2, link0=jack2; stream=3, link0=jack3");
          break;
        default:
          fprintf(stderr, "Illegal number of streams specified.\n");
        } // switch
    } // swtich
  if (!XNVCTRLStringOperation(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, 0,
			     NV_CTRL_STRING_OPERATION_GVI_CONFIGURE_STREAMS,
			     instr, &outstr)) {
		fprintf(stderr, "Error configuring input jacks as specified streams\n");
        return GL_FALSE;
  } else {
		fprintf(stderr, "Jack configuration successful\n");
  }
  XFree(outstr);

  // Configure sampling for each stream.
  for (int i = 0; i < m_numStreams; i++)
  {
    //
    // Set desired parameters
    //
    // Signal format.
    XNVCTRLSetTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
			      NV_CTRL_GVIO_REQUESTED_VIDEO_FORMAT,
			      m_videoFormat);

    //Bits per component
    XNVCTRLSetTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
			      NV_CTRL_GVI_REQUESTED_STREAM_BITS_PER_COMPONENT,
			      m_captureOptions.bitsPerComponent);


	// Component sampling
    XNVCTRLSetTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
			      NV_CTRL_GVI_REQUESTED_STREAM_COMPONENT_SAMPLING,
			      m_captureOptions.sampling);


    // Chroma sampling
    if(m_captureOptions.bChromaExpansion)
      XNVCTRLSetTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
				NV_CTRL_GVI_REQUESTED_STREAM_CHROMA_EXPAND,
				NV_CTRL_GVI_CHROMA_EXPAND_TRUE);


    else
      XNVCTRLSetTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
				NV_CTRL_GVI_REQUESTED_STREAM_CHROMA_EXPAND,
				NV_CTRL_GVI_CHROMA_EXPAND_FALSE);

    //
    // Query set parameters
    //

    // Signal format
    XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
				NV_CTRL_GVIO_REQUESTED_VIDEO_FORMAT,
				&value);

    fprintf(stderr, "Jack %d : Video format: %s\n", i+1,
	    decodeSignalFormat(value));


    // Query the width and height of the input signal format
    XNVCTRLQueryAttribute(dpy, m_captureOptions.xscreen, value,
			  NV_CTRL_GVIO_VIDEO_FORMAT_WIDTH,
			  &m_videoWidth);



    XNVCTRLQueryAttribute(dpy, m_captureOptions.xscreen, value,
			  NV_CTRL_GVIO_VIDEO_FORMAT_HEIGHT,
			  &m_videoHeight);

    // Component sampling.
    XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
				NV_CTRL_GVI_REQUESTED_STREAM_COMPONENT_SAMPLING,
				&value);


    fprintf(stderr, "Jack %d : Sampling:: %s\n", i+1,
	    decodeComponentSampling(value));

    // Chroma sampling
    XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
				NV_CTRL_GVI_REQUESTED_STREAM_CHROMA_EXPAND,
				&value);


    fprintf(stderr, "Jack %d : Chroma Expand:: %s\n", i+1,
		decodeChromaExpand(value));


    // Chroma sampling
    XNVCTRLQueryTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
				NV_CTRL_GVI_REQUESTED_STREAM_BITS_PER_COMPONENT,
				&value);

    fprintf(stderr, "Jack %d : Bits per component:: %s\n", i+1,
		decodeBitsPerComponent(value));

    // Number of ring buffers
    XNVCTRLSetTargetAttribute(dpy, NV_CTRL_TARGET_TYPE_GVI, 0, i,
            NV_CTRL_GVI_NUM_CAPTURE_SURFACES,
            1);
    fprintf(stderr, "Number of capture surface:: 1\n");
  } // for numStreams


    return GL_TRUE;
}

#if defined __cplusplus
} /* extern "C" */
#endif








