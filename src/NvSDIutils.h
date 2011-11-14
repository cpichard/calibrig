#ifndef __NVSDIUTILS_H_
#define __NVSDIUTILS_H_


#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>

#include <list>
#include <iostream>
#include <sstream> 
#include <string>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>

typedef
struct HGPUNV
{	
    int  deviceXScreen;
    char deviceName[256];
}HGPUNV;


int ScanHW(Display *dpy, HGPUNV * gpuList);
const char *decodeBitsPerComponent(int _value);
const char *decodeChromaExpand(int _value);
const char *decodeColorSpace(int _value);
const char *decodeComponentSampling(int _value);
const char *decodeSignalFormat(int _value);
const char *decodeSDISyncInputDetected(int _value);
GLfloat CalcFPS();


#endif // __NVSDIUTILS_H_