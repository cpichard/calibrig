#ifndef __CUDAUTILS_H__
#define __CUDAUTILS_H__

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>
#include "Utils.h"

void _checkError( CUresult err, const char * filename, const int linenum);
void _checkLastError(const char * filename, const int linenum);

bool cudaInitDevice(CUcontext &cuContext);
void cudaReleaseDevice(CUcontext &cuContext);
// GL Stuff
//1280 GL_INVALID_ENUM
//1281 GL_INVALID_VALUE
//1282 GL_INVALID_OPERATION
//1283 GL_STACK_OVERFLOW
//1284 GL_STACK_UNDERFLOW
//1285 GL_OUT_OF_MEMORY


//define to add the line & filename to the error output
#define checkError(err) _checkError(err, __FILE__, __LINE__)
#define checkLastError() _checkLastError( __FILE__, __LINE__)
#endif
