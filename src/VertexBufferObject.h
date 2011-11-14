#ifndef __VERTEXBUFFEROBJECT_H__
#define __VERTEXBUFFEROBJECT_H__

#include "Utils.h"
#include <GL/gl.h>
#include <cuda.h>
#include <CudaUtils.h>
#include "ImageGL.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <cudaGL.h>
#include <cuda_runtime.h>
#include <cudpp.h>


struct VertexBufferObject
{
    VertexBufferObject()
    :  m_bufId(0),
       m_memSize(0),
       m_nbElements(0),
       m_capacity(0),
       m_elementType(GL_POINTS) // default
    {}

    GLuint   		m_bufId;        // Buffer id
    unsigned int        m_memSize;      // memSize in bytes
    unsigned int        m_nbElements;   // nbElements
    unsigned int        m_capacity;     // Capacity in elements
    unsigned int        m_elementType;  // vertex type : Points ? quads ?
    
    GLuint & bufId(){ return m_bufId; }
    const GLuint & bufId() const { return m_bufId; }
 
};

bool allocBuffer( VertexBufferObject &vbo, unsigned int vboSize );
bool releaseBuffer( VertexBufferObject &vbo );

// Simple wrappers, easy to remember
inline unsigned int & MemSize( VertexBufferObject &vbo ){ return vbo.m_memSize; }
//inline unsigned int & Size( VertexBufferObject &vbo ){ return vbo.m_size; }
inline unsigned int & NbElements( VertexBufferObject &vbo ){ return vbo.m_nbElements; }
inline unsigned int & Capacity( VertexBufferObject &vbo ){ return vbo.m_capacity; }

void drawObject( VertexBufferObject &vbo );
bool copyVertexBuffer( VertexBufferObject &src, VertexBufferObject &dst );

#endif//__VERTEXBUFFEROBJECT_H__
