#include "VertexBufferObject.h"
#include "SurfDescriptor.h"

bool allocBuffer( VertexBufferObject &vbo, unsigned int nbElement )
{
    // TODO 
    if( NbElements(vbo) != nbElement && nbElement != 0 )
    {
        // Allocate VBO ( vertex buffer object )
        glGenBuffersARB( 1, &(vbo.m_bufId) );
        assert(glGetError() == GL_NO_ERROR);
        glBindBufferARB( GL_ARRAY_BUFFER_ARB, vbo.m_bufId );
        assert(glGetError() == GL_NO_ERROR);
        MemSize(vbo) = nbElement * 2 * sizeof(float); // nbElement*(x,y)*float
        glBufferDataARB( GL_ARRAY_BUFFER_ARB, MemSize(vbo), NULL, GL_DYNAMIC_DRAW );
        assert(glGetError() == GL_NO_ERROR);

        glBindBufferARB( GL_ARRAY_BUFFER_ARB, 0 );
        assert(glGetError() == GL_NO_ERROR);
        NbElements(vbo) = nbElement;
        Capacity(vbo) = nbElement;
        return true;
    }

   return false;

}

bool releaseBuffer( VertexBufferObject &vbo )
{
    glDeleteBuffersARB(1, &vbo.m_bufId);
    vbo.m_bufId = 0;
    NbElements(vbo) = 0;
    Capacity(vbo) = 0;
    MemSize(vbo) = 0;
    return true;
}

void drawObject( VertexBufferObject &vbo )
{
    if(NbElements(vbo)==0)
        return;
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, BufId(vbo) );
    assert(glGetError() == GL_NO_ERROR);
    // activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    assert(glGetError() == GL_NO_ERROR);
    glVertexPointer(2, GL_FLOAT, 0, 0);
    assert(glGetError() == GL_NO_ERROR);
    // draw points
    glDrawArrays(GL_POINTS, 0, NbElements(vbo));
    assert(glGetError() == GL_NO_ERROR);
    // deactivate vertex arrays after drawing
    glDisableClientState(GL_VERTEX_ARRAY);
    assert(glGetError() == GL_NO_ERROR);
    glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
    assert(glGetError() == GL_NO_ERROR);
}

bool copyVertexBuffer( VertexBufferObject &src, VertexBufferObject &dst )
{
    assert( MemSize(src) == MemSize(dst) );

    // Map buffer
    CudaDevicePtrWrapper<VertexBufferObject,void*> inDevicePtr(src);
    CudaDevicePtrWrapper<VertexBufferObject,void*> outDevicePtr(dst);

    cudaMemcpy( (void*)outDevicePtr, (void*)inDevicePtr, MemSize(src), cudaMemcpyDeviceToDevice );

    dst.m_nbElements = src.m_nbElements;

    return true;
}
