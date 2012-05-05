#ifndef __IMAGEGL_H__
#define __IMAGEGL_H__

#include "Utils.h"
#include <cuda.h>
#include <CudaUtils.h>

// TODO clean code here
// Lots of code here can be put elsewhere
// Create new files for new types, etc ...

// Note only Texture ?
// Template ?? template< typename TextureFormat, typename TextureType, Size ? >
struct ImageGL
{
    ImageGL()
    :m_texId(0), m_bufId(0), m_size(0,0), m_texFormat(GL_RGBA), m_texType(GL_UNSIGNED_BYTE){}

    // CAREFULL HERE one image can be desallocated while others still points to bad infos
    ImageGL( const ImageGL &a )
    :m_texId(a.m_texId), m_bufId(a.m_bufId), m_size(a.m_size),
    m_texFormat(a.m_texFormat),m_texType(a.m_texType){}

    GLuint   m_texId;   //! Texture id
    GLuint   m_bufId;   //! Buffer id
    UInt2    m_size;    //! Image size
    GLuint   m_texFormat; // GL_RGB, GL_RGBA, GL_LUMINANCE, ...
    GLuint   m_texType;   // GL_BYTE, GL_FLOAT, GL_INT, ....

    GLuint & texFormat(){ return m_texFormat; }
    GLuint & texType(){ return m_texType; }

    GLuint & bufId(){ return m_bufId; }
    GLuint & texId(){ return m_texId; }
    const GLuint & bufId() const { return m_bufId; }
    const GLuint & texId() const { return m_texId; }
};

// 1 Plan floating point buffer used for surf
// Refactor to work with a lot of different buffer types
struct PixelBufferObject
{
    PixelBufferObject():m_bufId(0), m_size(0,0){}
    
    GLuint   m_bufId;   // Buffer
    UInt2    m_size;    // Image size
    
    GLuint & bufId(){ return m_bufId; }
    const GLuint & bufId() const { return m_bufId; }
};


template<typename WrappedType,typename MemoryType>
struct CudaDevicePtrWrapper
{
    CudaDevicePtrWrapper( const WrappedType &in )
    :m_in(in)
    {
        // NOTE : it may be inefficient to register buffer each time we need it . TODO Check !!!
        //        This code might change soon
        CUresult cerr = cuGraphicsGLRegisterBuffer( &m_cudaResource, BufId(m_in), CU_GRAPHICS_REGISTER_FLAGS_NONE );
        checkError(cerr);
        cerr = cuGraphicsMapResources( 1, &m_cudaResource, 0 );
        checkError(cerr);
        cerr = cuGraphicsResourceGetMappedPointer( &m_devicePtr, &m_bufferSize, m_cudaResource );
        checkError(cerr);
    }

    ~CudaDevicePtrWrapper()
    {
        CUresult cerr = cuGraphicsUnmapResources( 1, &m_cudaResource, 0 );
        checkError(cerr);
        cerr = cuGraphicsUnregisterResource( m_cudaResource ); 
        checkError(cerr);
    }

    operator MemoryType ()
    {
        return (MemoryType)(m_devicePtr);
    }


    const WrappedType   &m_in;
    CUgraphicsResource  m_cudaResource;
    CUdeviceptr         m_devicePtr;
    unsigned int        m_bufferSize;
};


template<typename Type >
struct CudaImageBuffer
{
    CudaImageBuffer():m_pitch(0), m_pitchInElements(0),m_ptr(0), m_size(0,0){}
    virtual ~CudaImageBuffer(){}

    size_t m_pitch;
    size_t m_pitchInElements;
    
    Type *m_ptr;

    operator Type * () const { return m_ptr;}

    UInt2    m_size;

    const Type *bufId() const { return m_ptr; }
};

template<typename Type, unsigned int NbPlanes >
struct LocalImagePlanes
{
    LocalImagePlanes( UInt2 &size )
    : m_size(size)
    {
        for(int i=0; i<NbPlanes; i++)
        {
            m_plane[i] = new CudaImageBuffer<Type>();
            m_allocated[i] = allocBuffer(*m_plane[i], size);
        }
    }

    ~LocalImagePlanes()
    {
        for(int i=0; i<NbPlanes; i++)
        {
            if(m_plane[i] != NULL)
            {
                if(m_allocated[i]==true)
                {
                    releaseBuffer( *m_plane[i] );
                    m_allocated[i]=false;
                }
                delete m_plane[i];
                m_plane[i] = NULL;
            }
        }
    }
    inline
    CudaImageBuffer<Type> & operator [] (unsigned int i){return *m_plane[i];}

    bool m_allocated[NbPlanes];
    CudaImageBuffer<Type> *m_plane[NbPlanes];
    UInt2    m_size;
};


template<typename Type>
struct LocalCudaImageBuffer : public CudaImageBuffer<Type>
{
    LocalCudaImageBuffer( UInt2 &size )
    : CudaImageBuffer<Type>()
    , m_allocated(false)
    {
        m_allocated = allocBuffer( *this, size );
    }

    virtual ~LocalCudaImageBuffer()
    {
        if( m_allocated )
            releaseBuffer( *this );
        m_allocated = false;
    }

    bool m_allocated;
};


template<typename Type>
inline
bool isAllocated( Type &img)
{
    return img.m_allocated;
}

template<typename Type>
bool allocBuffer( CudaImageBuffer<Type> &img, UInt2 &imgSize )
{
    if( Size(img) != imgSize && imgSize != UInt2(0,0) )
    {
        size_t dpitch = Width(imgSize) * sizeof(Type);
        cudaMallocPitch( (void**) &img.m_ptr, &img.m_pitch, dpitch, Height(imgSize));
        //std::cout << "Alloc " << img.m_ptr << std::endl;
        img.m_pitchInElements = img.m_pitch / sizeof(Type);
        img.m_size = imgSize;
        return true;
    }
    return false;
}

template<typename Type>
bool releaseBuffer( CudaImageBuffer<Type> &img )
{
    if( img.m_ptr != NULL )
    {
         //std::cout << "Released " << img.m_ptr << std::endl;
         cudaFree(img.m_ptr);
         img.m_ptr = NULL;
         img.m_size = UInt2(0,0);
         img.m_pitchInElements = img.m_pitch = 0;
         return true;
    }
    return false;
}

bool allocBuffer( PixelBufferObject &img, UInt2 &imgSize );
bool releaseBuffer( PixelBufferObject &img );

bool allocBufferAndTexture( ImageGL &img, UInt2 &imgSize );
bool releaseBufferAndTexture( ImageGL &img );

bool copyBufToTex( ImageGL & );

template<typename ImageType>
inline UInt2 & Size( ImageType &t ){ return t.m_size; }

template<typename ImageType, typename SizeType >
inline void SetSize( ImageType &size, SizeType newValue )
{ size.m_size = newValue; }

template<>
inline void SetSize( UInt2 &s, UInt2 newSize){ s = newSize;}

template<typename ImageType>
inline unsigned int & Width ( ImageType &t ){ return Width( Size(t) ); }

template<typename ImageType>
inline unsigned int & Height( ImageType &t ){ return Height( Size(t) ); }


template<typename ImageType>
inline void  SetWidth( ImageType &t, unsigned int w ){t.m_size.m_x = w;}

template<typename ImageType>
inline void  SetHeight( ImageType &t, unsigned int h ){t.m_size.m_y = h;}

inline GLuint & TexId( ImageGL &t ){ return t.m_texId; }

template< typename BufferType>
inline GLuint & BufId( BufferType &t ){ return t.m_bufId; }

template< typename BufferType>
inline void SetBufId( BufferType &t, GLuint bufId ){ t.m_bufId = bufId; }

template<typename TextureType>
inline bool HasTexture( const TextureType &t ) { return t.TexId() != 0; }

template< typename BufferType>
inline bool HasBuffer( const BufferType &t ) { return t.bufId() != 0; }

// Const versions
template<typename TextureType>
inline const GLuint & TexId( const TextureType &t ) { return t.TexId(); }

template<typename BufferType>
inline const GLuint & BufId( const BufferType &t ) { return t.bufId(); }



#endif //__IMAGEGL_H___
