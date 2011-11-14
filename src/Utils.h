#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <X11/Xlib.h> // Display

// Here be the dragons
// All the functions in this file must be moved to some good place

// Simple structure
// TODO : use template for other types if needed
// use std::pair ?
typedef 
struct UInt2
{
    UInt2( const UInt2 &a )
    :m_x(a.m_x),m_y(a.m_y){}
    
    UInt2( unsigned int x, unsigned int y )
    :m_x(x),m_y(y){}

    UInt2 & operator = ( const UInt2 &a ){ m_x=a.m_x; m_y = a.m_y; return *this;}

    unsigned int m_x;
    unsigned int m_y;
} UInt2;

inline UInt2 SwapXY( const UInt2 &a ){ return UInt2(a.m_y,a.m_x); }

static UInt2 ZeroUInt2(0,0);

inline unsigned int & Width( UInt2 &s ){ return s.m_x; }
inline unsigned int & Height( UInt2 &s ){ return s.m_y; }
inline const unsigned int & Width( const UInt2 &s ){ return s.m_x; }
inline const unsigned int & Height( const UInt2 &s ){ return s.m_y; }


inline
bool operator != ( const UInt2 &a, const UInt2 &b )
{
    return a.m_x != b.m_x || a.m_y != b.m_y ;
}


// Do we really need these ?
// Prefer to create a log struct
inline void WARNING_INFO( const char *msg1 )
{
    std::cerr << msg1 << std::endl;
}

inline void ERROR_INFO( const char *msg1 )
{
    std::cerr << msg1 << std::endl;
}

inline void SUCCESS_INFO( const char *msg1 )
{
    std::cerr << msg1 << std::endl;
}

// Check glx, xwin, etc. only call in main.cpp
bool checkSystem( Display * );

#endif//__UTILS_H__
