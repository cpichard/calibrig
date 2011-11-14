#ifndef __MAINWINDOW_H__
#define __MAINWINDOW_H__

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>
#include <GL/glut.h>

//! Creates an X11 window
Window createMainWindow( Display *dpy, GLXContext &ctx, int xscreen, unsigned int windowWidth, unsigned int windowHeight );

// To be done
//void deleteMainWindow();

#endif//__MAINWINDOW_H__
