/*
 * calibrig - compute geometric deformation between two video flux
 * Copyright (C) 2010-2011 Cyril Pichard
 * cyril.pichard at gmail dot com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>

#include <iomanip>
#include <iostream>
#include <cstdlib>

#include "AnalyzerFunctor.h"
#include "HomographyAnalyzerCPU.h"
#include "HomographyAnalyzerGPU.h"
#include "FMatrixAnalyzerCPU.h"
#include "NvSDIin.h"

#include "ImageProcessing.h"

#include "CudaUtils.h"
#include "Monitor.h"
#include "QuadViewScreen.h"
#include "DiffScreen.h"
#include "HistogramScreen.h"
#include "HistogramScreen.h"
#include "AnaglyphScreen.h"
#include "MixScreen.h"
#include "VisualComfortScreen.h"
#include "GrabberSDI.h"
#include "GrabberTest.h"
#include <boost/thread.hpp>
#include "NetworkServer.h"
#include "CommandStack.h"
#include "ProgramOptions.h"

#include "GraphicSystemX11.h"

const char *version = "14042012";


int main(int argc, char *argv[])
{
    // Parse command line for program options
    ProgramOptions po(argc, argv, version);

    // Network server variables 
    LockDecorator<Deformation> sharedResult;
    CommandStack commandStack;
    NetworkServer server(sharedResult, commandStack, po.m_serverPort);
    boost::thread t(boost::ref(server));

    // Graphic system, X11, GPUs, GLX, basic window manager
    GraphicSystemX11 gs(po.m_winSize);

    // Setup CUDA
    CUcontext cuContext;
    if( cudaInitDevice(cuContext) == false )
    {
        std::cerr << "No CUDA device available - exiting" << std::endl;
        exit(EXIT_FAILURE);
    }
   
    // First tests with cuda stream, needs lot of refactoring before 
    //cudaStream_t streams[2];
    //cudaStreamCreate(&streams[0]);
    //cudaStreamCreate(&streams[1]);

    // gs.m_display -> GraphicRenderContext ? gs.m_display + cuda + gl ?yy

    // Screens : TODO a vector of screen ?
    // TODO : gs.addScreenLayout(screen1,screen2, screen3, ...)
    QuadViewScreen      *screen1 = new QuadViewScreen( gs.m_display, po.m_winSize );
    DiffScreen          *screen2 = new DiffScreen( gs.m_display, po.m_winSize );
    AnaglyphScreen      *screen3 = new AnaglyphScreen( gs.m_display, po.m_winSize );
    MixScreen           *screen4 = new MixScreen( gs.m_display, po.m_winSize ); 
    HistogramScreen     *screen5 = new HistogramScreen( gs.m_display, po.m_winSize );
    VisualComfortScreen *screen6 = new VisualComfortScreen( gs.m_display, po.m_winSize );
    ScreenLayout    *activeScreen = screen1;

    // Create an image grabber
    //Grabber *grabber = new GrabberSDI(gs.m_display, gs.m_gpu, gs.m_glxContext);
    Grabber *grabber = new GrabberTest(gs.m_display, gs.m_gpu, gs.m_glxContext);
    
    if( grabber->init() )
    {
        // Init all screens with the size of the grabbed image
        screen1->resizeImage(grabber->videoSize());
        screen2->resizeImage(grabber->videoSize());
        screen3->resizeImage(grabber->videoSize());
        screen4->resizeImage(grabber->videoSize());
        screen5->resizeImage(grabber->videoSize());
        screen6->resizeImage(grabber->videoSize());

        // Set capture handle to all screens
        activeScreen->setVideoStreams( grabber->stream1(), grabber->stream2() );
    }

#if TEST
    ImageGL m_YTmp;
    ImageGL m_warpedYTmp;
    if( ! allocBufferAndTexture( m_YTmp, grabber->videoSize() ) )
        return 0;

    if( ! allocBufferAndTexture( m_warpedYTmp, grabber->videoSize() ) )
        return 0;

    // TODO : fill matrix and add matrix to warp function
    float matrix[9];
#endif
    
    // Create an analyzer
    StereoAnalyzer *analyzer = NULL;
    //if(po.m_useGPU)
    //    analyzer = new HomographyAnalyzerGPU();
    //else
    //    analyzer = new HomographyAnalyzerCPU();
    // Testing a new analyser
    analyzer = new FMatrixAnalyzerCPU();

    // Create a thread to run analysis on background
    // Launch analyser in background
    analyzer->resizeImages( grabber->videoSize() );
    AnalyzerFunctor runAnalysis( *analyzer, cuContext, gs.m_display, gs.m_glxContext );
    boost::thread *analysisThread = NULL;
    // If option is use thread (by default)
    if(po.m_noThread == false)
    {
        analysisThread = new boost::thread( boost::ref(runAnalysis) );
    }

    // Variable needed in the loop
    ComputationData *result = NULL;
    Command currentCommand;
    
    // Main XWindows event loop
    XEvent event;
    bool bNotDone = true;
    bool captureOK = false;
    bool saveImages = false;

    while( bNotDone )
    {
        // while( eventManager.continue() )
        // graphicSystem.processX11Events()
        // eventManager.dispatch()
        // flush all pending events
        while(XPending(gs.m_display))
        {
            XNextEvent(gs.m_display, &event);

            //printf("Event: %d\n", event.type);
            switch(event.type)
            {
                case KeyPress:
                  {
                    XKeyEvent *kpe  = (XKeyEvent *)&event;
                    //	printf("keycode = %d\n", kpe->keycode);

                    // ESC
                    if( kpe->keycode == 9 )
                    {
                        bNotDone = false;
                    }

                    // SPACE
                    else if( kpe->keycode == 65 ) //SPACE
                    {
                    }
                    // Key_1
                    else if( kpe->keycode == 10 ) 
                    {
                        activeScreen = screen1;
                    }
                    // Key_2
                    else if( kpe->keycode == 11 ) 
                    {
                        activeScreen = screen2;
                    }
                    // Key_3
                    else if( kpe->keycode == 12 ) 
                    {
                        activeScreen = screen3;
                    }
                    // Key_4
                    else if( kpe->keycode == 13 ) 
                    {
                        activeScreen = screen4;
                    }
                    // Key_5
                    else if( kpe->keycode == 14 ) 
                    {
                        activeScreen = screen5;
                    }
                    // Key_6
                    else if( kpe->keycode == 15 ) 
                    {
                        activeScreen = screen6;
                    }
                    //std::cout << "KeyPress = " << kpe->keycode << std::endl;

                  }
                  break;
                case ConfigureNotify:
                  {
                    // Resize
                    XConfigureEvent *ce = (XConfigureEvent *)&event;
                    if(activeScreen->winWidth()  != ce->width
                    || activeScreen->winHeight() != ce->height )
                    {
                        activeScreen->resizeWindow( UInt2(ce->width,ce->height));
                        activeScreen->draw();
                    }
                  }
                  break;
                case ClientMessage:
                  {
                    if(event.xclient.data.l[0] == gs.m_wmDeleteMessage)
                    {
                        bNotDone = false;    
                    }
                  }
                  break;
                default:
                  ;
	  	//printf("Event: %d\n", event.type);
            } // switch
        }

        // Flush all received commands 
        while( commandStack.popCommand(currentCommand) == true )
        {
            if( currentCommand.m_dest == "MAIN" 
            && currentCommand.m_action == "EXIT" )
            {
                bNotDone = false;
            }

            else if( currentCommand.m_dest == "MAIN"
            && currentCommand.m_action == "SNAPSHOT")
            {
                saveImages = true;    
            }

            else if( currentCommand.m_dest == "MAIN"
            && currentCommand.m_action == "SCREEN")
            {
                int screenNumber = currentCommand.m_value;
                switch(screenNumber)
                {
                case 1:
                    activeScreen = screen1;
                    break;
                case 2:
                    activeScreen = screen2;
                    break;
                case 3:
                    activeScreen = screen3;
                    break;
                case 4:
                    activeScreen = screen4;
                    break;
                case 5:
                    activeScreen = screen5;
                    break;
                case 6:
                    activeScreen = screen6;
                    break;
                default:
                    // do nothing
                    break;
                } 
            }

            // Redirect command for analyser
            else if( currentCommand.m_dest == "ANALYSER" )
            {
                analyzer->acceptCommand(currentCommand);
            }
        }

        // Capture video
        captureOK = grabber->captureVideo();

        // for now ....
        if( captureOK == false )
        {
            std::cerr << "Unable to capture image - exiting" << std::endl;
            bNotDone = false;
        }

        // Analysis
        if( captureOK == true )
        {
            if(saveImages)
            {
                grabber->saveImages();
                saveImages = false;    
            }

            if(analyzer->try_lock())
            {
                ComputationData *newResult = analyzer->acquireLastResult();
                if( newResult != NULL )
                {

                    if(result != NULL)
                        analyzer->disposeResult(result);
                    result = newResult;
                    newResult = NULL;

                    // Give new results to analyser
                    activeScreen->setResult(result);
                    activeScreen->updateResult();

                    // Give new result to tcp server
                    sharedResult.set( result->m_d );
                }

                if( analyzer->imagesAreNew() == false )
                {

                    analyzer->updateLeftImageWithSDIVideo (grabber->stream1());

#if TEST            // Transform image for testing
                    convertYCbYCrToY( grabber->stream2(), m_YTmp );
                    matrix[0] = rand()%10*5;
                    matrix[1] = rand()%10*6;
                    matrix[2] = 1.f;//(rand()%10)*2.f/10.f;
                    //std::cout << "matrix = " << matrix[0] << std::endl;
                    warpImage(m_YTmp, m_warpedYTmp, matrix );
                    convertYToYCbYCr(m_warpedYTmp,m_YTmp);
                    analyzer->updateRightImageWithSDIVideo (m_YTmp);
#else
                    analyzer->updateRightImageWithSDIVideo(grabber->stream2());
#endif
                }

                analyzer->unlock();
                if(po.m_noThread==true)
                {
                    analyzer->analyse();
                }
                
            }

            // Next frame
            activeScreen->nextFrame();
        }
        
        activeScreen->draw();

        // Swap buffer
        glXSwapBuffers(gs.m_display, gs.m_mainWin);

    }

#if TEST
        releaseBufferAndTexture( m_YTmp );
        releaseBufferAndTexture( m_warpedYTmp );
#endif
    
    // Wait for thread to stop
    if(po.m_noThread==false)
    {
        analyzer->lock();
        runAnalysis.stop();
        analyzer->unlock();
        analysisThread->join();
        delete analysisThread;
        analysisThread = NULL;
    }
	delete analyzer;

    // Free screens
    delete screen1;
    delete screen2;
    delete screen3;
    delete screen4;
    delete screen5;
    delete screen6;

    // TODO : free OpenGL memory
    // Test in CPU
    // Free last result
    if(result != NULL)
    {
        delete result;
        result = NULL;
    }

    // ???
    //cudaReleaseDevice(cuContext);
    //cudaStreamDestroy(streams[0]);
    //cudaStreamDestroy(streams[1]);

    // Shutdown grabber
    grabber->shutdown();

    // Wait for result server to stop
    server.stop();
    t.join();

    return EXIT_SUCCESS;
}
