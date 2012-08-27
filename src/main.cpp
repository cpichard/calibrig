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
#include "StereoAnalyzerCPU.h"
#include "StereoAnalyzerGPU.h"
#include "NvSDIin.h"

#include "ImageProcessing.h"

#include "CudaUtils.h"
#include "Monitor.h"
#include "QuadViewScreen.h"
#include "DiffScreen.h"
#include "HistogramScreen.h"
#include "Utils.h"
#include "Grabber.h"
#include "MainWindow.h"
#include <boost/thread.hpp>
#include "NetworkServer.h"
#include "CommandStack.h"

#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

namespace po = boost::program_options;

#define TEST 0

const char *version = "27082012";

int main(int argc, char *argv[])
{
    std::cout << "calibrig v" << version << " - cpu + gpu beta" << std::endl;
    std::cout << "Copyright (C) 2010-2012  C. Pichard"<< std::endl;
    std::cout << "This program comes with ABSOLUTELY NO WARRANTY;" << std::endl;
    std::cout << "This is free software, and you are welcome to redistribute it" << std::endl;
    std::cout << "under certain conditions; " << std::endl;

    unsigned int tcpPort = 8090;
    unsigned int udpPort = 8091;
    bool useGPU = false;
    bool noThread = false;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("port", po::value<unsigned int>(), "tcp server port - deprecated, use --tcp")
        ("tcp", po::value<unsigned int>(), "tcp server port")
        ("udp", po::value<unsigned int>(), "udp server port")
        ("gpu", "enable gpu computing")
        ("nothread", "remove multi threading - for debugging purposes")
    ;

	// Parse command line
    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if( vm.count("help") )
        {
            std::cout << desc << "\n";
            return 1;
        }

        if( vm.count("port") )
        {
            tcpPort = vm["port"].as<unsigned int>();
            std::cout << "WARNING Option --port is deprecated, please use --tcp instead" << std::endl;
        }

        if( vm.count("tcp") )
        {
            tcpPort = vm["tcp"].as<unsigned int>();
        }

        if( vm.count("udp") )
        {
            udpPort = vm["udp"].as<unsigned int>();
        }

        if( vm.count("gpu") )
        {
            useGPU = true;
        }
        if( vm.count("nothread") )
        {
            noThread = true;
        }

        std::cout << "Using tcp port " << tcpPort << std::endl;
        std::cout << "Using udp port " << udpPort << std::endl;
    }

    catch(std::exception &e)
    {
        std::cout << e.what() << std::endl;
        exit(1);
    }

#if TEST
    // Development, force use of GPU mode
    useGPU = true;
#endif

    // Value for the server
    CommandStack commandStack;
    Command currentCommand;
    LockDecorator<AnalysisResult> sharedResult;
    MessageHandler msgHandler(sharedResult, commandStack);
    NetworkServer server(msgHandler, tcpPort, udpPort);
    boost::thread t(boost::ref(server));

    // Default window size
    const UInt2 winSize(1024,768);

    // Connect to X server
    Display *dpy = XOpenDisplay(NULL);
    if( dpy == NULL )
    {
        ERROR_INFO( "Couldn't find X11 display - existing" );
        exit(0);
    }
    else
    {
        SUCCESS_INFO( "X11 display opened" );
    }

    // Check system devices and configuration
    if( checkSystem( dpy ) == false )
    {
        ERROR_INFO( "Invalid devices - exiting" );
        XCloseDisplay(dpy);
        exit(0);
    }
    else
    {
        SUCCESS_INFO( "System checked" );
    }


    // Scan the systems for GPUs
    HGPUNV gpuList[MAX_GPUS];
    int	num_gpus = ScanHW( dpy, gpuList );
    if( num_gpus < 1 )
    {
        ERROR_INFO( "No GPU found - exiting" );
        XCloseDisplay(dpy);
		exit(1);
    }

    // Grab the first GPU for now for DVP
    HGPUNV *gpu = &gpuList[0];

    // Create window
    GLXContext ctx;
    Window mainWin = createMainWindow( dpy, ctx, gpu->deviceXScreen, Width(winSize), Height(winSize) );

    // Register interest in the close window message
    Atom wmDeleteMessage = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, mainWin, &wmDeleteMessage, 1);

    // Setup CUDA
    CUcontext cuContext;
    if( cudaInitDevice(cuContext) == false )
    {
        ERROR_INFO( "No CUDA device available - exiting" );
        XCloseDisplay(dpy);
        exit(1);
    }

    // Screens
    QuadViewScreen  *screen1 = new QuadViewScreen( dpy, winSize );
    DiffScreen      *screen2 = new DiffScreen( dpy, winSize );
    HistogramScreen *screen3 = new HistogramScreen( dpy, winSize );
    ScreenLayout    *activeScreen = NULL;
    activeScreen = screen1;

    // Create an image grabber
    Grabber grabber(dpy, gpu, ctx) ;
    if( grabber.init() )
    {
        screen1->resizeImage(grabber.videoSize());
        screen2->resizeImage(grabber.videoSize());

        // Set capture handle to all screens
        activeScreen->setVideoStreams( grabber.stream1(), grabber.stream2() );
    }

#if TEST
    ImageGL m_YTmp;
    ImageGL m_warpedYTmp;
    if( ! allocBufferAndTexture( m_YTmp, grabber.videoSize() ) )
        return 0;

    if( ! allocBufferAndTexture( m_warpedYTmp, grabber.videoSize() ) )
        return 0;

    // TODO : fill matrix and add matrix to warp function
    float matrix[9];
    float *d_matrix; // device matrix
    cudaMalloc((void**)&d_matrix, sizeof(float)*9);
#endif

    // Create an analyzer
    StereoAnalyzer *analyzer = NULL;
    if(useGPU)
        analyzer = new StereoAnalyzerGPU();
    else
        analyzer = new StereoAnalyzerCPU();

    // Create a thread to run analysis on background
    // Launch analyser in background
    analyzer->resizeImages( grabber.videoSize() );
    AnalyzerFunctor runAnalysis( *analyzer );
    boost::thread *analysisThread=NULL;
    if(noThread == false)
    {
        analysisThread = new boost::thread( boost::ref(runAnalysis) );
    }
    ComputationData *result = NULL;

    // Main XWindows event loop
    XEvent event;
    bool bNotDone = true;
    bool captureOK = false;
    bool saveImages = false;

    while( bNotDone )
    {
        // flush all pending events
        while(XPending(dpy))
        {
            XNextEvent(dpy, &event);

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
                    if( kpe->keycode == 65 ) //SPACE
                    {
                    }
                    // Key_1
                    if( kpe->keycode == 10 )
                    {
                        activeScreen = screen1;
                    }
                    // Key_2
                    if( kpe->keycode == 11 )
                    {
                        activeScreen = screen2;
                    }
                    // Key_3
                    if( kpe->keycode == 12 )
                    {
                        activeScreen = screen3;
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
                    if(event.xclient.data.l[0] == wmDeleteMessage)
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
        captureOK = grabber.captureVideo();

        // for now ....
        if( captureOK == false )
        {
            ERROR_INFO( "Unable to capture image - exiting" );
            bNotDone = false;
        }

        // Analysis
        if( captureOK == true )
        {
            if(saveImages)
            {
                grabber.saveImages();
                saveImages = false;
            }

            if(analyzer->try_lock())
            {
                ComputationData *newResult = analyzer->acquireLastResult();
                if( newResult != NULL )
                {

                    if(result != NULL)
                        activeScreen->setResult(NULL);
                        delete result;

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

                    analyzer->updateLeftImageWithSDIVideo (grabber.stream1());

#if TEST            // Transform image for tests
                    convertYCbYCrToY( grabber.stream2(), m_YTmp );
                    matrix[0] = rand()%10*5;
                    matrix[1] = rand()%10*6;
                    matrix[2] = 1.f;//(rand()%10)*2.f/10.f;
                    //std::cout << "matrix = " << matrix[0] << std::endl;
                    cudaMemcpy(d_matrix, matrix, sizeof(float)*9, cudaMemcpyHostToDevice);
                    warpImage(m_YTmp, m_warpedYTmp, d_matrix );
                    convertYToYCbYCr(m_warpedYTmp,m_YTmp);
                    analyzer->updateRightImageWithSDIVideo (m_YTmp);
#else
                    analyzer->updateRightImageWithSDIVideo(grabber.stream2());
#endif
                    analyzer->processImages();
                }

                analyzer->unlock();
                if(noThread==true)
                {
                    analyzer->analyse();
                }

            }

            // Next frame
            activeScreen->nextFrame();
        }

        activeScreen->draw();

        // Swap buffer
        glXSwapBuffers(dpy, mainWin);

    }

#if TEST
        releaseBufferAndTexture( m_YTmp );
        releaseBufferAndTexture( m_warpedYTmp );
#endif
    
    // Wait for thread to stop
    if(noThread==false)
    {
        analyzer->lock();
        runAnalysis.stop();
        analyzer->unlock();
        analysisThread->join();
        delete analysisThread;
        analysisThread = NULL;
        std::cout << "Using threaded analyzer" << std::endl;
    }
	delete analyzer;

    // Free screens
    delete screen1;
    delete screen2;
    delete screen3;

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

    // Shutdown grabber
    grabber.shutdown();

    // Wait for result server to stop
    server.stop();
    t.join();

    // Disconnect from X server
    XCloseDisplay( dpy );

    return 1;
}
