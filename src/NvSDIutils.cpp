#include "NvSDIutils.h"
#include "NVCtrlLib.h"
#include "NVCtrl.h"



//assuming there is one XServer running in the system
int ScanHW(Display *dpy, HGPUNV * gpuList)
{
    HGPUNV gpuDevice;
    int num_gpus, num_screens;
    int gpu, screen;
    int mask;
    int *pData;
    int len, j;
    char *str=NULL;
    bool ret;

    /* Get the number of gpus in the system */
    ret = XNVCTRLQueryTargetCount(dpy, NV_CTRL_TARGET_TYPE_GPU, &num_gpus);
    if(!ret) 
    {
        fprintf(stderr, "Failed to query number of gpus\n");
        return 1;
    }
    printf("number of GPUs: %d\n", num_gpus);
    int num_gpusWithXScreen = 0;
    for(gpu = 0; gpu < num_gpus; gpu++)
    {
        printf("GPU %d information:\n", gpu);
        /* GPU name */
        ret = XNVCTRLQueryTargetStringAttribute(dpy,
            NV_CTRL_TARGET_TYPE_GPU,
            gpu, // target_id
            0, // display_mask
            NV_CTRL_STRING_PRODUCT_NAME,
            &str);
        if(!ret) 
        {
            fprintf(stderr, "Failed to query gpu product name\n");
            return 1;
        }
        printf("   Product Name                    : %s\n", str);
        /* X Screens driven by this GPU */
        ret = XNVCTRLQueryTargetBinaryData
            (dpy,
            NV_CTRL_TARGET_TYPE_GPU,
            gpu, // target_id
            0, // display_mask
            NV_CTRL_BINARY_DATA_XSCREENS_USING_GPU,
            (unsigned char **) &pData,
            &len);
        if(!ret) 
        {
            fprintf(stderr, "Failed to query list of X Screens\n");
            exit(EXIT_FAILURE);
        }
        printf("   Number of X Screens on GPU %d    : %d\n", gpu, pData[0]);
        //only return GPUs that have XScreens
        if(pData[0]) 
        {
            gpuDevice.deviceXScreen = pData[1]; //chose the first screen
            strcpy(gpuDevice.deviceName, str);
            gpuList[gpu] = gpuDevice;
            num_gpusWithXScreen++;
        }
        free(str);
        str=NULL;
        XFree(pData);
    }
    return num_gpusWithXScreen;
}
//
// Calculate fps
//

GLfloat CalcFPS()
{
    static int t0 = -1;
    static int count = 0;
    struct timeval tv;
    struct timezone tz;
    static GLfloat __fps = 0.0;
    int t;
    gettimeofday(&tv, &tz);
    t = (int) tv.tv_sec;
    if(t0 < 0) t0 = t;
    count++;
    if(t - t0 >= 5.0) {
        GLfloat seconds = t - t0;
        __fps = count / seconds;
        t0 = t;
        count = 0;
    }
    return (__fps);
}
//
// Decode SDI input value returned.
//

const char *decodeSDISyncInputDetected(int _value)
{
    switch(_value) {
        case NV_CTRL_GVO_SDI_SYNC_INPUT_DETECTED_HD:
            return ("NV_CTRL_GVO_SDI_SYNC_INPUT_DETECTED_HD");
            break;
        case NV_CTRL_GVO_SDI_SYNC_INPUT_DETECTED_SD:
            return ("NV_CTRL_GVO_SDI_SYNC_INPUT_DETECTED_SD");
            break;
        case NV_CTRL_GVO_SDI_SYNC_INPUT_DETECTED_NONE:
        default:
            return ("NV_CTRL_GVO_SDI_SYNC_INPUT_DETECTED_NONE");
            break;
    } // switch
}
//
// Decode provided signal format.
//

const char *decodeSignalFormat(int _value)
{
    switch(_value) {
        case NV_CTRL_GVIO_VIDEO_FORMAT_487I_59_94_SMPTE259_NTSC:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_480I_59_94_SMPTE259_NTSC");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_576I_50_00_SMPTE259_PAL:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_576I_50_00_SMPTE259_PAL");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_59_94_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_59_94_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_60_00_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_60_00_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1035I_59_94_SMPTE260:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1035I_59_94_SMPTE260");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1035I_60_00_SMPTE260:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1035I_60_00_SMPTE260");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE295:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE295");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_976_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_976_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_50_00_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_50_00_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_30_00_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_30_00_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_29_97_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_29_97_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_25_00_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_25_00_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_24_00_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_24_00_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_720P_23_98_SMPTE296:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_720P_23_98_SMPTE296");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_25_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_25_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_29_97_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_29_97_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_30_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_30_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_24_00_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_24_00_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_23_98_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_23_98_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_50_00_3G_LEVEL_A_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_50_00_3G_LEVEL_A_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_A_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_A_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_60_00_3G_LEVEL_A_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_A_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_60_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_60_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_50_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_50_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_59_94_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_98_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_98_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_3G_LEVEL_B_SMPTE274:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_3G_LEVEL_B_SMPTE274");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_3G_LEVEL_B_SMPTE372:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_3G_LEVEL_B_SMPTE372");
            break;
        case NV_CTRL_GVIO_VIDEO_FORMAT_NONE:
        default:
            return ("NV_CTRL_GVIO_VIDEO_FORMAT_NONE");
            break;
    } // switch
}
//
// Decode provided component sampling.
//

const char *decodeComponentSampling(int _value)
{
    switch(_value) {
        case NV_CTRL_GVI_COMPONENT_SAMPLING_UNKNOWN:
            return ("NV_CTRL_GVI_COMPONENT_SAMPLING_UNKNOWN");
            break;
        case NV_CTRL_GVI_COMPONENT_SAMPLING_4444:
            return ("NV_CTRL_GVI_COMPONENT_SAMPLING_4444");
            break;
        case NV_CTRL_GVI_COMPONENT_SAMPLING_4224:
            return ("NV_CTRL_GVI_COMPONENT_SAMPLING_4224");
            break;
        case NV_CTRL_GVI_COMPONENT_SAMPLING_444:
            return ("NV_CTRL_GVI_COMPONENT_SAMPLING_444");
            break;
        case NV_CTRL_GVI_COMPONENT_SAMPLING_422:
            return ("NV_CTRL_GVI_COMPONENT_SAMPLING_422");
            break;
        default:
            return ("NV_CTRL_GVI_COMPONENT_SAMPLING_UNKNOWN");
            break;
    } // switch
}
//
// Decode provided color space
//

const char *decodeColorSpace(int _value)
{
    switch(_value) {
        case NV_CTRL_GVI_COLOR_SPACE_GBR:
            return ("NV_CTRL_GVI_COLOR_SPACE_GBR");
            break;
        case NV_CTRL_GVI_COLOR_SPACE_GBRA:
            return ("NV_CTRL_GVI_COLOR_SPACE_GBRA");
            break;
        case NV_CTRL_GVI_COLOR_SPACE_GBRD:
            return ("NV_CTRL_GVI_COLOR_SPACE_GBRD");
            break;
        case NV_CTRL_GVI_COLOR_SPACE_YCBCR:
            return ("NV_CTRL_GVI_COLOR_SPACE_YCBCR");
            break;
        case NV_CTRL_GVI_COLOR_SPACE_YCBCRA:
            return ("NV_CTRL_GVI_COLOR_SPACE_YCBCRA");
            break;
        case NV_CTRL_GVI_COLOR_SPACE_YCBCRD:
            return ("NV_CTRL_GVI_COLOR_SPACE_YCBCRD");
            break;
        case NV_CTRL_GVI_COLOR_SPACE_UNKNOWN:
        default:
            return ("NV_CTRL_GVI_COLOR_SPACE_UNKNOWN");
            break;
    } // switch
}
//
// Decode bits per component
//

const char *decodeBitsPerComponent(int _value)
{
    switch(_value) {
        case NV_CTRL_GVI_BITS_PER_COMPONENT_UNKNOWN:
            return ("NV_CTRL_GVI_BITS_PER_COMPONENT_UNKNOWN");
            break;
        case NV_CTRL_GVI_BITS_PER_COMPONENT_8:
            return ("NV_CTRL_GVI_BITS_PER_COMPONENT_8");
            break;
        case NV_CTRL_GVI_BITS_PER_COMPONENT_10:
            return ("NV_CTRL_GVI_BITS_PER_COMPONENT_10");
            break;
        case NV_CTRL_GVI_BITS_PER_COMPONENT_12:
            return ("NV_CTRL_GVI_BITS_PER_COMPONENT_12");
            break;
        default:
            return ("NV_CTRL_GVI_BITS_PER_COMPONENT_UNKNOWN");
            break;
    } // switch
}
//
// Decode chroma expand
//

const char *decodeChromaExpand(int _value)
{
    switch(_value) {
        case NV_CTRL_GVI_CHROMA_EXPAND_FALSE:
            return ("NV_CTRL_GVI_CHROMA_EXPAND_FALSE");
            break;
        case NV_CTRL_GVI_CHROMA_EXPAND_TRUE:
            return ("NV_CTRL_GVI_CHROMA_EXPAND_TRUE");
            break;
        default:
            return ("NV_CTRL_GVI_CHROMA_EXPAND_UNKNOWN");
            break;
    } // switch
}
