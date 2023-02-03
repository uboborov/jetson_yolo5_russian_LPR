#include "NvUtils.h"
#include <errno.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <nvbuf_utils.h>

#include "common.h"
#include "decoder.h"
#include "rtsp_client.h"

#ifdef USE_CUDA_EGL
#include "yolov5/yolov5.h"
#include "yolov5/yololayer.h"
#endif

#define USE_FPS_MEASUREMENT

#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define MICROSECOND_UNIT 1000000
#define CHUNK_SIZE 4000000 //64*1024//
//#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
        !buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
        (buffer_ptr[2] == 1))

#define H264_NAL_UNIT_CODED_SLICE  1
#define H264_NAL_UNIT_CODED_SLICE_IDR  5

#define HEVC_NUT_TRAIL_N  0
#define HEVC_NUT_RASL_R  9
#define HEVC_NUT_BLA_W_LP  16
#define HEVC_NUT_CRA_NUT  21

#define IVF_FILE_HDR_SIZE   32
#define IVF_FRAME_HDR_SIZE  12

#define IS_H264_NAL_CODED_SLICE(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE)
#define IS_H264_NAL_CODED_SLICE_IDR(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE_IDR)

#define IS_MJPEG_START(buffer_ptr) (buffer_ptr[0] == 0xFF && buffer_ptr[1] == 0xD8)
#define IS_MJPEG_END(buffer_ptr) (buffer_ptr[0] == 0xFF && buffer_ptr[1] == 0xD9)

#define GET_H265_NAL_UNIT_TYPE(buffer_ptr) ((buffer_ptr[0] & 0x7E) >> 1)


using namespace std;

# ifndef MAX
#  define MAX(x, y) ( ((x)>(y))?(x):(y) )
# endif

# ifndef MIN
#  define MIN(x, y) ( ((x)<(y))?(x):(y) )
# endif


//static int measurement_en = 0;

/*
*/
unsigned long int time_diff(struct timespec *ts1, struct timespec *ts2) {
    static struct timespec ts;
    ts.tv_sec = MAX(ts2->tv_sec, ts1->tv_sec) - MIN(ts2->tv_sec, ts1->tv_sec);
    ts.tv_nsec = MAX(ts2->tv_nsec, ts1->tv_nsec) - MIN(ts2->tv_nsec, ts1->tv_nsec);

    if (ts.tv_sec > 0) {
        ts.tv_sec--;
        ts.tv_nsec += 1000000000;
    }

    return((ts.tv_sec * 1000000000) + ts.tv_nsec);
}

/*
 *
 */
static void decoder_abort(dec_context_t *ctx) {
    ctx->got_error = true;
    ctx->dec->abort();
#ifndef USE_NVBUF_TRANSFORM_API
    if (ctx->conv) {
        ctx->conv->abort();
        pthread_cond_broadcast(&ctx->queue_cond);
    }
#endif
}

/*
 *
 */
static void decoder_set_defaults(dec_context_t *ctx) {
    int i;
    memset(ctx, 0, sizeof(dec_context_t));
    for (i = 0;i < MAX_BUFFERS;i++) {
        ctx->dst_dma_fd[i] = -1;
    }
    ctx->fullscreen = false;
    ctx->window_height = 0;
    ctx->window_width = 0;
    ctx->window_x = 0;
    ctx->window_y = 0;
    ctx->out_pixfmt = 1;
    ctx->fps = 30;
    ctx->output_plane_mem_type = V4L2_MEMORY_MMAP;
    ctx->capture_plane_mem_type = V4L2_MEMORY_DMABUF;
    ctx->vp9_file_header_flag = 0;
    ctx->vp8_file_header_flag = 0;
    ctx->stress_test = 1;
    ctx->copy_timestamp = false;
    ctx->flag_copyts = false;
    ctx->start_ts = 0;
    ctx->file_count = 1;
    ctx->dec_fps = 30;
    //ctx->dst_dma_fd = -1;
    //memset(ctx->dst_dma_fd, -1, MAX_BUFFERS);
    ctx->bLoop = false;
    ctx->bQueue = false;
    ctx->loop_count = 0;
    ctx->max_perf = 0;
    ctx->extra_cap_plane_buffer = 1;
    ctx->blocking_mode = 1;
#ifndef USE_NVBUF_TRANSFORM_API
    ctx->conv_output_plane_buf_queue = new queue < NvBuffer * >;
    ctx->rescale_method = V4L2_YUV_RESCALE_NONE;
#endif
    ctx->current_frame = 0;
    pthread_mutex_init(&ctx->queue_lock, NULL);
    pthread_cond_init(&ctx->queue_cond, NULL);
#ifdef USE_CUDA_EGL
    ctx->dec_output_queue = new queue< int >;
#endif    
    ctx->cam_id = -1;
    ctx->use_aux_decoder = 0;
    ctx->nvosd_context = NULL;
    ctx->display_bbox = 0;
    ctx->enc_dma_fd = -1;
    ctx->detector_init_done = 0;
}



/*
 * read data from file or fifo
 */
static int pkt_num = 0;
static int first_time = 1;

static int read_decoder_input_chunk_file(int fd, NvBuffer * buffer) {
    int len = 0;
    int total = 0;
    int size;

    if (pkt_num++ < 4 && first_time) {
        size = 256 * 1024;
        while (total < size) {
            len = read(fd, buffer->planes[0].data + total, size - total);
            if (len == 0)
                break;
            total += len;
        }
    } else {
        first_time = 0;
        size = MIN(CHUNK_SIZE, buffer->planes[0].length);
        total = read(fd, buffer->planes[0].data + total, size - total);
    }

    

    /* NOTE: It is necessary to set bytesused properly, so that decoder knows how
             many bytes in the buffer are valid. */
    buffer->planes[0].bytesused = total;
    if(buffer->planes[0].bytesused == 0) {
        return -1;
    }
    return 0;
}

/*
 * 
 */
int decoder_lock(dec_context_t *ctx, pthread_mutex_t *mutex) { 
    struct timespec abs_time = {0}; 
    clock_gettime(CLOCK_REALTIME, &abs_time);
    abs_time.tv_sec += 10; 
    if (pthread_mutex_timedlock(mutex, &abs_time) == ETIMEDOUT) {
        ctx->got_eos = true;
        DBG_PRINT("TIMEOUT waiting data for DECODER\n");
        return -1;
    }
    
    //pthread_mutex_lock(mutex);
    return 0;
}

/*
 *
 */
int decoder_wait_input_data(dec_context_t *ctx, pthread_mutex_t *pmutex) {
    //pthread_mutex_lock(pmutex);
    return decoder_lock(ctx, pmutex);
}

/*
 *
 */
static int read_decoder_input_chunk_rtsp(dec_context_t *ctx, NvBuffer * buffer) {
    int len = 0;

    do {
        ctx->rtsp_data = (void *)buffer->planes[0].data;
        ctx->rtsp_len = 0;
        // wait until RTSP client release the data_ready mutex
        if (decoder_wait_input_data(ctx, &ctx->rtsp_data_ready_mutex) != 0) {
            return -1;
        }

        buffer->planes[0].bytesused = ctx->rtsp_len;

    } while(buffer->planes[0].bytesused == 0);

    if(buffer->planes[0].bytesused == 0) {
        return -1;
    }

    return 0;
}

/*
 *
 */
static int read_decoder_input_chunk_vlog(dec_context_t *ctx, NvBuffer * buffer) {
    int len = 0;

    do {
        ctx->vlog_data = (void *)buffer->planes[0].data;
        ctx->vlog_len = 0;

        if (decoder_wait_input_data(ctx, &ctx->vlog_data_ready_mutex) != 0) {
            return -1;
        }

        buffer->planes[0].bytesused = ctx->vlog_len;

    } while(buffer->planes[0].bytesused == 0);

    if(buffer->planes[0].bytesused == 0) {
        return -1;
    }

    return 0;
}

/*
 *
 */
int dump_dmabuf(int dmabuf_fd, unsigned int plane, std::ofstream * stream) {
    if (dmabuf_fd <= 0)
        return -1;

    int ret = -1;
    NvBufferParams parm;
    ret = NvBufferGetParams(dmabuf_fd, &parm);

    if (ret != 0) {
        DBG_PRINT("GetParams failed \n");
        return -1;
    }

    void *psrc_data;

    ret = NvBufferMemMap(dmabuf_fd, plane, NvBufferMem_Read_Write, &psrc_data);
    if (ret == 0) {
        unsigned int i = 0;
        NvBufferMemSyncForCpu(dmabuf_fd, plane, &psrc_data);
        for (i = 0; i < parm.height[plane]; ++i) {
            if((parm.pixel_format == NvBufferColorFormat_NV12 ||
                parm.pixel_format == NvBufferColorFormat_NV16 ||
                parm.pixel_format == NvBufferColorFormat_NV24 ||
                parm.pixel_format == NvBufferColorFormat_NV12_ER ||
                parm.pixel_format == NvBufferColorFormat_NV12_709 ||
                parm.pixel_format == NvBufferColorFormat_NV12_709_ER ||
                parm.pixel_format == NvBufferColorFormat_NV12_2020) &&
                plane == 1) {
                stream->write((char *)psrc_data + i * parm.pitch[plane], parm.width[plane] * 2);
                if (!stream->good())
                    return -1;
            } else {
                stream->write((char *)psrc_data + i * parm.pitch[plane], parm.width[plane]);
                if (!stream->good())
                    return -1;
            }
        }
        NvBufferMemUnMap(dmabuf_fd, plane, &psrc_data);
    } else {
        DBG_PRINT("NvBufferMap failed \n");
        return -1;
    }

    return 0;
}

/*
 *
 */
static void query_and_set_capture(dec_context_t * ctx) {
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    int32_t min_dec_capture_buffers;
    int ret = 0;
    int error = 0;
    uint32_t window_width;
    uint32_t window_height;
    NvBufferCreateParams input_params = {0};
    NvBufferCreateParams cParams = {0};

    ret = dec->capture_plane.getFormat(format);
    TEST_ERROR(ret < 0,"Error: Could not get format from decoder capture plane", error);

    ret = dec->capture_plane.getCrop(crop);
    TEST_ERROR(ret < 0, "Error: Could not get crop from decoder capture plane", error);

    DBG_PRINT("Video Resolution: %dx%d\n",crop.c.width, crop.c.height);

    ctx->display_height = crop.c.height;
    ctx->display_width = crop.c.width;

#ifdef USE_CUDA_EGL
    ctx->dma_egl_map.clear();
#endif      

#ifdef USE_NVBUF_TRANSFORM_API

     {
        NvBufferCreateParams i_params = {0};
        i_params.payloadType = NvBufferPayload_SurfArray;

# if defined(USE_CUDA_EGL) && defined(USE_YOLOV5_SIZE_TRANSFORM)
        i_params.width =  Yolo::INPUT_W;
        i_params.height = Yolo::INPUT_H;
# else    
        i_params.width = ctx->w;//crop.c.width;
        i_params.height = ctx->h;//crop.c.height;
# endif    
        i_params.layout = NvBufferLayout_Pitch;
        i_params.colorFormat = NvBufferColorFormat_YUV420;
        i_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;

        if (ctx->enc_dma_fd != -1) {
            NvBufferDestroy(ctx->enc_dma_fd);
        }

        NvBufferCreateEx(&ctx->enc_dma_fd, &i_params);
    }
    ctx->cur_odma_idx = 0;
    for(int index = 0 ; index < MAX_BUFFERS ; index++) {
        if(ctx->dst_dma_fd[index] != -1) {
            NvBufferDestroy(ctx->dst_dma_fd[index]);
            ctx->dst_dma_fd[index] = -1;
        }
    }
    /* Create PitchLinear output buffer for transform. */
    input_params.payloadType = NvBufferPayload_SurfArray;

    // NOTE: !!! set width and height the frame to be transformed to
#if defined(USE_CUDA_EGL) && defined(USE_YOLOV5_SIZE_TRANSFORM)
    input_params.width =  Yolo::INPUT_W;
    input_params.height = Yolo::INPUT_H;
#else    
    input_params.width = ctx->w;//crop.c.width;
    input_params.height = ctx->h;//crop.c.height;
#endif    
    input_params.layout = NvBufferLayout_Pitch;
    if (ctx->out_pixfmt == 1)
      input_params.colorFormat = NvBufferColorFormat_NV12;
    else if (ctx->out_pixfmt == 2)
      input_params.colorFormat = NvBufferColorFormat_YUV420;
    else if (ctx->out_pixfmt == 3)
      input_params.colorFormat = NvBufferColorFormat_NV16;
    else if (ctx->out_pixfmt == 4)
      input_params.colorFormat = NvBufferColorFormat_NV24;
    else if (ctx->out_pixfmt == 5)
      input_params.colorFormat = NvBufferColorFormat_ABGR32;

    input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;

    // NOTE: !!! DMA fds are common for decoder as output and encoder as input
    for(int index = 0 ; index < MAX_BUFFERS ; index++) {
        ret = NvBufferCreateEx(&ctx->dst_dma_fd[index], &input_params);
        TEST_ERROR(ret == -1, "create dmabuf failed", error);

#ifdef USE_CUDA_EGL
        CUresult status;
        ctx->egl_imagePtr[index] = NvEGLImageFromFd(ctx->egl_display, ctx->dst_dma_fd[index]);
        TEST_ERROR(ctx->egl_imagePtr[index] == NULL, "mapping dmabuf fd to EGLImage failed", error);
        ctx->pResource[index] = NULL;
        cudaFree(0);
        status = cuGraphicsEGLRegisterImage(&(ctx->pResource[index]), ctx->egl_imagePtr[index], CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        TEST_ERROR(status != CUDA_SUCCESS, "cuGraphicsEGLRegisterImage failed", error);
        status = cuGraphicsResourceGetMappedEglFrame(&(ctx->eglFramePtr[index]), ctx->pResource[index], 0, 0);
        TEST_ERROR(status != CUDA_SUCCESS, "cuGraphicsSubResourceGetMappedArray failed", error);
        cuCtxSynchronize();
        ctx->dma_egl_map.insert(pair<int, CUeglFrame>(ctx->dst_dma_fd[index], ctx->eglFramePtr[index]));
        //printf("egl ptr: %016lX <=> %016lX\n", ctx->dst_dma_fd[index], ctx->eglFramePtr[index]);
        //printf("idx: %d, w: %d, h: %d, chan: %d\n", index, ctx->eglFramePtr[index].width, ctx->eglFramePtr[index].height, ctx->eglFramePtr[index].numChannels);
#endif        
    }
#endif
    /* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
    dec->capture_plane.deinitPlane();
    if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF) {
        for(int index = 0 ; index < ctx->numCapBuffers ; index++) {
            if(ctx->dmabuff_fd[index] != 0) {
                ret = NvBufferDestroy (ctx->dmabuff_fd[index]);
                TEST_ERROR(ret < 0, "Failed to Destroy NvBuffer", error);
                ctx->dmabuff_fd[index] = 0;
            }
        }
    }

    /* Not necessary to call VIDIOC_S_FMT on decoder capture plane.
       But decoder setCapturePlaneFormat function updates the class variables */
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", error);

    ctx->video_height = format.fmt.pix_mp.height;
    ctx->video_width = format.fmt.pix_mp.width;
    /* Get the minimum buffers which have to be requested on the capture plane. */
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    TEST_ERROR(ret < 0,"Error while getting value of minimum capture plane buffers",error);

    /* Request (min + extra) buffers, export and map buffers. */
    if(ctx->capture_plane_mem_type == V4L2_MEMORY_MMAP) {
        
        ret = dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                          min_dec_capture_buffers + ctx->extra_cap_plane_buffer, 
                                          false,
                                          false);
        TEST_ERROR(ret < 0, "Error in decoder capture plane setup", error);
    } else if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF) {
        /* Set colorformats for relevant colorspaces. */
        switch(format.fmt.pix_mp.colorspace) {
            case V4L2_COLORSPACE_SMPTE170M:
                if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT) {
                    DBG_PRINT("Decoder colorspace ITU-R BT.601 with standard range luma (16-235)\n");
                    cParams.colorFormat = NvBufferColorFormat_NV12;
                } else {
                    DBG_PRINT("Decoder colorspace ITU-R BT.601 with extended range luma (0-255)\n");
                    cParams.colorFormat = NvBufferColorFormat_NV12_ER;
                }
                break;
            case V4L2_COLORSPACE_REC709:
                if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT) {
                    DBG_PRINT("Decoder colorspace ITU-R BT.709 with standard range luma (16-235)\n");
                    cParams.colorFormat = NvBufferColorFormat_NV12_709;
                } else {
                    DBG_PRINT("Decoder colorspace ITU-R BT.709 with extended range luma (0-255)\n");
                    cParams.colorFormat = NvBufferColorFormat_NV12_709_ER;
                }
                break;
            case V4L2_COLORSPACE_BT2020:
                DBG_PRINT("Decoder colorspace ITU-R BT.2020\n");
                cParams.colorFormat = NvBufferColorFormat_NV12_2020;
                break;
            default:
                DBG_PRINT("supported colorspace details not available, use default\n");
                if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT) {
                    DBG_PRINT("Decoder colorspace ITU-R BT.601 with standard range luma (16-235)\n");
                    cParams.colorFormat = NvBufferColorFormat_NV12;
                } else {
                    DBG_PRINT("Decoder colorspace ITU-R BT.601 with extended range luma (0-255)\n");
                    cParams.colorFormat = NvBufferColorFormat_NV12_ER;
                }
                break;
        }

        ctx->numCapBuffers = min_dec_capture_buffers + ctx->extra_cap_plane_buffer;

        cParams.width = crop.c.width;
        cParams.height = crop.c.height;
        cParams.layout = NvBufferLayout_BlockLinear;
        cParams.payloadType = NvBufferPayload_SurfArray;
        cParams.nvbuf_tag = NvBufferTag_VIDEO_DEC;

        if (ctx->decoder_pixfmt == V4L2_PIX_FMT_MJPEG) {
            cParams.layout = NvBufferLayout_Pitch;
            cParams.nvbuf_tag = NvBufferTag_JPEG;
            if (format.fmt.pix_mp.pixelformat == V4L2_PIX_FMT_YUV422M) {
                cParams.colorFormat = NvBufferColorFormat_YUV422;
            } else {
                cParams.colorFormat = NvBufferColorFormat_YUV420;
            }
        }

        /* Create decoder capture plane buffers. */
        for (int index = 0; index < ctx->numCapBuffers; index++) {
            ret = NvBufferCreateEx(&ctx->dmabuff_fd[index], &cParams);
            TEST_ERROR(ret < 0, "Failed to create buffers", error);
        }

        /* Request buffers on decoder capture plane.*/
        ret = dec->capture_plane.reqbufs(V4L2_MEMORY_DMABUF,ctx->numCapBuffers);
        TEST_ERROR(ret, "Error in request buffers on capture plane", error);
    }

    /* Decoder capture plane STREAMON.*/
    ret = dec->capture_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", error);

    /* Enqueue all the empty decoder capture plane buffers. */
    for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_buf.memory = ctx->capture_plane_mem_type;

        if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF) {
            v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[i];
        }
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
        TEST_ERROR(ret < 0, "Error Qing buffer at output plane", error);
    }  

    DBG_PRINT("Query and set capture successful\n");
    return;

error:
    if (error) {
        decoder_abort(ctx);
        DBG_PRINT("Error in %s \n", __func__ );
    }
}

static void *dec_capture_loop_fcn(void *arg) {
    dec_context_t *ctx = (dec_context_t *) arg;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_event ev;
    int ret;
    int nframes_ps = 0;
    unsigned long nframes_total = 0;

    DBG_PRINT("Starting decoder capture loop thread\n");

    do {
        ret = dec->dqEvent(ev, 50000);
        if (ret < 0) {
            if (errno == EAGAIN) {
                cerr << "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE" << endl;
            } else {
                cerr << "Error in dequeueing decoder event" << endl;
            }
            decoder_abort(ctx);
            break;
        }
    } while ((ev.type != V4L2_EVENT_RESOLUTION_CHANGE) && !ctx->got_error);

    DBG_PRINT("!!!!!!!!!!\n");

    /* Received the resolution change event, now can do query_and_set_capture. */
    if (!ctx->got_error) {
        query_and_set_capture(ctx);
        pthread_mutex_unlock(&ctx->init_mutex);
    }

    /* Exit on error or EOS which is signalled in main() */
    while (!(ctx->got_error || dec->isInError() || ctx->got_eos)) {
        NvBuffer *dec_buffer;

        /* Check for Resolution change again.
           Refer ioctl VIDIOC_DQEVENT */
        ret = dec->dqEvent(ev, false);
        if (ret == 0) {
            switch (ev.type) {
                case V4L2_EVENT_RESOLUTION_CHANGE:
                    query_and_set_capture(ctx);
                    continue;
            }
        }

        /* Decoder capture loop */
        while (!ctx->got_eos) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];
#ifdef USE_FPS_MEASUREMENT
            struct timespec tm[2];
            clock_gettime(CLOCK_MONOTONIC, &tm[0]);
#endif            

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;

            /* Dequeue a filled buffer. */

            //DBG_PRINT("DQ here 1\n");

            if (dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0)) {
                if (errno == EAGAIN) {
                    DBG_PRINT("-------\n");
                    usleep(1000);
                } else {
                    decoder_abort(ctx);
                    DBG_PRINT("Error while calling dequeue at capture plane\n");
                }
                break;
            }

            //DBG_PRINT("DQ here 2\n");

            /* If we need to write to file or display the buffer, give
               the buffer to video converter output plane instead of
               returning the buffer back to decoder capture plane. */
            if (ctx->out_file || (!ctx->disable_rendering && !ctx->stats)) {
                /* Clip & Stitch can be done by adjusting rectangle. */
                NvBufferRect src_rect, dest_rect;
                src_rect.top = 0;
                src_rect.left = 0;
                src_rect.width = ctx->display_width;
                src_rect.height = ctx->display_height;
                dest_rect.top = 0;
                dest_rect.left = 0;

                NvBufferTransformParams transform_params;
                memset(&transform_params,0,sizeof(transform_params));

#if defined(USE_CUDA_EGL) && defined(USE_YOLOV5_SIZE_TRANSFORM)
                // crop the central region from the source image
                src_rect.width = Yolo::INPUT_W;
                src_rect.height = Yolo::INPUT_H;
                src_rect.top = (ctx->display_height - Yolo::INPUT_H)/2;
                src_rect.left = (ctx->display_width - Yolo::INPUT_W )/2;

                transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER | NVBUFFER_TRANSFORM_CROP_SRC;
                transform_params.transform_flip = NvBufferTransform_None;
                transform_params.transform_filter = NvBufferTransform_Filter_Smart;
                transform_params.src_rect = src_rect;
                transform_params.dst_rect = dest_rect;
#else                
                dest_rect.width = ctx->display_width;
                dest_rect.height = ctx->display_height;
                
                /* Indicates which of the transform parameters are valid. */
                transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER | NVBUFFER_TRANSFORM_FLIP;
                transform_params.transform_flip = NvBufferTransform_None;
                transform_params.transform_filter = NvBufferTransform_Filter_Nearest;
#endif                
                int buf_idx = v4l2_buf.index;

                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    dec_buffer->planes[0].fd = ctx->dmabuff_fd[v4l2_buf.index];
                /* Perform Blocklinear to PitchLinear conversion. */

                //DBG_PRINT("DMA idx %d\n",ctx->cur_odma_idx);
                //ctx->cur_odma_idx = (ctx->cur_odma_idx + 1) % 10;

                ret = NvBufferTransform(dec_buffer->planes[0].fd, ctx->dst_dma_fd[ctx->cur_odma_idx], &transform_params);
                if (ret == -1) {
                    DBG_PRINT("=== DEC: Transform failed\n ==");
                    continue;
                    //break;
                }             

#ifdef USE_CUDA_EGL
                //if (!ctx->detector_busy) {
                if (ctx->input_type == DEC_INPUT_FILE) {
                    decoder_lock(ctx, &ctx->detector_ready_mutex);
                } else if (ctx->detector_busy) {
                    goto skip_cb;
                }

                ctx->dec_output_queue->push(ctx->dst_dma_fd[ctx->cur_odma_idx]);

                if (ctx->decoded_cb != NULL) {
                    ctx->decoded_cb(ctx, NULL);
                }
                //}
#else 
                if (ctx->decoded_cb != NULL) {
                    ctx->decoded_cb(ctx, NULL);
                }    
#endif    
                
                
skip_cb:
                /* If not writing to file, Queue the buffer back once it has been used. */
                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF) {
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                }

                if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                    decoder_abort(ctx);
                    DBG_PRINT("Error while queueing buffer at decoder capture plane\n");
                    break;
                }
            } else {
                DBG_PRINT("FRAME DECODED SUCCESSFULLY!");
                /* If not writing to file, Queue the buffer back once it has been used. */
                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF) {
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                }

                if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                    decoder_abort(ctx);
                    DBG_PRINT("Error while queueing buffer at decoder capture plane\n");
                    break;
                }
            }
#ifdef USE_FPS_MEASUREMENT
            /* measure fps of the capture device */
            clock_gettime(CLOCK_MONOTONIC, &tm[1]);

            int dt_ms = (time_diff(&tm[0], &tm[1])) / 1000000;
            if (dt_ms >= 1000) {
                if (ctx->measurement_en) {
                    DBG_PRINT("CAPTURE FPS: %d\n",nframes_ps);
                    ctx->measurement_en--;
                } 
                nframes_ps = 0;
            }
            nframes_ps++;
#endif
            nframes_total++;
        }
    }
lexit:

   
 
    ctx->loop_count = 1;
    DBG_PRINT("Exiting decoder capture loop thread\n");
    pthread_exit(NULL);
}

/*
 *
 */
static bool decoder_proc_blocking(dec_context_t &ctx, bool eos, int fd, struct buffer *pb) {
    int allow_DQ = true;
    int ret = 0;
    struct v4l2_buffer temp_buf;

    /* Since all the output plane buffers have been queued, we first need to
       dequeue a buffer from output plane before we can read new data into it
       and queue it again. */
    while (!eos && !ctx.got_error && !ctx.dec->isInError()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        /* dequeue a buffer for output plane. */
        if(allow_DQ) {
            ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
            if (ret < 0) {
                DBG_PRINT("Error DQing buffer at output plane\n");
                decoder_abort(&ctx);
                break;
            }
        } else {
            allow_DQ = true;
            memcpy(&v4l2_buf,&temp_buf,sizeof(v4l2_buffer));
            buffer = ctx.dec->output_plane.getNthBuffer(v4l2_buf.index);
        }

        if ((ctx.decoder_pixfmt == V4L2_PIX_FMT_H264) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_H265) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG2) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG4)) {
            
                /* read the input chunks. */
                //read_decoder_input_chunk(ctx.in_file[current_file], buffer);
                if (ctx.input_type == DEC_INPUT_FILE) {
                    eos = read_decoder_input_chunk_file(ctx.input_fd, buffer);
                } else if (ctx.input_type == DEC_INPUT_RTSP) {
                    eos = read_decoder_input_chunk_rtsp(&ctx, buffer);
                } 
                //DBG_PRINT("DMA idx %d\n",ctx.cur_odma_idx);
        }

        if (ctx.decoder_pixfmt == V4L2_PIX_FMT_MJPEG) {
            //read_mjpeg_decoder_input(ctx.in_file[current_file], buffer);
        }

        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        if (v4l2_buf.m.planes[0].bytesused == 0) {
            continue;
        }

        // signal capture loop to break after the last 
        // succeed decoding
        if (ctx.bLoop) ctx.got_eos = true;

        /* enqueue a buffer for output plane. */
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            DBG_PRINT("Error Qing buffer at output plane\n");
            decoder_abort(&ctx);
            break;
        } 

        if (v4l2_buf.m.planes[0].bytesused == 0) {
            eos = true;
            DBG_PRINT("Input file read complete\n");
            break;
        }

        // WARN: break while() after capture loop finished ONLY
        // otherwise capture loop hangs on capture_plane.dqBuffer() because of
        // incomplete data for decoding
        // thus we MUST feed it output_plane.qBuffer() until it's done
        if (ctx.bLoop && ctx.loop_count != 0) {
            eos = 1;
            break;
        }
    }
    return (eos|ctx.got_error);
}

#define DEF_VIDEO_DEV   "/dev/video1"
#define DEF_VIDEO_H     640
#define DEF_VIDEO_W     480
#define DEF_PIX_FMT     "H264"

static char VIDEO_DEV[20] = DEF_VIDEO_DEV;

#ifdef USE_CUDA_EGL
static bool init_display(EGLDisplay& egl_display) {
    // Get default EGL display
    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) {
        cout << "Error while get EGL display connection" << endl;
        return false;
    }

    // Init EGL display connection
    if (!eglInitialize(egl_display, NULL, NULL)) {
        cout << "Erro while initialize EGL display connection" << endl;
        return false;
    }

    return true;
}

static bool terminate_display(EGLDisplay& egl_display) {
    // Terminate EGL display connection
    if (egl_display) {
        if (!eglTerminate(egl_display)) {
            cerr << "Error while terminate EGL display connection\n" << endl;
            return false;
        }
    }
    return true;
}
#endif
/* 
 * decoder callback for encoder
 */
static int decoder_callback(struct _dec_context *ctx, void *out_data) {
    ctx->cur_odma_idx = (ctx->cur_odma_idx + 1) % ctx->num_odma_buf;
    if (!ctx->got_eos) {
        pthread_mutex_unlock(&ctx->data_ready_mutex);
    }
    return 0;
}

pthread_t p_main_decoder[4] = {0,0,0,0};


int setup_decoder(dec_context_t& dctx, uint32_t fmt, int w, int h, const char *in_file_path, int stage) {
    int error = 0;
    int ret = 0;
    int i;
    int eos = 0;

    DBG_PRINT("DECODER stage: %d\n", stage);

    switch(stage) {
        case 0: {
            // 0. default settings
            decoder_set_defaults(&dctx);
#ifdef USE_CUDA_EGL            
            if (init_display(dctx.egl_display) == false) goto cleanup;
            dctx.detector_busy = 1;
#endif           
        } return 0;

        case 1: {
            // 1. entire settings
#ifdef USE_CUDA_EGL
            dctx.out_pixfmt = 5; //ARGB
#else            
            dctx.out_pixfmt = 2; //YUV420P
#endif            
            dctx.decoder_pixfmt = fmt; //V4L2_PIX_FMT_H264
            dctx.decoded_cb = decoder_callback;
            dctx.w = w;
            dctx.h = h;

            dctx.dec = NvVideoDecoder::createVideoDecoder("dec0"); //blocking mode
            TEST_ERROR(!dctx.dec, "Could not create decoder", cleanup);

            ret = dctx.dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
            TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE", cleanup);

            ret = dctx.dec->setOutputPlaneFormat(dctx.decoder_pixfmt, CHUNK_SIZE);
            TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

            ret = dctx.dec->setFrameInputMode(1); // set chunked mode
            TEST_ERROR(ret < 0,"Error in decoder setFrameInputMode", cleanup);

            ret = dctx.dec->disableDPB(); // make decoder not freeze
            TEST_ERROR(ret < 0, "Error in decoder disableDPB", cleanup);

            ret = dctx.dec->setMaxPerfMode(1);
            TEST_ERROR(ret < 0,"Failed to DEC setMaxPerfMode", cleanup);

            if (dctx.output_plane_mem_type == V4L2_MEMORY_MMAP) {
                ret = dctx.dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
            } else if (dctx.output_plane_mem_type == V4L2_MEMORY_USERPTR) {
                ret = dctx.dec->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
            }
            TEST_ERROR(ret < 0, "Error while setting up output plane", cleanup);

            ret = dctx.dec->output_plane.setStreamStatus(true);
            TEST_ERROR(ret < 0, "Error in output plane stream on", cleanup);

            if (dctx.input_type == DEC_INPUT_RTSP) {
                pthread_mutex_lock(&dctx.rtsp_data_ready_mutex);
            } else if (dctx.input_type == DEC_INPUT_VLOG) {
                pthread_mutex_lock(&dctx.vlog_data_ready_mutex);
            } else if (dctx.input_type == DEC_INPUT_FILE) {
                //pthread_mutex_lock(&dctx.detector_ready_mutex);
            }
        } return 0;

        case 2: {
            // 2. queue buffers
            i = 0;
            while (!eos && !dctx.got_error && !dctx.dec->isInError() && i < dctx.dec->output_plane.getNumBuffers()) {
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];
                NvBuffer *buffer;
                memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                memset(planes, 0, sizeof(planes));

                buffer = dctx.dec->output_plane.getNthBuffer(i);

                if ((dctx.decoder_pixfmt == V4L2_PIX_FMT_H264) ||
                        (dctx.decoder_pixfmt == V4L2_PIX_FMT_H265) ||
                        (dctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG2) ||
                        (dctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG4)) {
                    // read frame here
                    if (dctx.input_type == DEC_INPUT_FILE) {
                        eos = read_decoder_input_chunk_file(dctx.input_fd, buffer);
                    } else if (dctx.input_type == DEC_INPUT_RTSP) {
                        eos = read_decoder_input_chunk_rtsp(&dctx, buffer);
                    }
                } else if (dctx.decoder_pixfmt == V4L2_PIX_FMT_MJPEG) {
                    //read MJPEG frame here
                } else {
                    continue;
                }

                v4l2_buf.index = i;
                v4l2_buf.m.planes = planes;
                v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

                ret = dctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
                if (ret < 0) {
                    DBG_PRINT("Error Qing buffer at output plane\n");
                    decoder_abort(&dctx);
                    break;
                }
                i++;
            }
            //TEST_ERROR(eos != 0, "Error waiting data", cleanup);
        } return 0;

        case 3: {
            // start capture/decoding loop
            pthread_mutex_lock(&dctx.init_mutex);
            pthread_create(&dctx.dec_capture_loop, NULL, dec_capture_loop_fcn, &dctx);

            // pend on init_mutex untill capture setup done
            pthread_mutex_lock(&dctx.init_mutex);
            pthread_mutex_unlock(&dctx.init_mutex);
        } return 0;

        default: return 0;
    }
    return 0;
cleanup:    
    return -1;
}

#ifdef USE_CUDA_EGL 

#endif
/*
 *
 */
static void *thread_main_decoder(void *arg) {
    int ret = 0;
    int error = 0;
    bool eos = false;
    int i;
    dec_context_t *ctx = (dec_context_t *)arg;

    DBG_PRINT("== DECODER thread started ==\n");

    while (!eos) {
        eos = decoder_proc_blocking(*ctx, eos, ctx->input_fd, ctx->pb);
    }

    while (ctx->dec->output_plane.getNumQueuedBuffers() > 0 && !ctx->got_error && !ctx->dec->isInError()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
        if (ret < 0) {
            DBG_PRINT("DEC Error DQing buffer at output plane\n");
            decoder_abort(ctx);
            break;
        }
    }

    ctx->got_eos = true;
    DBG_PRINT("== DECODER thread exit ==\n");
    pthread_exit(NULL);
}

/*
 *
 */
int start_decoder(dec_context_t& ctx, int idx) {
    pthread_attr_t attr;

    pthread_attr_init(&attr);

    if (pthread_create(&p_main_decoder[idx & 3], &attr, thread_main_decoder, (void *)&ctx) != 0) {
        return -1;
    }  
   
    return 0;
}

/*
 *
 */
int stop_decoder(dec_context_t& dctx, int idx) {
    int ret;
    //dctx.got_eos = true;
    dctx.bLoop = true;

    printf("Stop decoder...\n");

    pthread_join(dctx.dec_capture_loop, NULL);
    if (p_main_decoder[idx & 3] != 0) {
        pthread_join(p_main_decoder[idx & 3], NULL);
        p_main_decoder[idx & 3] = 0;
    }

    if(dctx.capture_plane_mem_type == V4L2_MEMORY_DMABUF) {
        for(int index = 0 ; index < dctx.numCapBuffers ; index++) {
            if(dctx.dmabuff_fd[index] != 0) {
                ret = NvBufferDestroy (dctx.dmabuff_fd[index]);
                if(ret < 0){
                    DBG_PRINT("Failed to Destroy NvBuffer\n");
                }
            }
        }
    }
    /* The decoder destructor does all the cleanup i.e set streamoff on output and
       capture planes, unmap buffers, tell decoder to deallocate buffer (reqbufs
       ioctl with count = 0), and finally call v4l2_close on the fd. */
    delete dctx.dec;
    
#ifdef USE_CUDA_EGL
    for(int index = 0 ; index < MAX_BUFFERS ; index++) {
        // Destroy EGLImage
        NvDestroyEGLImage(dctx.egl_display, dctx.egl_imagePtr[index]);
        dctx.egl_imagePtr[index] = NULL;
        cuGraphicsUnregisterResource(dctx.pResource[index]);
    }
    delete dctx.dec_output_queue;
    terminate_display(dctx.egl_display);
#endif      
    // destroy all dma buffers
    if (dctx.enc_dma_fd != -1) {
        NvBufferDestroy(dctx.enc_dma_fd);
    }

    for(int index = 0 ; index < MAX_BUFFERS ; index++) {
        if(dctx.dst_dma_fd[index] != -1) {
            NvBufferDestroy(dctx.dst_dma_fd[index]);
            dctx.dst_dma_fd[index] = -1;
        }
    }

    printf("Decoder stopped\n");
    return 0;
}

