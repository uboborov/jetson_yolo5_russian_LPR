#include "NvVideoDecoder.h"
#include "NvVideoConverter.h"
#include "NvEglRenderer.h"
#include <queue>
#include <fstream>
#include <pthread.h>
#include <semaphore.h>
#include "common.h"
#include "encoder.h"

#include "utils.h"
#include "nvosd.h"

#ifdef USE_CUDA_EGL
# include <map>
# include <cuda_runtime_api.h>
# include <cuda.h>
# include <cuda_runtime.h>
# include <cudaEGL.h>
using namespace std;
#endif

//#define USE_YOLOV5_SIZE_TRANSFORM // transform to YOLOv5 (640x640) on the decoder side 
                                    // (clip the rectangle of Yolov5 size from the source frame)
#define USE_NVBUF_TRANSFORM_API

#define MAX_BUFFERS 32

enum {
    DEC_INPUT_V4L2 = 0,
    DEC_INPUT_FILE,
    DEC_INPUT_RTSP,
    DEC_INPUT_VLOG
};

enum {
    DEC_MAIN_INDEX = 0,
    DEC_AUX_INDEX,

    DEC_NUMBER
};

typedef struct _dec_context {
    NvVideoDecoder *dec;
    NvVideoConverter *conv;
    uint32_t decoder_pixfmt;

    NvEglRenderer *renderer;

    char **in_file_path;
    std::ifstream **in_file;

    char *out_file_path;
    std::ofstream *out_file;

    bool disable_rendering;
    bool fullscreen;
    uint32_t window_height;
    uint32_t window_width;
    uint32_t window_x;
    uint32_t window_y;
    uint32_t out_pixfmt;
    uint32_t video_height;
    uint32_t video_width;
    uint32_t display_height;
    uint32_t display_width;
    uint32_t file_count;
    uint32_t w;
    uint32_t h;
    float fps;

    bool disable_dpb;

    bool input_nalu;

    bool copy_timestamp;
    bool flag_copyts;
    uint32_t start_ts;
    float dec_fps;
    uint64_t timestamp;
    uint64_t timestampincr;

    bool stats;

    int  stress_test;
    bool enable_metadata;
    bool bLoop;
    bool bQueue;
    bool enable_input_metadata;
    enum v4l2_skip_frames_type skip_frames;
    enum v4l2_memory output_plane_mem_type;
    enum v4l2_memory capture_plane_mem_type;
#ifndef USE_NVBUF_TRANSFORM_API
    enum v4l2_yuv_rescale_method rescale_method;
#endif

    std::queue < NvBuffer * > *conv_output_plane_buf_queue;
    pthread_mutex_t queue_lock;
    pthread_cond_t queue_cond;

    sem_t pollthread_sema; // Polling thread waits on this to be signalled to issue Poll
    sem_t decoderthread_sema; // Decoder thread waits on this to be signalled to continue q/dq loop
    pthread_t   dec_pollthread; // Polling thread, created if running in non-blocking mode.

    pthread_t dec_capture_loop; // Decoder capture thread, created if running in blocking mode.
    bool got_error;
    bool got_eos;
    bool got_det_stop;
    bool vp9_file_header_flag;
    bool vp8_file_header_flag;
    int dst_dma_fd[MAX_BUFFERS];
    int dmabuff_fd[MAX_BUFFERS];
    int numCapBuffers;
    int loop_count;
    int max_perf;
    int extra_cap_plane_buffer;
    int blocking_mode; // Set to true if running in blocking mode
    int (*decoded_cb)(struct _dec_context *ctx, void *out_data);
    pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t data_ready_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t rtsp_data_ready_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t vlog_data_ready_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t detector_ready_mutex = PTHREAD_MUTEX_INITIALIZER;
    int cur_odma_idx;
    int num_odma_buf;
    //v4l2
    int input_fd;
    struct buffer *pb;
    int measurement_en;
    int input_type;
    //rtsp
    void *rtsp_data;
    int rtsp_len;
    //vlog
    //rtsp
    void *vlog_data;
    int vlog_len;
    int current_frame;
    int seek_frames;
    int need_screenshot;
    // jpeg
    void *jpeg_ctx;
    // cuda
#ifdef USE_CUDA_EGL
    CUeglFrame eglFramePtr[MAX_BUFFERS];
    CUgraphicsResource pResource[MAX_BUFFERS];
    EGLImageKHR egl_imagePtr[MAX_BUFFERS];
    map<int, CUeglFrame> dma_egl_map;
    EGLDisplay egl_display;
    volatile int detector_busy;
    volatile int detector_init_done;
    char *detector_run_from_folder;
    char *detector_detect_to_folder;
    std::queue < int > *dec_output_queue;
#endif
    int cam_id;
    int use_aux_decoder;
    // OSD
    void *nvosd_context;
    int display_bbox;
    std::queue < int > *fd_queue;
    NvOSD_RectParams g_rect[1024];
    NvOSD_TextParams textParams[100];
    VideoEncoder *encoder;
    char* osd_text;
    int g_rect_num;
    int enc_dma_fd;
    char output_file_name[128];
} dec_context_t;

int setup_decoder(dec_context_t& dctx, uint32_t fmt, int w, int h, const char *in_file_path, int stage);
int start_decoder(dec_context_t& ctx, int idx);
int stop_decoder(dec_context_t& ctx, int idx);
