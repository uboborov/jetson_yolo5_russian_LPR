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
#include <sys/stat.h>
#include <poll.h>
#include <signal.h>
#include <nvbuf_utils.h>

#include "common.h"
#include "decoder.h"
#include "rtsp_client.h"
#ifdef USE_CUDA_EGL
#include "yolov5/yolov5.h"
#include "yolov5/yololayer.h"
#endif

#define DEF_VIDEO_DEV   "/dev/video1"
#define DEF_FPS         30
#define DEF_VIDEO_H     640
#define DEF_VIDEO_W     480
#define DEF_PIX_FMT     "H264"
#define PIPE_DET_RD_FIFO      "/tmp/detector_in.fifo"

#define DEF_CAMLOG_FILE_PATH "/tmp/vlog" 

//#define DECODER_ONLY

using namespace std;

extern int log_level;
static int measurement_en = 0;
pthread_t ext_thread;

static char VIDEO_DEV[20] = DEF_VIDEO_DEV;
static char VIDEO_LB[20] = "/dev/video8";

#define LB_DRV_NAME     "v4l2loopback"
#define LB_NAME_OFFSET  8
#define N_LB_DEV    1

static volatile unsigned int video_stop = 0;

void DBG_PRINT_RESP(uint8_t *resp, int resp_len) {
    int i;
    for (i = 0;i < resp_len;i++) {
        DBG_PRINT("%02X ", resp[i]);
    }
    DBG_PRINT("\n");
}

/*
 *
 */
static void signal_handler(int sig) {
    if (sig == SIGTERM || sig == SIGHUP) {
        DBG_PRINT("=== capture: got signal %d ===\n", sig);
        __sync_fetch_and_add(&video_stop, 1);
#ifndef USE_THREADING        
        //__sync_fetch_and_add(&video_running,1);
#endif        
    } else if (sig == SIGINT) {
        exit(0);
    }
}

/*
 *
 */
static int init_signal_handler() {
    struct sigaction act;

    //set signals handler
    act.sa_handler = signal_handler;
    (void) sigemptyset (&act.sa_mask);
    act.sa_flags = SA_RESTART;
    (void) sigaction (SIGTERM, &act, (struct sigaction *) NULL); // 15: Program to be terminated
    (void) sigaction (SIGHUP, &act, (struct sigaction *) NULL);
    (void) sigaction (SIGINT, &act, (struct sigaction *) NULL);
    
    return 0;
}

/*
 * decoder callback for encoder
 */
static int decoder_callback(struct _dec_context *ctx, void *out_data) {
    return 0;
}

/*
 * RTSP client received data ready callback
 */
static int rtsp_data_ready_cb(void *pctx, void *data, int len) {
    dec_context_t *ctx = (dec_context_t *)pctx;
    if (ctx != NULL && ctx->rtsp_data != NULL) {
        memcpy(ctx->rtsp_data, data, len);
        ctx->rtsp_len = len;
        pthread_mutex_unlock(&ctx->rtsp_data_ready_mutex);
    }
    return 0;
}

pthread_mutex_t encoder_data_ready_mutex = PTHREAD_MUTEX_INITIALIZER;
static int rtp_len = 0;
static void *rtp_data = NULL;

/*
 *
 */
static int timed_data_wait(pthread_mutex_t *mutex, int tm_sec) { 
    struct timespec abs_time = {0}; 
    clock_gettime(CLOCK_REALTIME, &abs_time);
    abs_time.tv_sec += tm_sec; 
    if (pthread_mutex_timedlock(mutex, &abs_time) == ETIMEDOUT) {
        return -1;
    }
    return 0;
}

/*
 * interneal rtp streamer wait input data callback
 */
static int rtp_streamer_wait_data_cb(void *user_data, void *data, int *len) {
    int ret = -1;
    rtp_data = data;
    ret = timed_data_wait(&encoder_data_ready_mutex, 10);
    *len = rtp_len;
    
    return ret;
}

static void set_powermode_model(int mode) {
    char buf[128] = {0};
    int res = 0;
    sprintf(buf, "/usr/sbin/nvpmodel -m %d -o l4t", mode == 1);
    system(buf);

    usleep(100000);
    sprintf(buf, "/usr/bin/jetson_clocks");
    system(buf);
}

/*
 *
 */
int main(int argc, char *argv[]) {
    dec_context_t dctx[DEC_NUMBER]; // MAIN (dec from stream) and AUX (dec from file) decoders
    struct stat sb;
    int error = 0;
    int ret = 0;
    int i, cnt;
    int eos = 0;
    char output_file[50] = "";
    char input_file[50] = "";
    char str_fmt[10] = "";
    int width, height;
    int opt;
    int video_fd, input_fd;
    int input_is_v4l2 = 0;
    int cap_dev_pix_fmt =  v4l2_fourcc(DEF_PIX_FMT[0], DEF_PIX_FMT[1], DEF_PIX_FMT[2], DEF_PIX_FMT[3]);
    int frame_rate = DEF_FPS;
    struct buffer *buffers;
    static int n_buffers;
    char mod_path[128];
    char mod_param[128];
    FILE *fflag = NULL;
    char rtsp_url[128] = "";
    int rtsp_stream = 0;
    int input_type = INPUT_NONE;
    int rtp_streamer = 0;
    char RTP_ADDR[512] = {0};
    int  RTP_PORT = 553;

    int need_screenshot = 0;
    int seek_frames = -1;
    int exit_status = EXIT_STATUS_SUCCESS;
    int skip_decoder = 0;
    int det_desabled = 0;

    char det_from_path[256];
    char det_to_path[256];
    int cam_id = -1;

    if (argc < 2) {
        printf("Usage: \n\t%s -v videodev -i input file -o output file -w width -h height -f format\n"
                       "\t-m measure enable, -r frame rate, -I input=VLOG file, -u RTSP client url\n"
                       "\t-F input folder, -T output folder, -d dry run", argv[0]);
        exit(0);
    }

    memset(det_from_path, 0, sizeof(det_from_path));
    memset(det_to_path, 0, sizeof(det_to_path));

    init_signal_handler();

    //log_level= LOG_LEVEL_DEBUG;

    while ((opt = getopt(argc, argv, "i:o:w:h:f:m:r:u:F:T:dc:")) != -1) {
        switch (opt) {
            case 'o':
                strcpy(output_file, optarg);
                break;
            case 'w':
                width = atoi(optarg);
                break;   
            case 'h':
                height = atoi(optarg);
                break; 
            case 'f':
                strcpy(str_fmt, optarg);
                cap_dev_pix_fmt = v4l2_fourcc(optarg[0], optarg[1], optarg[2], optarg[3]);
                break;             
            case 'm':
                measurement_en = atoi(optarg);
                break;
            case 'r':
                frame_rate = atoi(optarg);
                break;
            case 'i':
                strcpy(input_file, optarg);
                break;
	    case 'u':
                strcpy(rtsp_url, optarg);
                break; 
            case 'F':
                strcpy(det_from_path, optarg);
                skip_decoder = 1;
                break;
            case 'T':
                strcpy(det_to_path, optarg);
                break;
            case 'd':
                det_desabled = 1;
                break;
            case 'c':
                cam_id = atoi(optarg);
                break;                    

            default:
                printf("Usage: \n\t%s -v videodev -i input file -o output file -w width -h height -f format\n"
                       "\t-m measure enable, -r frame rate, -I input=VLOG file, -u RTSP client url\n"
                       "\t-F input folder, -T output folder, -d dry run", argv[0]);
                exit(0);
                break;    
        }
    }

    if (det_desabled) {
        while(!video_stop) {
            sleep(1);
        }
        exit(0);
    }

    set_powermode_model(0);

    /* in case of reading images from the dir */
    if (skip_decoder) goto dec_skip;

    if (input_type == INPUT_VLOG && strlen(input_file) == 0) {
        input_type = INPUT_NONE;
    }

    if (input_type == INPUT_NONE && strlen(rtsp_url) > 0 && strncmp(rtsp_url, "rtsp://", 7) == 0) {
        input_type = INPUT_RTSP;
    }

    if (strlen(input_file) > 0  && input_type == INPUT_NONE) {
        int flags = 0;
        if (strcmp(input_file, "-") != 0) {
            // some file
            if (!file_exist(input_file)) {
                DBG_PRINT("File %s doesn't exist\n", input_file);
                return EXIT_FAILURE;
            }
        } else {
            input_fd = 0; // stdin
        }
        input_type = INPUT_FILE;

        DBG_PRINT("Runnig transcoding from file %s...\n", (input_fd == 0)?"stdin":input_file);
    }

    
    if (input_type != INPUT_FILE) {
        need_screenshot = 0;
        seek_frames = -1;
    }
    
    if (input_type == INPUT_FILE) {
        if ((input_fd = open(input_file, O_RDONLY)) == -1) {
            exit_status = EXIT_STATUS_FILE_FAILED;
            goto cleanup;
        }
    }

    if (input_type == INPUT_NONE) {
        DBG_PRINT("Something went wrong! No input source.\n");
        while(video_stop == 0) {
            sleep(1);
        }
    }
    
    // setup decoder stage 0
    ret = setup_decoder(dctx[DEC_MAIN_INDEX], V4L2_PIX_FMT_H264, width, height, NULL, 0);
    if (ret) {
        DBG_PRINT("Decoder intit failed at stage 0!\n");
        exit_status = EXIT_STATUS_ENC_DEC_FAILED;
        goto cleanup;
    }
    // setup decoder stage 1
    ret = setup_decoder(dctx[DEC_MAIN_INDEX], V4L2_PIX_FMT_H264, width, height, NULL, 1);
    if (ret) {
        DBG_PRINT("Decoder intit failed at stage 1!\n");
        exit_status = EXIT_STATUS_ENC_DEC_FAILED;
        goto cleanup;
    }
    // extra settings
    dctx[DEC_MAIN_INDEX].measurement_en = measurement_en;
    dctx[DEC_MAIN_INDEX].num_odma_buf = 1;

    sprintf(dctx[DEC_MAIN_INDEX].output_file_name, "%s", output_file);

    if (input_type == INPUT_FILE) {
        dctx[DEC_MAIN_INDEX].input_type = DEC_INPUT_FILE;
        dctx[DEC_MAIN_INDEX].input_fd = input_fd;
        dctx[DEC_MAIN_INDEX].need_screenshot = 0;
        dctx[DEC_MAIN_INDEX].seek_frames = 0;
    } else if (input_type == INPUT_RTSP) {
        dctx[DEC_MAIN_INDEX].input_type = DEC_INPUT_RTSP;
        // start RTSP client
        start_rtsp_client(rtsp_url, rtsp_data_ready_cb, &dctx[DEC_MAIN_INDEX]);
    } else {
        DBG_PRINT("Invalid input source!\n");
        exit(-1);
    }

    // setup decoder stage 2
    ret = setup_decoder(dctx[DEC_MAIN_INDEX], V4L2_PIX_FMT_H264, width, height, NULL, 2);
    if (ret) {
        DBG_PRINT("Decoder intit failed at stage 2!\n");
        exit_status = EXIT_STATUS_ENC_DEC_FAILED;
        goto cleanup;
    }
    // setup decoder stage 3
    ret = setup_decoder(dctx[DEC_MAIN_INDEX], V4L2_PIX_FMT_H264, width, height, NULL, 3);
    if (ret) {
        DBG_PRINT("Decoder intit failed at stage 3!\n");
        exit_status = EXIT_STATUS_ENC_DEC_FAILED;
        goto cleanup;
    }

#ifndef DECODER_ONLY
    // init encoder

#endif

skip_enc:    

#ifndef DECODER_ONLY  
#if defined(USE_DEC_DMA_ENC_CONNECTED)
    dctx[DEC_MAIN_INDEX].num_odma_buf = MAX_BUFFERS - 1;
    // setup decoded data ready mutex for encoder
    pthread_mutex_lock(&dctx[DEC_MAIN_INDEX].data_ready_mutex);
    //ectx.pdata_ready_mutex = &dctx.data_ready_mutex;

    
#endif
    
#else 
    dctx[DEC_MAIN_INDEX].num_odma_buf = MAX_BUFFERS - 1;
#endif
    
    // start decoder
    //start_decoder(dctx[DEC_MAIN_INDEX], DEC_MAIN_INDEX);
dec_skip:
    dctx[DEC_MAIN_INDEX].detector_run_from_folder = det_from_path;
    dctx[DEC_MAIN_INDEX].detector_detect_to_folder = det_to_path;
    dctx[DEC_MAIN_INDEX].cam_id = 0;

#ifdef USE_CUDA_EGL
    // start detector
    start_detector(&dctx);
    while(!dctx[DEC_MAIN_INDEX].detector_init_done) {
        usleep(1000);
    }
    start_decoder(dctx[DEC_MAIN_INDEX], DEC_MAIN_INDEX);
#endif      

    while(video_stop == 0) {
        if (dctx[DEC_MAIN_INDEX].got_eos) {
            DBG_PRINT("=== Got EOS from %s ===\n", dctx[DEC_MAIN_INDEX].got_eos?"DECODER":"Unknown");
            break;
        }
        sleep(1);
    }

    if (video_stop) {
        DBG_PRINT("=== Video stopped by signal ===\n");
    }

 #ifdef USE_CUDA_EGL
    stop_detector(&dctx);
 #endif   
    // stop decoder
    if (!skip_decoder) {
        stop_decoder(dctx[DEC_MAIN_INDEX], DEC_MAIN_INDEX);
    }
    
    // stop video source
    if (input_type == INPUT_FILE) {
        close(input_fd);
    } else if (input_type == INPUT_RTSP) {
        stop_rtsp_client();
    }

    DBG_PRINT("\nEXIT\n");
    exit(0);

cleanup:
    exit(exit_status);
}
