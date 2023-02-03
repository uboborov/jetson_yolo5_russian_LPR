#ifndef COMMON_H_
#define COMMON_H_

#define USE_DEC_DMA_ENC_CONNECTED
#define USE_INTERNAL_RTMP

#ifdef DETDEBUG
# define THIS_APP "DETECTOR"
# define DBG_PRINT(FMT, ...) printf(THIS_APP":" FMT , ## __VA_ARGS__)
#else
# define DBG_PRINT(...)
#endif

# ifndef MAX
#  define MAX(x, y) ( ((x)>(y))?(x):(y) )
# endif

# ifndef MIN
#  define MIN(x, y) ( ((x)<(y))?(x):(y) )
# endif

//#define USE_ENCODER

#define DEF_SCREENSHOT_PATH "/tmp"
#define DEF_DET_PATH        "/tmp/vlog/det"

enum {
    INPUT_NONE = 0,
    INPUT_V4L2,
    INPUT_FILE,
    INPUT_RTSP,
    INPUT_VLOG
};

enum {
    EXIT_STATUS_SUCCESS = 0,
    EXIT_STATUS_ENC_DEC_FAILED,
    EXIT_STATUS_RTSP_CLIENT_FAILED,
    EXIT_STATUS_STREAMER_FAILED,
    EXIT_STATUS_LOOPBACK_FAILED,
    EXIT_STATUS_V4L2_FAILED,
    EXIT_STATUS_FILE_FAILED
};

int write_encoding_data_cb(int lb_idx, void *pdata, unsigned int *len);
int file_exist(const char *filename);
void DBG_PRINT_RESP(unsigned char *resp, int resp_len);


#endif

