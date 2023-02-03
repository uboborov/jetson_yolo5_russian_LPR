#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h> 
#include <signal.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"

#include "NvUtils.h"
#include "NvCudaProc.h"
#include <nvbuf_utils.h>

#include "../decoder.h"

//#define USE_PLATE_ANGLE_DETECT    // detect plate angle and rotate

#define YOLO_NUM_CLASSES 1

#define DEF_SCREENSHOT_PATH "/tmp/vlog/img"
#define GIGABYTE 1073741824UL
#define BEST_FRAME_THRESHOLD  10

#define TEMPERATURE_HIGH_LIMIT   80.0
#define TEMPERATURE_LOW_LIMIT    40.0

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

#define MAX_WORKSPACE (1 << 20) 
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

using namespace nvinfer1;

//***************************************************************************
class chDet {
public: 
    chDet(std::string enginePath, int batchSize, ILogger *gLogger, int mode);
    ~chDet();
    enum {
        MODE_DET = 0,
        MODE_TEXT
    };

public:
    int detect(cv::Mat& frame, std::string &text, std::vector<Yolo::Detection>& res);
    int m_init_done;
    int m_mode;

private:
    std::string m_enginePath;
    int m_batchSize;
    ILogger *m_gLogger;

private:
    IRuntime *m_runtime;
    ICudaEngine *m_engine;
    IExecutionContext *m_context;
    cudaStream_t m_stream;
    float *m_buffers[2];
    float *m_out;
    int m_inputIndex;
    int m_outputIndex;
    uint8_t *m_img_host;
    uint8_t *m_img_device;
    
    int init();
};

/*
 *
 */
chDet::chDet(std::string enginePath, int batchSize, ILogger *gLogger, int mode) {
    m_enginePath = enginePath;
    m_gLogger = gLogger;
    m_batchSize = batchSize;
    m_mode = mode;
    m_init_done = init();
}

/*
 *
 */
chDet::~chDet() {
    // Release stream and buffers
    cudaStreamDestroy(m_stream);
    CUDA_CHECK(cudaFree(m_img_device));
    CUDA_CHECK(cudaFreeHost(m_img_host));

    CUDA_CHECK(cudaFree(m_buffers[m_inputIndex]));
    CUDA_CHECK(cudaFree(m_buffers[m_outputIndex]));
 
    // Destroy the engine
    m_context->destroy();
    m_engine->destroy();
    m_runtime->destroy();
    delete m_out;
}

/*
 *
 */
int chDet::init() {
    std::ifstream file(m_enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << m_enginePath << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    m_runtime = createInferRuntime(*m_gLogger);
    assert(m_runtime != nullptr);
    m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
    assert(m_engine != nullptr);
    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);
    delete[] trtModelStream;
    //printf("N bindings %d, output size: %d\n", engine->getNbBindings(), OUTPUT_SIZE);

    assert(m_engine->getNbBindings() == 2);
    int outSize = OUTPUT_SIZE;
    m_out = new float[outSize * BATCH_SIZE];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    m_inputIndex = m_engine->getBindingIndex(INPUT_BLOB_NAME);
    m_outputIndex = m_engine->getBindingIndex(OUTPUT_BLOB_NAME);
    
    assert(m_inputIndex == 0);
    assert(m_outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&m_buffers[m_inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&m_buffers[m_outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    m_img_host = nullptr;
    m_img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&m_img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&m_img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    return 0;
}

/*
 *
 */
int chDet::detect(cv::Mat& frame, std::string &text, std::vector<Yolo::Detection>& res) {
    std::vector<std::pair<int,char>> items;

    if (frame.empty()) return -1;
    size_t  size_image = frame.cols * frame.rows * 3;
    size_t  size_image_dst = INPUT_H * INPUT_W * 3;
    float* buffer_idx = (float*)m_buffers[m_inputIndex];
    //copy data to pinned memory
    memcpy(m_img_host, frame.data, size_image);

    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(m_img_device, m_img_host, size_image, cudaMemcpyHostToDevice, m_stream));
    preprocess_kernel_img(m_img_device, frame.cols, frame.rows, buffer_idx, INPUT_W, INPUT_H, m_stream);
    
    m_context->enqueue(m_batchSize, (void**)m_buffers, m_stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(m_out, m_buffers[m_outputIndex], m_batchSize * OUTPUT_SIZE * sizeof(float), 
               cudaMemcpyDeviceToHost, m_stream));

    cudaStreamSynchronize(m_stream);

    //std::vector<std::vector<Yolo::Detection>> batch_res(1);
    //auto& res = batch_res[0];
    nms(res, &m_out[0], CONF_THRESH, NMS_THRESH);

    if (m_mode = MODE_TEXT) {
        char symbols[] = "0123456789ABCEHKMOPTXY ";
        //printf("Text detections %d\n", res.size());

        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(frame, res[j].bbox);
            items.push_back(std::make_pair(r.x, symbols[(int)res[j].class_id]));
        }
        // sort symbols by X coordinate
        std::sort(items.begin(), items.end());
        text.clear();

        if (items.size() > 0) {
            for (int i = 0;i < items.size();i++) {
                text += items[i].second;
            }
        } else {
            return -1;
        }
        items.clear();
    }

    return 0;
}
//***********************************************************
#ifdef USE_PLATE_ANGLE_DETECT
double getAngle(cv::Mat& plate);
unsigned getBottomBound(cv::Mat& plate);
unsigned getHistTopBound(cv::Mat& plate);
unsigned getTopBound(cv::Mat& plate);
unsigned getLeftBound(cv::Mat plate, bool iswhite);
unsigned getRightBound(cv::Mat plate, bool iswhite);
void rotateImage(cv::Mat& image, const double angle);

const unsigned thresh         = 160;
const unsigned scale          = 2;
const double minDegree = -10;
const double maxDegree = 10;
const double stepDegree= 0.1;

/*
 *
 */
double getAngle(cv::Mat& plate) {
    unsigned min = plate.size().height;
    double angle = 0;
    cv::Mat temp;

    for(double a = minDegree; a < maxDegree; a += stepDegree) {
        temp = plate.clone();
        rotateImage(temp, a);

        unsigned bottomBound = getBottomBound(temp);
        if(bottomBound < min) {
            angle = a;
            min = bottomBound;
        }
    }

    return angle; 
}

/*
 *
 */
void rotateImage(cv::Mat& image, const double angle) {
    cv::Mat rot_mat(2, 3, CV_32FC1);

    cv::Point center = cv::Point(image.cols/2, image.rows/2);
    double scale = 1;

    rot_mat = getRotationMatrix2D(center, angle, scale);

    warpAffine(image, image, rot_mat, image.size());    
}

/*
 *
 */
unsigned getBottomBound(cv::Mat& plate) {
    //equalizeHist(plate, plate);

    size_t height = plate.size().height;
    unsigned lastCount = 0;
    cv::Mat data;
    
    for(unsigned i = height/2; i < height; ++i) {
        data = plate.row(i);
        unsigned count = cv::countNonZero(data);
        
        if(count < lastCount/2)
            return i;

        lastCount = count;
    }
    
    return height;
}

/*
 *
 */
unsigned getHistTopBound(cv::Mat& plate) {
    size_t height = plate.size().height;
    cv::Mat data;
    
    for(unsigned i = 0; i < height/2; ++i) {
        data = plate.row(i);
        unsigned count = cv::countNonZero(data);
        
        if(count > height*0.5)
            return i;
    }
    
    return 0;
}

/*
 *
 */
unsigned getTopBound(cv::Mat& plate) {
    return getHistTopBound(plate);
}

/*
 *
 */
unsigned getLeftBound(cv::Mat plate, bool iswhite) {
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                                       cv::Size( 2*1 + 1, 2*1+1 ),
                                       cv::Point( 1, 1) );
    cv::erode(plate, plate, element);
    cv::dilate(plate, plate, element);

    size_t width = plate.size().width;
    double height= plate.size().height;

    cv::Mat data;

    for(unsigned i = 2; i < width/2; ++i) {
        data = plate.col(i);
        unsigned count = cv::countNonZero(data);
        
        if((!iswhite && count > height*0.5) || (iswhite && count < height*0.60))
            return i;
    }
    
    return 0;
}

/*
 *
 */
unsigned getRightBound(cv::Mat plate, bool iswhite) {
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                                       cv::Size( 2*1 + 1, 2*1+1 ),
                                       cv::Point( 1, 1) );
    cv::erode(plate, plate, element);
    cv::dilate(plate, plate, element);

    size_t width = plate.size().width;
    double height= plate.size().height;

    cv::Mat data;
    
    for(unsigned i = width-2; i > width/2; --i) {
        data = plate.col(i);
        unsigned count = cv::countNonZero(data);
        
        if((!iswhite && count > height*0.5) || (iswhite && count < height*0.60))
            return i+1;
    }
    
    return width;
}
#endif
//***************************************************

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
    if (argc < 3) return false;

    printf("argc: %d\n", argc);

    if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}


/*
 *
 */
static bool getFileContent(std::string fileName, std::vector<std::string> & Names) {

    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in.is_open()) {
        return false;
    }

    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str)) {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            Names.push_back(str);
    }
    //Close The File
    in.close();
    return true;
}

/*
 *
 */
static void set_powermode_model() {
    char buf[128] = {0};
    int res = 0;
    system("/usr/sbin/nvpmodel -m 0 -o l4t");
    system("/usr/bin/jetson_clocks");
}

/*
 *
 */
static int check_temperature() {
    float gputemp = 0;
    float cputemp = 0;
    char cpu[20], gpu[20];
    FILE* fcputemp = fopen("/sys/devices/virtual/thermal/thermal_zone1/temp", "r");
    FILE* fgputemp = fopen("/sys/devices/virtual/thermal/thermal_zone2/temp", "r");
    if (!fcputemp || !fgputemp ) {
        DBG_PRINT("Something went wrong with reading temperature\n");
        return -1;
    }

    cputemp = atoi(fgets(cpu, 6, fcputemp))/1000;
    gputemp = atoi(fgets(gpu, 6, fgputemp))/1000;
    
    printf("CPU : %.2f, GPU : %.2f\n", cputemp, gputemp);
    fclose(fcputemp);
    fclose(fgputemp);

    if (cputemp >= TEMPERATURE_HIGH_LIMIT || gputemp >= TEMPERATURE_HIGH_LIMIT) {
        return -1;
    } else if (cputemp >= TEMPERATURE_LOW_LIMIT || gputemp >= TEMPERATURE_LOW_LIMIT) {
        return 1;
    }

    return 0;
}

void encBufferDoneCallback(int dmabuf_fd, void *arg) {
    //DBG_PRINT("Encoder buffer DONE\n");
}


/*
 *
 */
static int detector_lock(pthread_mutex_t *mutex) { 
    pthread_mutex_lock(mutex);
    return 0;
}

/*
 *
 */
static int is_directory(char *path) {
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISDIR(path_stat.st_mode);
}

/*
 *
 */
static int get_image_from_fd(int dma_fd, int w, int h, cv::Mat& out) {
    void *pdata = NULL;
    int status = -1;
    cv::Mat imgbuf;
    NvBufferParams params;
    status = NvBufferGetParams(dma_fd, &params);
    if (status) goto serror;
    status |= NvBufferMemMap(dma_fd, 0, NvBufferMem_Read_Write, &pdata);
    if (status) goto serror;
    status |= NvBufferMemSyncForCpu(dma_fd, 0, &pdata);
    if (status) goto sdone;
    // RGBA -> CV_8UC4
    imgbuf = cv::Mat(h, w, CV_8UC4, pdata, params.pitch[0]);
    cv::cvtColor(imgbuf, out, cv::COLOR_RGBA2BGR);
sdone:
    NvBufferMemUnMap(dma_fd, 0, &pdata);

serror:
    return status;
}

//***************************************************
/*
 *
 */
static void *thread_detector(void *arg) {
    dec_context_t *pctx = (dec_context_t *)arg;
    dec_context_t *ctx_main = (pctx + DEC_MAIN_INDEX);
    dec_context_t *ctx_aux =  (pctx + DEC_AUX_INDEX);
    dec_context_t *ctx = ctx_main;

    cv::Mat frame;
    bool eos = false;
    int use_tracker = 1;
    int ret;
    unsigned long n_frames = 0;
    //**************************************

    std::vector<std::string> file_names;

    __sync_lock_test_and_set(&ctx->detector_busy, 1);
    //**************************************
    set_powermode_model();
    cudaSetDevice(DEVICE);

    //**** Yolo classes number *************
    Yolo::CLASS_NUM = YOLO_NUM_CLASSES;
    //**************************************

    // in case of using cropped image on decoder side
#ifdef USE_YOLOV5_SIZE_TRANSFORM 
    void* offset_gpu;
    void* scales_gpu;
    int        offsets[3] = {0, 0, 0};
    float      input_scale[3] = {0.0039215697906911373, 0.0039215697906911373, 0.0039215697906911373};

    CUDA_CHECK(cudaMalloc(&offset_gpu, sizeof(int) * 3));
    CUDA_CHECK(cudaMalloc(&scales_gpu, sizeof(float) * 3));
    CUDA_CHECK(cudaMemcpy(offset_gpu, (void*)offsets, sizeof(int) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scales_gpu, (void*)input_scale, sizeof(float) * 3, cudaMemcpyHostToDevice));
#endif
    __sync_lock_release(&ctx->detector_busy);

    auto startt = std::chrono::system_clock::now();

    // plate detector
    chDet *pldet = new chDet("./yolov5s_lpr_plate.engine", 1, &gLogger, chDet::MODE_DET);
    if (!pldet || pldet->m_init_done) {
        printf("Plate detector init FAILED!\n");
        exit(0);
    } else {
        printf("Plate detector init DONE!\n");
    }

    // text detector
    chDet *chdet = new chDet("./yolov5m_lpr_pre_v2.engine", 1, &gLogger, chDet::MODE_TEXT);
    if (!chdet || chdet->m_init_done) {
        printf("Text detector init FAILED!\n");
        exit(0);
    } else {
        printf("Text detector init DONE!\n");
    }

    ctx->detector_init_done = 1;

#if 1
    //*************************** FROM FOLDER **********************
    file_names.clear();
    use_tracker = (strlen(ctx->detector_run_from_folder) == 0);
    if (use_tracker == 0) {
        DBG_PRINT("Running from directory...\n");
        if (read_files_in_dir(ctx->detector_run_from_folder, file_names) < 0) {
            DBG_PRINT("read_files_in_dir failed\n");
            ctx->got_error = true;
            goto on_exit;
        }

        int fcount = 0;
        for (auto f: file_names) {
            cv::Mat img = cv::imread(std::string(ctx->detector_run_from_folder) + "/" + f);
            if (img.empty()) continue;
            
            std::vector<Yolo::Detection> res;
            std::string text;
            text.clear();
            res.clear();
            
            pldet->detect(img, text, res);           
            
            DBG_PRINT("DETECTED PLATES: %d\n", res.size());

            for (size_t j = 0; j < res.size(); j++) {
                char t[256];
                cv::Rect r = get_rect(img, res[j].bbox);
            
                if (!(0 <= r.x && 0 <= r.width && r.x + r.width <= img.cols && 
                0 <= r.y && 0 <= r.height && r.y + r.height <= img.rows)) continue;
                
                cv::Mat img_roi = img(r).clone();
                if (img_roi.total() == 0) continue;
                text.clear();
                std::vector<Yolo::Detection> rs;
                rs.clear();
#ifdef USE_PLATE_ANGLE_DETECT
                std::vector<cv::Vec4i> hierarchy;
                cv::Mat cannyOutput, srcGray, srcThreshold;

                cvtColor(img_roi, srcGray, cv::COLOR_BGR2GRAY);
                threshold(srcGray, srcThreshold, 0, 255, cv::THRESH_BINARY  | cv::THRESH_OTSU);
                medianBlur(srcThreshold, srcThreshold, 5);

                double angle = getAngle(srcThreshold);
                printf("Angle: %.2f\n", angle);
                rotateImage(img_roi, angle);
#endif                 
                chdet->detect(img_roi, text, rs);

                img_roi.release();

                if (text.size() > 0) {
                    printf("%s\n", text.c_str());
                }
                sprintf(t, "%s", text.c_str());
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, t, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 0xFF), 2);
            }
            if (is_directory(ctx->detector_detect_to_folder)) {
                try {
                    cv::imwrite(std::string(ctx->detector_detect_to_folder) + "/" + f, img);
                }
                catch(cv::Exception& e) {
                    const char* err_msg = e.what();
                    std::cout << "exception caught: " << err_msg << std::endl;
                }
                
            }
            img.release();
        }
        ctx->got_eos = true;
        goto on_exit;
    }
    //***************************************************************
#endif
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

    //transform_params.src_rect = src_rect;
    //transform_params.dst_rect = dest_rect;
    
    /* Indicates which of the transform parameters are valid. */
    transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER | NVBUFFER_TRANSFORM_FLIP;
    transform_params.transform_flip = NvBufferTransform_None;
    transform_params.transform_filter = NvBufferTransform_Filter_Nearest;
#endif                

    if (strlen(ctx->output_file_name) > 0) {
        ctx->nvosd_context = nvosd_create_context();
        ctx->encoder = new VideoEncoder("encoder", ctx->output_file_name, ctx->w, ctx->h, V4L2_PIX_FMT_H264);
        //
        if (ctx->encoder != NULL) {
            ctx->encoder->initialize();
            ctx->encoder->setBufferDoneCallback(encBufferDoneCallback, NULL);
            ctx->encoder->set_bitrate(3 * 1024 * 1024);
        } else {
            DBG_PRINT("ENCODER setup failed\n");
            ctx->got_eos = true;
            goto on_exit;
        }
    }
    
    DBG_PRINT("Running from decoder...\n");

    //pthread_mutex_unlock(&ctx->detector_ready_mutex);

    while(!eos) {
        int dma_buf_fd = -1;
        int ntracks = 0;

        detector_lock(&ctx->data_ready_mutex);
        __sync_lock_test_and_set(&ctx->detector_busy, 1);

        if (ctx->dec_output_queue->empty()) {
            __sync_lock_release(&ctx->detector_busy);
            pthread_mutex_unlock(&ctx->detector_ready_mutex);
            continue;
        }
        dma_buf_fd = ctx->dec_output_queue->front();
        ctx->dec_output_queue->pop();
        
        auto start = std::chrono::system_clock::now();

        auto search = ctx->dma_egl_map.find(dma_buf_fd);
        if (search == ctx->dma_egl_map.end()) {
            __sync_lock_release(&ctx->detector_busy);
            pthread_mutex_unlock(&ctx->detector_ready_mutex);
            continue;
        }

        /* EGL frame from the decoder */
        CUeglFrame eglFrame = search->second;
        //printf("w: %d, h: %d, chan: %d\n", eglFrame.width, eglFrame.height, eglFrame.numChannels);
#ifdef USE_YOLOV5_SIZE_TRANSFORM        
        convertEglFrameIntToFloat(&eglFrame,
                    INPUT_W,
                    INPUT_H,
                    COLOR_FORMAT_RGB,
                    buffer_idx,
                    offset_gpu,
                    scales_gpu,
                    &conv_stream);
        cudaStreamSynchronize(conv_stream);
#else
        /* convert DMA fd to cv::Mat */
        get_image_from_fd(dma_buf_fd, eglFrame.width, eglFrame.height, frame);
        if (frame.empty()) {
            __sync_lock_release(&ctx->detector_busy);
            pthread_mutex_unlock(&ctx->detector_ready_mutex);
            continue;
        }
           
#endif        
        std::vector<Yolo::Detection> res;
        std::string text;
        text.clear();
        res.clear();
        
        pldet->detect(frame, text, res);

        auto end = std::chrono::system_clock::now();
        //std::cout << "Total inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
        //DBG_PRINT("DETECTIONS: %d\n", res.size());
        int nplate = 0;
        int ntext = 0;

        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(frame, res[j].bbox);
        
            if (!(0 <= r.x && 0 <= r.width && r.x + r.width <= frame.cols && 
            0 <= r.y && 0 <= r.height && r.y + r.height <= frame.rows)) continue;
            
            cv::Mat img_roi = frame(r).clone();
            if (img_roi.total() == 0) continue;
            text.clear();
            std::vector<Yolo::Detection> rs;
            rs.clear();

            if (ctx->encoder != NULL) {
                ctx->g_rect[nplate].left = r.x;
                ctx->g_rect[nplate].top = r.y;
                ctx->g_rect[nplate].width = r.width;;
                ctx->g_rect[nplate].height = r.height;
                ctx->g_rect[nplate].border_width = 3;
                ctx->g_rect[nplate].border_color.red = 0.0f;
                ctx->g_rect[nplate].border_color.green = 1.0;
                ctx->g_rect[nplate].border_color.blue = 0.0;
                nplate++;
            }
            

            chdet->detect(img_roi, text, rs);

            img_roi.release();

            if (text.size() > 0) {
                printf("%s\n", text.c_str());
                if (ctx->encoder != NULL) {
                    ctx->textParams[ntext].display_text = strdup(text.c_str());
                    ctx->textParams[ntext].x_offset = r.x;
                    ctx->textParams[ntext].y_offset = r.y - 35;
                    ctx->textParams[ntext].font_params.font_name = "Arial";
                    ctx->textParams[ntext].font_params.font_size = 25;
                    ctx->textParams[ntext].font_params.font_color.red = 0.0;
                    ctx->textParams[ntext].font_params.font_color.green = 0.0;
                    ctx->textParams[ntext].font_params.font_color.blue = 1.0;
                    ctx->textParams[ntext].font_params.font_color.alpha = 1.0;
                    ntext++;
                }
            }
        }

        if (ntext > 0) {
            if (ctx->nvosd_context != NULL) {
                nvosd_put_text(ctx->nvosd_context,
                                  MODE_CPU,
                                  dma_buf_fd,
                                  ntext,
                                  ctx->textParams);
            }

            for (int k = 0;k < ntext;k++) {
                free(ctx->textParams[k].display_text);
            }
        }

        // encode to H264
        if (ctx->enc_dma_fd != -1 && ctx->encoder != NULL) {
            ret = NvBufferTransform(dma_buf_fd, ctx->enc_dma_fd, &transform_params);
            
            if (ret != -1) {
                if (ctx->nvosd_context != NULL) {
                    nvosd_draw_rectangles(ctx->nvosd_context,
                                              MODE_HW,
                                              ctx->enc_dma_fd,
                                              nplate,
                                              ctx->g_rect);
                }
                ctx->encoder->encodeFromFd(ctx->enc_dma_fd);
            }
        }
        

        pthread_mutex_unlock(&ctx->detector_ready_mutex);

check_temp:
        frame.release();
 
        auto endtt = std::chrono::system_clock::now();
        std::chrono::milliseconds timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>
                                                   ((endtt - startt));
        /* temperature checking */
        if (timeTaken.count() >= 10000) {
            ret = check_temperature();
            if (ret < 0) {
                printf("Chip temperature limit exceeded\n");
                do {
                    if (ctx->got_eos || ctx->got_det_stop) goto on_exit;
                    sleep(1);
                } while(check_temperature() != 0);
            }
            startt = std::chrono::system_clock::now();
        }

        __sync_lock_release(&ctx->detector_busy);
        n_frames++;
        if ((n_frames % 10) == 0) {
            DBG_PRINT("FRAME %d\n", n_frames);
        }
        eos = (ctx->got_eos || ctx->got_det_stop);
    }
on_exit:    
    delete pldet;
    delete chdet;

    if (ctx->encoder != NULL) {
        ctx->encoder->shutdown();
    }

    if (ctx->nvosd_context != NULL) {
        nvosd_destroy_context(ctx->nvosd_context);
        ctx->nvosd_context = NULL;
    }

    DBG_PRINT("Exiting detector\n");

}

pthread_t p_main_detector = -1;

/*
 *
 */
int start_detector(void *pctx) {
    pthread_attr_t attr;
    dec_context_t *ctx = (dec_context_t *)pctx;
    (ctx + DEC_MAIN_INDEX)->got_det_stop = 0;

    pthread_attr_init(&attr);

    if (pthread_create(&p_main_detector, &attr, thread_detector, (void *)ctx) != 0) {
        return -1;
    }
   
    return 0;
}

/*
 *
 */
int stop_detector(void *pctx) {
    dec_context_t *ctx = (dec_context_t *)pctx;
    DBG_PRINT("Stop detector...\n");
    (ctx + DEC_MAIN_INDEX)->got_det_stop = 1;
    if (p_main_detector != -1) {
        pthread_join(p_main_detector, NULL);
    }
    DBG_PRINT("Detector stopped\n");
}

//***************************************************
#if 0
#define YOLO_NUM_CLASSES      1

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    //**** Yolo classes number *************
    Yolo::CLASS_NUM = YOLO_NUM_CLASSES;
    //**************************************

    std::vector<std::pair<int,char>> items;

    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

#if 1
    std::vector<std::string> Names;
    if(!getFileContent("./labels.txt", Names)) {
        fprintf(stderr, "loading names failed\n");
        return -1;
    }

    chDet *pldet = new chDet("./yolov5l_plate_detect_1class.engine", 1, &gLogger, chDet::MODE_DET);
    if (!pldet || pldet->m_init_done) {
        printf("Plate detector init FAILED!\n");
        exit(0);
    } else {
        printf("Plate detector init DONE!\n");
    }

    chDet *chdet = new chDet("./yolov5l_char_detect_37classes.engine", 1, &gLogger, chDet::MODE_TEXT);
    if (!chdet || chdet->m_init_done) {
        printf("Text detector init FAILED!\n");
        exit(0);
    } else {
        printf("Text detector init DONE!\n");
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }


    int fcount = 0;
    //std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < (int)file_names.size(); f++) {
        cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + 0]);
        if (img.empty()) continue;
        std::vector<Yolo::Detection> res;
        std::string text;
        text.clear();
        res.clear();
        pldet->detect(img, text, res);
        
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
        
            if (!(0 <= r.x && 0 <= r.width && r.x + r.width <= img.cols && 
            0 <= r.y && 0 <= r.height && r.y + r.height <= img.rows)) continue;
            
            cv::Mat img_roi = img(r).clone();
            if (img_roi.total() == 0) continue;
            text.clear();
            std::vector<Yolo::Detection> rs;
            rs.clear();
            chdet->detect(img_roi, text, rs);
#if 0
            for (int k = 0;k < rs.size();k++) {
                cv::Rect rc = get_rect(img_roi, rs[k].bbox);
                cv::Mat roi = img_roi(rc).clone();
                char buf[200];
                sprintf(buf, "_%d.png", (int)rs[k].class_id);
                cv::imwrite(buf, roi);
            }
#endif
            img_roi.release();

            if (text.size() > 0) {
                printf("%s\n", text.c_str());
            }
        }
    }

    delete pldet;
    delete chdet;
    

#else
    chDet *chdet = new chDet("./yolov5l_char_detect_37classes.engine", 1, &gLogger);
    if (!chdet || chdet->m_init_done) {
        printf("Text detector init FAILED!\n");
        exit(0);
    } else {
        printf("Text detector init DONE!\n");
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    int fcount = 0;
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < (int)file_names.size(); f++) {
        cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + 0]);
        if (img.empty()) continue;
        std::string text;
        text.clear();
        chdet->detect(img, text);
        if (text.size() > 0) {
            printf("%s\n", text.c_str());
        }
    }

#endif    

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
#endif