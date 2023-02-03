
include Rules.mk

DEL = /bin/rm -f

USE_CUDA_EGL=yes
USE_OBJ_TRACKING_SORT=no
USE_OBJ_TRACKING_DEEPSORT=no

APP := lpr
MKFILE=Makefile
ROOT_DST := /home/ubobrov/develop/projects/intercom/rootfs/root/jetson

LIVE_ROOT=/home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/live
LIVE_MEDIA_PATH=$(LIVE_ROOT)/liveMedia
GROUPSOCK_PATH=$(LIVE_ROOT)/groupsock
USAGE_ENV_PATH=$(LIVE_ROOT)/UsageEnvironment
BASIC_USAGE_ENV_PATH=$(LIVE_ROOT)/BasicUsageEnvironment

LDFLAGS += $(LIVE_MEDIA_PATH)/libliveMedia.a \
			$(GROUPSOCK_PATH)/libgroupsock.a \
			$(BASIC_USAGE_ENV_PATH)/libBasicUsageEnvironment.a \
			$(USAGE_ENV_PATH)/libUsageEnvironment.a

LIVE555_DEFS =  -DSOCKLEN_T=socklen_t -DNO_SSTREAM=1 -D_LARGEFILE_SOURCE=1 \
       			-D_FILE_OFFSET_BITS=64 -DALLOW_RTSP_SERVER_PORT_REUSE=1 -DNO_OPENSSL=1			

COMMON_DEFS = -DDETDEBUG

CPPFLAGS += -I$(LIVE_MEDIA_PATH)/include \
			-I$(GROUPSOCK_PATH)/include \
			-I$(USAGE_ENV_PATH)/include \
			-I$(BASIC_USAGE_ENV_PATH)/include \
			$(LIVE555_DEFS)	$(COMMON_DEFS) 

CPFLAGS += $(COMMON_DEFS)

CLASS_SRC := $(CLASS_DIR)/NvBuffer.cpp \
	$(CLASS_DIR)/NvElement.cpp \
	$(CLASS_DIR)/NvElementProfiler.cpp \
	$(CLASS_DIR)/NvLogging.cpp \
	$(CLASS_DIR)/NvV4l2Element.cpp \
	$(CLASS_DIR)/NvV4l2ElementPlane.cpp \
	$(CLASS_DIR)/NvVideoDecoder.cpp \
	$(CLASS_DIR)/NvVideoEncoder.cpp \
	$(CLASS_DIR)/NvJpegEncoder.cpp 

SRCS := \
	decoder.cpp \
	encoder.cpp \
	utils.cpp \
	main.cpp \
	rtsp_client.cpp \
	$(CLASS_SRC)

CSRC := 

ifeq ($(USE_CUDA_EGL),yes)
	ALGORITHM_DIR := $(TOP_DIR)/common/algorithm/cuda
	SRCS += $(ALGORITHM_DIR)/NvCudaProc.cpp
	CPPFLAGS += -I$(ALGORITHM_DIR)
	CUDA_PATH = /usr/local/cuda-10.2
	CV_PATH = /home/ubobrov/develop/projects/intercom/rootfs/root/jetson/nv_lib/nv_lib_cudnn_cv_tensorrt/usr/local/lib
	LIB_PATH1 = /home/ubobrov/develop/projects/intercom/rootfs/root/jetson/nv_lib/nv_lib_cudnn_cv_tensorrt/usr/lib/aarch64-linux-gnu
	LIB_PATH2 = /home/ubobrov/develop/projects/intercom/rootfs/root/jetson/nv_lib/nv_lib_cudnn_cv_tensorrt/usr/local/lib/aarch64-linux-gnu
	LIB_PATH3 = /home/ubobrov/develop/projects/intercom/rootfs/root/jetson/nv_lib/nv_lib_cudnn_cv_tensorrt/usr/local/cuda-10.2/lib
	NVC = $(CUDA_PATH)/bin/nvcc -m64 -ccbin $(filter-out $(AT), $(CPP)) -gencode=arch=compute_53,code=sm_53 -O3
	CUSRC := yolov5/preprocess.cu yolov5/yololayer.cu $(ALGORITHM_DIR)/NvAnalysis.cu
	YOLOSRC := ./yolov5/calibrator.cpp ./yolov5/yolov5.cpp
	CUFLAGS := -I/home/ubobrov/develop/projects/nvidia/jetson_nano/CUDA/include \
		    -I. -I./yolov5 \
		    -I/home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/opencv_cuda/cuda-aarch64-cc-mxnet-opencv/opencv-4.5.3-cudnn/usr/local/include/opencv4
	CPPFLAGS += $(CUFLAGS)
	CPPFLAGS += -DUSE_CUDA_EGL
	LDFLAGS += -lEGL -lcudart -lcuda
	LDFLAGS += -L/home/ubobrov/develop/projects/intercom/rootfs/root/jetson/nv_lib/nv_lib_cudnn_cv_tensorrt/usr/lib/aarch64-linux-gnu
	LDFLAGS += -L$(CV_PATH)
	LDFLAGS += -L$(LIB_PATH1)
	LDFLAGS += -L$(LIB_PATH2)
	LDFLAGS += -L$(LIB_PATH3)
	LDFLAGS += -lnvinfer
	LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn  \
      		   -lopencv_objdetect -lopencv_features2d \
      		   -lopencv_flann -lopencv_calib3d \
		   -lopencv_cudawarping \
		   -lopencv_cudaimgproc \
		   -lopencv_cudaarithm \
      		   -ldl -lrt
      	LDFLAGS += -Wl,-rpath-link=$(CV_PATH)
      	LDFLAGS += -Wl,-rpath-link=$(LIB_PATH1)
      	LDFLAGS += -Wl,-rpath-link=$(LIB_PATH2)
      	LDFLAGS += -Wl,-rpath-link=$(LIB_PATH3)

	vpath %.cu $(sort $(dir $(CUSRC:.cu=.o)))
	vpath %.cpp $(sort $(dir $(YOLOSRC:.cpp=.o)))
endif
#OBJS := $(SRCS:.cpp=.o)  $(CSRC:.c=.o)

vpath %.c $(sort $(dir $(CSRC:.c=.o)))
vpath %.cpp $(sort $(dir $(SRCS:.cpp=.o)))

OBJS := $(notdir $(SRCS:.cpp=.o)  $(CSRC:.c=.o))
DEPS := $(notdir $(SRCS:.cpp=.d)  $(CSRC:.c=.d))

ifeq ($(USE_CUDA_EGL),yes)
	OBJS += $(notdir $(CUSRC:.cu=.o))
	OBJS += $(notdir $(YOLOSRC:.cpp=.o))
	DEPS += $(notdir $(YOLOSRC:.cpp=.d))
endif	

all: $(APP) $(MKFILE)


%.o: %.cpp $(MKFILE)
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -c $<

%.o: %.c $(MKFILE)
	@echo "Compiling '$<'"
	$(CC) -c $(CPFLAGS) -I . $< -o $@

%.o: %.cu $(MKFILE)
	@echo "Compiling cuda'$<'"
	$(NVC) -c $(CUFLAGS) -I . $< -o $@

%.d: %.cpp $(MKFILE) ./yolov5/logging.h
	@echo "Building dependencies for '$<'"
	@$(CPP) -E -MM -MQ $(<:.cpp=.o) $(CPPFLAGS) $< -o $@
	@$(DEL) $(<:.cpp=.o)

%.d: %.c $(MKFILE)
	@echo "Building dependencies for '$<'"
	@$(CC) -E -MM -MQ $(<:.c=.o) $(CPFLAGS) $< -o $@
	@$(DEL) $(<:.c=.o)		

$(APP): $(OBJS) $(MKFILE)
	@echo "Linking: $@"
	$(CPP) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)
	/bin/cp $(APP) $(ROOT_DST)

clean:
	-$(DEL) $(OBJS:/=\)
	-$(DEL) $(APP:/=\)
	-$(DEL) $(DEPS:/=\)

.PHONY: dep
dep: $(DEPS) $(SRCS) $(CSRC)
	@echo "##########################"
	@echo "### Dependencies built ###"
	@echo "##########################"

-include $(DEPS)