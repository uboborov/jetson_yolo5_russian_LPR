# Nvidia jetson-nano yolov5 russian license plate recognition
Nvidia jetson nano + RTSP IP camera + yolov5 + russian license plate recognition

## The project if for building on HOST machine (not on target jetson-nano board)

## This demo project is distibuted as is without any support
### requires:
* Linux machine with any distribution as a build host
* CUDA [Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive "Toolkit 10.2") for x86 installed
* CUDA == 10.2, TensorRT >= 8.0, OpenCV+CUDA >= 4.4.x libraries prebuilt for ARM64 + headers
* [live555](http://www.live555.com/liveMedia/public/) library prebuilt for ARM64 + headers
* [Linux for Tegra](https://disk.yandex.ru/d/fzU_KEtDGLtYvA) rootfs deployed on the build host
* [quemu-user-static](https://github.com/multiarch/qemu-user-static) for deploying Linux for Tegra rootfs
* GCC compiler for ARM64 (aarch64-linux-gnu-7.5.0 has been used)

### prepare:
* download and unpack Linux for Tegra roots
* with root privilegies run `./apply_binaries.sh` from the Linux_for_Tegra directory
* install CUDA Toolkit
* download or install GCC compiler
* prepare libraries form Nvidia (TensorRT, OpenCV+CUDA, CUDA) for ARM64
* change paths in Makefile and Rules.mk 

### build
run `make`

TensorRT engines (yolov5m_lpr_pre_v2.engine, yolov5s_lpr_plate.engine) for jetson-nano are in **engines** directory

### run on jetson-nano board
**run from the RTSP ip camera:**
`./lpr -u rtsp://user:psssword@192.168.100.64:554 -w 1920 -h 1080 -f H264`
**run from the H264 video file: (result output to the H264 file lpr.h264 )**
`./lpr -i ./city_drive.h264 -o ./lpr.h264 -w 1280 -h 720 -f H264`
**run from the folder with images: (result images to another folder img_out)**
`./lpr -F img_in -T img_out`

![img1](img1.jpg)
![img2](img2.jpg)
![img3](img3.jpg)
![img4](img4.jpg)
