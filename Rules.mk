###############################################################################
#
# Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

# Clear the flags from env
CPPFLAGS :=
LDFLAGS :=

# Verbose flag
ifeq ($(VERBOSE), 1)
AT =
else
AT = @
endif

# ARM ABI of the target platform
ifeq ($(TEGRA_ARMABI),)
TEGRA_ARMABI ?= aarch64-linux-gnu
endif

# Use absolute path for better access from everywhere

#CUR_DIR 	:= $(shell pwd | awk '{split($$0, f, "/samples"); print f[1]}')
#TOP_DIR		:= $(shell pwd | awk '{split($$0, f, "/$(notdir $(CUR_DIR))"); print f[1]}')
TOP_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
CLASS_DIR 	:= $(TOP_DIR)/common/classes

ifeq ($(shell uname -m), aarch64)
	CROSS_COMPILE :=
	TARGET_ROOTFS ?=
else
	PATH=$PATH:/usr/local/arm/aarch64-linux-gnu-7.5.0/bin
	CROSS_COMPILE ?= aarch64-linux-gnu-
#	TARGET_ROOTFS := $(ROOT_DIR)
	TARGET_ROOTFS ?= /home/ubobrov/develop/projects/nvidia/jetson_nano/Linux_for_Tegra/rootfs
	MAKE := /usr/bin/make
endif

# Location of the target rootfs
ifeq ($(shell uname -m), aarch64)
TARGET_ROOTFS ?=
else
ifeq ($(TARGET_ROOTFS),)
	$(error Please specify the target rootfs path if you are cross-compiling)
endif
endif

AS             = $(AT) $(CROSS_COMPILE)as
LD             = $(AT) $(CROSS_COMPILE)ld
CC             = $(AT) $(CROSS_COMPILE)gcc
CPP            = $(AT) $(CROSS_COMPILE)g++
AR             = $(AT) $(CROSS_COMPILE)ar
NM             = $(AT) $(CROSS_COMPILE)nm
STRIP          = $(AT) $(CROSS_COMPILE)strip
OBJCOPY        = $(AT) $(CROSS_COMPILE)objcopy
OBJDUMP        = $(AT) $(CROSS_COMPILE)objdump
NVCC           = $(AT) $(CUDA_PATH)/bin/nvcc -ccbin $(filter-out $(AT), $(CPP))

# Specify the logical root directory for headers and libraries.
ifneq ($(TARGET_ROOTFS),)
CPPFLAGS += --sysroot=$(TARGET_ROOTFS)
LDFLAGS += \
	-Wl,-rpath-link=$(TARGET_ROOTFS)/lib/$(TEGRA_ARMABI) \
	-Wl,-rpath-link=$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI) \
	-Wl,-rpath-link=$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)/tegra \
	-Wl,-rpath-link=$(TARGET_ROOTFS)/$(CUDA_PATH)/lib64
endif

# All common header files
CPPFLAGS += -std=c++11 \
	-I"$(TOP_DIR)/include" \
	-I"$(TOP_DIR)/include/libjpeg-8b" \
	-I"$(ALGO_CUDA_DIR)" \
	-I"$(ALGO_TRT_DIR)" \
	-I"$(TARGET_ROOTFS)/$(CUDA_PATH)/include" \
	-I"$(TARGET_ROOTFS)/usr/include/$(TEGRA_ARMABI)" \
	-I"$(TARGET_ROOTFS)/usr/include/libdrm" \
	-I"$(TARGET_ROOTFS)/usr/include/opencv4"

# All common dependent libraries
LDFLAGS += \
	-lpthread -lnvv4l2 \
	-lnvbuf_utils -lnvjpeg -lnvosd -lrt \
	-L"$(TARGET_ROOTFS)/$(CUDA_PATH)/lib64" \
	-L"$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)" \
	-L"$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)/tegra"
