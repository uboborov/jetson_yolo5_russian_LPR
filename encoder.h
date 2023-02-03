#ifndef __VIDEOENCODER_H__
#define __VIDEOENCODER_H__

#include <fstream>
#include <iostream>
#include <set>
#include <NvVideoEncoder.h>

class NvBuffer;

/*
 * A helper class to simplify the usage of V4l2 encoder
 * Steps to use this class
 *   (1) Create the object
 *   (2) Call setBufferDoneCallback. The callback is called to return buffer to caller
 *   (3) Call initialize
 *   (4) Feed encoder by calling encodeFromFd
 *   (5) Call shutdown
 */
class VideoEncoder
{
public:
    VideoEncoder(const char *name, const char *outputFilename,
            int width, int height, uint32_t pixfmt = V4L2_PIX_FMT_H265);
    ~VideoEncoder();

    bool initialize();
    bool shutdown();
    void set_bitrate(unsigned long br);

    // Encode API
    bool encodeFromFd(int dmabuf_fd);

    // Callbackt to return buffer
    void setBufferDoneCallback(void (*callback)(int, void*), void *arg)
    {
        m_callback = callback;
        m_callbackArg = arg;
    }

private:

    NvVideoEncoder *m_VideoEncoder;     // The V4L2 encoder
    bool createVideoEncoder();

    static bool encoderCapturePlaneDqCallback(
            struct v4l2_buffer *v4l2_buf,
            NvBuffer *buffer,
            NvBuffer *shared_buffer,
            void *arg)
    {
        VideoEncoder *thiz = static_cast<VideoEncoder*>(arg);
        return thiz->encoderCapturePlaneDqCallback(v4l2_buf, buffer, shared_buffer);
    }

    bool encoderCapturePlaneDqCallback(
            struct v4l2_buffer *v4l2_buf,
            NvBuffer *buffer,
            NvBuffer *shared_buffer);

    std::string m_name;     // name of the encoder
    int m_width;
    int m_height;
    uint32_t m_pixfmt;
    std::string m_outputFilename;
    std::ofstream *m_outputFile;
    std::set<int> m_dmabufFdSet;    // Collection to track all queued buffer
    void (*m_callback)(int, void*);        // Output plane DQ callback
    void *m_callbackArg;
};

#endif  // __VIDEOENCODER_H__
