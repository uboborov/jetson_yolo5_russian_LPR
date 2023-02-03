#ifndef _RTSP_CLNT_H
#define _RTSP_CLNT_H

typedef int (*rtsp_data_cb_t)(void *, void *, int);
int start_rtsp_client(char *url, rtsp_data_cb_t cb, void *user_data);
void stop_rtsp_client();

#endif
