#ifndef UTILS_H_
#define UTILS_H_

#include <sys/mman.h>

struct bbox {
    float x;
    float y;
    float w;
    float h;
    int class_num;
};

typedef struct {
    size_t shmem_size;
    size_t nboxes;
    struct bbox boxes[100];
} dev_shmem_t;

dev_shmem_t *init_shmem();
dev_shmem_t *svinit_shmem();
extern dev_shmem_t *dev_shmem;

/* message queue */
struct job {
    char *data;
};

struct queue_t {
    pthread_mutex_t   lock;
    pthread_cond_t    wait_room;
    pthread_cond_t    wait_data;
    unsigned int      size;
    unsigned int      head;
    unsigned int      tail;
    unsigned int      len;
    struct job      **queue;
};

int queue_init(struct queue_t *const q, const unsigned int slots);
int queue_destroy(struct queue_t *const q);
struct job *queue_get(struct queue_t *const q);
int queue_put(struct queue_t *const q, struct job *const j);

dev_shmem_t *init_shmem();

#endif
