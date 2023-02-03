#include <errno.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>

#include "utils.h"

using namespace std;

int file_exist(const char *filename) {
  struct stat   buffer;   
  return (stat(filename, &buffer) == 0);
}

/*
*/
int queue_init(struct queue_t *const q, const unsigned int slots) {
    if (!q || slots < 1U)
        return errno = EINVAL;

    q->queue = (struct job **)malloc(sizeof (struct job *) * (size_t)(slots + 1));
    if (!q->queue)
        return errno = ENOMEM;
    memset(q->queue, 0, sizeof (struct job *) * (size_t)(slots + 1));

    q->size = slots;// + 1U; 
    q->head = 0U;
    q->tail = 0U;
    q->len  = 0U;

    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->wait_room, NULL);
    pthread_cond_init(&q->wait_data, NULL);

    return 0;
}

/*
*/
int queue_destroy(struct queue_t *const q) {
    if (q->queue) {
        free(q->queue);
    }

    return 0;
}

/*
*/
struct job *queue_get(struct queue_t *const q) {
    struct job *j;

    while (q->head == q->tail || !q->len)
        pthread_cond_wait(&q->wait_data, &q->lock);

    j = q->queue[q->tail];
    q->queue[q->tail] = NULL;
    q->tail = (q->tail + 1U) % q->size;
    q->len--;

    return j;
}

/*
*/
int queue_put(struct queue_t *const q, struct job *const j) {
    if ((q->head + 1U) % q->size == q->tail || q->len >= q->size) {
        //pthread_cond_signal(&q->wait_data);
        return -1;
    }
    q->head = (q->head + 1U) % q->size;

    q->queue[q->head] = j;
    q->len++;

    if (q->len == 1)
        pthread_cond_signal(&q->wait_data);

    return 0;
}

dev_shmem_t * dev_shmem = NULL;

static void * get_shmem(size_t len) {
    void *mem = NULL;
#ifdef SYSV_SHM
    char fname[] = "/tmp/";
#else
    int fd; 
    char fname[sizeof("shmem_XXX")];

    sprintf(fname, "shmem_0");
#endif
#ifdef SYSV_SHM
    // Is it ok +1?
    mem = shmat(shmget(ftok(fname, findex+1), len, 0777|IPC_CREAT),NULL, 0);
    if(mem == (void *)-1)
    {
        syslog(LOG_ERR, FILE_LINE_STR "shmat(2) failed:%s", strerror(errno));
        return NULL;
    }
#else
    fd = shm_open(fname, O_RDWR, S_IRWXU | S_IRWXG);
    if (fd == -1) {
        printf("shm_open(3) failed:%s\n", strerror(errno));
        return NULL;
    }
    
    mem = mmap(0, len, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    if(mem == MAP_FAILED)
    {
        printf("mmap(3) failed:%s\n", strerror(errno));
        return NULL;
    }
    close(fd);
#endif
    
    return mem;

}

dev_shmem_t *init_shmem() {
    if(dev_shmem == NULL) { // Just one shmem 
        if((dev_shmem = (dev_shmem_t *)get_shmem(sizeof(dev_shmem))) == NULL) {
            return NULL;
        }

        if((dev_shmem->shmem_size != 0) && (dev_shmem->shmem_size != sizeof(dev_shmem_t))) {
            printf("ERROR: sizeof(ntx_shmem_t) = %u, but must be %u\n",sizeof(dev_shmem_t), dev_shmem->shmem_size);
        }
    }
    printf("!!! SHMEM OK !!!!!\n");
    return dev_shmem;
}
