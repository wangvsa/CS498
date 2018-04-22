#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "keys.h"
#include "timer.h"

// the following three are parameters get from cmd line
int THREADS;        // e.g.  32
int BLOCKS;         // e.g. 2
int K, NUM_KEYS;       // 2^k

// How many keys one thread should process
// make sure that the number of keys is always larger than total number of threads
// and also make sure its divisible, i.e NUM_KEYS % (THREADS*NUM_KEYS) = 0
int KEYS_PER_THREAD;    // = NUM_KEYS/(THREADS*BLOCKS);


__global__
void bitonic_sort_step(int *dev_keys, int j, int k, int KEYS_PER_THREAD) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    unsigned int tid = (threadIdx.x + blockDim.x * blockIdx.x) * KEYS_PER_THREAD;
    for(i = tid; i < tid+KEYS_PER_THREAD; i++) {
        ixj = i^j;
        /* The threads with the lowest ids sort the array. */
        if ((ixj)>i) {
            if ((i&k)==0) {
                /* Sort ascending */
                if (dev_keys[i]>dev_keys[ixj]) {
                    /* exchange(i,ixj); */
                    float temp = dev_keys[i];
                    dev_keys[i] = dev_keys[ixj];
                    dev_keys[ixj] = temp;
                }
            }
            if ((i&k)!=0) {
                /* Sort descending */
                if (dev_keys[i]<dev_keys[ixj]) {
                    /* exchange(i,ixj); */
                    float temp = dev_keys[i];
                    dev_keys[i] = dev_keys[ixj];
                    dev_keys[ixj] = temp;
                }
            }
        }
    }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *keys) {
    KEYS_PER_THREAD = NUM_KEYS/(THREADS*BLOCKS);


    int *dev_keys;

    cudaMalloc((void**) &dev_keys, NUM_KEYS*sizeof(int));
    cudaMemcpy(dev_keys, keys, NUM_KEYS*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    int j, k;
    /* Major step */
    for (k = 2; k <= NUM_KEYS; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_keys, j, k, KEYS_PER_THREAD);
        }
    }
    cudaMemcpy(keys, dev_keys, NUM_KEYS*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_keys);
}


int main(int argc, char *argv[]) {

    if(argc != 3) return -1;
    BLOCKS = atoi(argv[1]);
    THREADS = atoi(argv[2]);

    for(K = 16; K <= 24; K++) {
        NUM_KEYS = 2 << (K-1);

        int *keys = new int[NUM_KEYS];
        uniform_keys(keys, NUM_KEYS, 0, 2147483647);
        //uniform_keys(keys, NUM_KEYS, 0, 255);
        //gaussian_keys(keys, NUM_KEYS);
        //print_keys(keys, NUM_KEYS);

        timer t;
        bitonic_sort(keys);
        printf("keys: %d, blocks: %d, threads: %d, time(s): %f\n", K, BLOCKS, THREADS, t.seconds_elapsed());

        //print_keys(keys, NUM_KEYS);
        delete keys;
    }

}
