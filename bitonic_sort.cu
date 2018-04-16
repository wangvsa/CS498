#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 32
#define BLOCKS 2
#define NUM_KEYS 128

// How many keys one thread should process
// make sure that the number of keys is always larger than total number of threads
// and also make sure its divisible, i.e NUM_KEYS % (THREADS*NUM_KEYS) = 0
#define KEYS_PER_THREAD NUM_KEYS/(THREADS*BLOCKS)


void print_elapsed(clock_t start, clock_t stop) {
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

void print_keys(int *keys, int length) {
    for (int i = 0; i < length; ++i) {
        printf("%d ",  keys[i]);
    }
    printf("\n");
}

void generate_keys(int *keys, int length) {
    srand(time(NULL));
    for (int i = 0; i < length; ++i) {
        keys[i] = rand() % NUM_KEYS;
    }
}

__global__
void bitonic_sort_step(int *dev_keys, int j, int k) {
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
            bitonic_sort_step<<<blocks, threads>>>(dev_keys, j, k);
        }
    }
    cudaMemcpy(keys, dev_keys, NUM_KEYS*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_keys);
}

int main(void)
{
    clock_t start, stop;

    int *keys = new int[NUM_KEYS];
    generate_keys(keys, NUM_KEYS);
    print_keys(keys, NUM_KEYS);

    start = clock();
    bitonic_sort(keys);
    stop = clock();

    print_elapsed(start, stop);
    print_keys(keys, NUM_KEYS);

    delete keys;
}
