
/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 32
#define BLOCKS 2            
#define NUM_KEYS THREADS*BLOCKS

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
        keys[i] = rand();
    }
}

__global__
void bitonic_sort_step(int *dev_keys, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
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
    bitonic_sort(keys); /* Inplace */
    stop = clock();

    print_elapsed(start, stop);
    print_keys(keys, NUM_KEYS);

    delete keys;
}
