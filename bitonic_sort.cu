#define KEYS_NUM 16
#define BLOCK_SIZE 32
#define GRID_SIZE 1

__global__
void bitnoic_sort(int *keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid: %d, %d\n", tid, keys[tid%KEYS_NUM]);

    for(int i=1; i<KEYS_NUM; i*=2) {    // i is the number of segments we should process

        int len = KEYS_NUM / i;         // length of the segment
        int seg = tid / (len / 2);      // find out which segment this thread should work on

        // the index of keys this thread should work on
        int idx = seg * len + tid % (len/2);

        // compare and swap
        if (keys[idx] > keys[idx+len/2]) {
            tmp = keys[idx]
            keys[idx] = keys[idx+len/2]
            keys[idx+len/2] = tmp
        }
    }
}

int main(int argc, char *argv[]) {

    int *keys = new int[KEYS_NUM];
    int *dev_keys;
    cudaMalloc((void**)&dev_keys, sizeof(int)*KEYS_NUM);
    cudaMemcpy(dev_keys, keys, KEYS_NUM*sizeof(int), cudaMemcpyHostToDevice);

    bitonic_sort<<<GRID_SIZE, BLOCK_SIZE>>>(dev_keys);
}

