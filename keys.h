/**
 * Generate N keys:
 * N= 2^k, k=16, 17...24
 *
 * The keys are
 * A. distributed uniformly in the rang 0..2^31-1
 * B. Distributed in the same range, with a Gaussian distribution
 * C. Distributed uniformly in the range 0..255
 *
 */
#ifndef KEYS_H
#define KEYS_H

#include <iostream>
#include <stdlib.h>
#include <random>

void print_keys(int *keys, int num_keys) {
    for(int i = 0; i < num_keys; i++)
        printf("%d ", keys[i]);
}

void random_keys(int *keys, int length) {
    srand(time(NULL));
    for (int i = 0; i < length; ++i) {
        keys[i] = rand() % length;
    }
}

void uniform_keys(int *keys, int num_keys, int min, int max) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int>  dist(min, max); 
    for(int i = 0; i < num_keys; i++) {
        keys[i] = dist(generator);
    }
}

void gaussian_keys(int *keys) {
}


#endif
