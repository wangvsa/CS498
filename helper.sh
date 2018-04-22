#!/bin/bash

echo "2.1 total threads: 64"
./a.out 1 64
echo "2.2 total threads: 128"
./a.out 2 64
echo "2.3 total threads: 256"
./a.out 4 64
echo "2.4 total threads: 512"
./a.out 8 64
echo "2.5 total threads: 1024"
./a.out 16 64
echo "2.6 total threads: 2048"
./a.out 32 64
echo "2.7 total threads: 4096"
./a.out 64 64
