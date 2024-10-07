#!/bin/bash
#$ -cwd

mkdir build
g++ -O2 -std=c++14 ./src/wtmd_lfts_cpu.cc -o ./build/wtmd-lfts-cpu -lfftw3 -lgsl -lgslcblas -lm