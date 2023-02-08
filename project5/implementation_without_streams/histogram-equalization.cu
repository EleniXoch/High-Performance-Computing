#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "hist-equ.h"

__global__ void equalizeIMG(unsigned char * input, unsigned char * output, int *lut, int size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < size) {
        output[tid] = (unsigned char)lut[input[tid]];
        tid += stride;
    }

}

__global__ void calcH(unsigned char *input, int *bins, int size) {
    __shared__ unsigned int cache[256];

    if (threadIdx.x < 256) {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < size) {
        atomicAdd(&(cache[input[tid]]), 1);
        tid += stride;
    }

    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&(bins[threadIdx.x]), cache[threadIdx.x]);
    }

}

__global__ void createCDF(float *cdf, int *bins, int size) {
    __shared__ float T[2 * 256];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < 256) {
        T[threadIdx.x] = (float)bins[idx];
    }

    __syncthreads();
    //REduction
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;

        //blockDIm.x = 256 <<<1,256>>>
        if (index < blockDim.x) {
            T[index] += T[index - stride];
        }

    }

    for (int stride = 256 / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride*2 - 1;

        if (index + stride < 256) {
            T[index + stride] += T[index];
        }
 
    }

    __syncthreads();
 
    for (int i = 0; i < 256; i++) {
        cdf[i] = T[i];
    }

}

// No indexing needed for <<<1,256>>>
__global__ void calcLut(float* cdf, int* lut, int min, int N_bins, int diff) {
    lut[threadIdx.x] = (int)((cdf[threadIdx.x] - min) * 255 / diff + 0.5);
    if (lut[threadIdx.x] < 0) {
        lut[threadIdx.x] = 0;
    }
    //Check for performance diff with equalize if > 255
    if (lut[threadIdx.x] > 255) {
        lut[threadIdx.x] = 255;
    }
}


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    //Done in GPU
    for ( i = 0; i < img_size; i ++) {
        hist_out[img_in[i]] ++;
    }
}

// void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
//                             int * hist_in, int img_size, int nbr_bin, float *cdf_d, int *lut){
//     int i;
    
//     // /* Construct the LUT by calculating the CDF */
//     // cdf = 0;
//     // min = 0;
//     // i = 0;
//     // //Get the first non zero position from the histogram array 0 0 1 2 -> min = 1
//     // while(min == 0){
//     //     min = hist_in[i++];
//     // }
//     // d = img_size - min;
    
//     // for(i = 0; i < nbr_bin; i ++){
//     //     cdf += hist_in[i];
//     //     //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
//     //     // 0.5 ?
//     //     lut[i] = (int)((cdf_d[i] - min)*255/d + 0.5);
        
//     //     if(lut[i] < 0){
//     //         lut[i] = 0;
//     //     }
        
        
//     // }
    
//     /* Get the result image */
//     // for(i = 0; i < img_size; i ++){
//     //     if(lut[img_in[i]] > 255){
//     //         img_out[i] = 255;
//     //     }
//     //     else{
//     //         img_out[i] = (unsigned char)lut[img_in[i]];
//     //     }
        
//     // }
// }
