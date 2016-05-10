/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* TODO: You'll need a kernel here, as well as any helper functions
to call it */
__global__
void
cudaWaveKernel(const float* dev_old_data,
    const float* dev_cur_data,
    const float* dev_new_data,
    float* file_output,
    const size_t numberOfNodes,
    const float courantSquared,
    const float dx,
    const float dt) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx >= 1 && idx <= numberOfNodes - 2) {
        dev_new_data[idx] = 2 * dev_cur_data[idx] - dev_old_data + courantSquared*dev_cur_data[idx+1] - 2*dev_cur_data[idx] + dev_cur_data[idx-1];
        idx += blockDIm.x * gridDim.x;
    }
}


void cudaCallWaveKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock,
    const float* dev_old_data,
    const float* dev_cur_data,
    const float* dev_new_data,
    float* file_output,
    const size_t numberOfNodes,
    const float courantSquared,
    const float dx,
    const float dt) {
    cudaWaveKernel<<<blocks, threadsPerBlock>>> (dev_old_data, dev_cur_data, dev_new_data, file_output, numberOfNodes, courantSquared, dx, dt);
}