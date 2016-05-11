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
    float* dev_new_data,
    const size_t numberOfNodes,
    const float courantSquared) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx >= 0 && idx <= numberOfNodes - 2) {
        if (idx == 0) {
            idx += blockDim.x * gridDim.x;
        }
        dev_new_data[idx] = 2 * dev_cur_data[idx] - dev_old_data[idx] + courantSquared * (dev_cur_data[idx+1] - 2*dev_cur_data[idx] + dev_cur_data[idx-1]);
        idx += blockDim.x * gridDim.x;
    }
}


void cudaCallWaveKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock,
    const float* dev_old_data,
    const float* dev_cur_data,
    float* dev_new_data,
    const size_t numberOfNodes,
    const float courantSquared) {
    cudaWaveKernel<<<blocks, threadsPerBlock>>> (dev_old_data, dev_cur_data, dev_new_data, numberOfNodes, courantSquared);
}