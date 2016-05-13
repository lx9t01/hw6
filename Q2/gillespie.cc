// gillespie.cc

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>


#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>

#include "gillespie_cuda.cuh"


using std::cerr;
using std::cout;
using std::endl;

/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


void check_args(int argc, char **argv){
    if (argc != 3){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "./gillespie <threads per block> <max number of blocks>\n";
        exit(EXIT_FAILURE);
    }
}


int main (int argc, char** argv) {

    check_args(argc, argv);
    const int threadsPerBlock = atoi(argv[1]);
    const int blocks = atoi(argv[2]);

    float* dev_points;
    const int N = 100; // each iteration there is N simulations running
    const int rand_number = 2 * N; // number of random numbers, which is 2 times of N

    cudaMalloc((void**)&dev_points, rand_number * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    float* state;
    float* dev_X;
    float* dev_timestep;
    // float* host_X[] = new float[T*N](); // after each resampling, the data will be stored here

    cudaMalloc((void**)&state, N * sizeof(float));
    cudaMemset(state, 0, N * sizeof(float));
    cudaMalloc((void**)&dev_X, N * sizeof(float));
    cudaMemset(dev_X, 0, N * sizeof(float));
    cudaMalloc((void**)&dev_timestep, N * sizeof(float));
    cudaMemset(dev_timestep, 0, N * sizeof(float));

    float* dev_accu_time;
    cudaMalloc((void**)&dev_accu_time, N * sizeof(float));
    cudaMemset(dev_accu_time, 0, N * sizeof(float));
    float host_min_time = 0;
    float* dev_min_time;
    cudaMalloc((void**)&dev_min_time, 1 * sizeof(float));

    // std::vector< std::vector<float> > vector_X;
    // std::vector< std::vector<float> > vector_accu_time;


    // resampling the data in vectors
    const int T = 1000; // the total time interval after resampling

    // the matrix for resampled data
    float* resamp_X = new float[N*T]();

    float* dev_resample_X;
    cudaMalloc((void**)&dev_resample_X, N * sizeof(float));
    // the matrix to mark if a time point has been ipdated
    int* is_resampled = new int[N*T]();
    int* dev_is_resampled;
    cudaMalloc((void**)&dev_is_resampled, N * sizeof(float));

    const float final_time = 100;
    curandSetPseudoRandomGeneratorSeed(gen, 1234); 

    while (host_min_time <= final_time) {
        curandGenerateUniform(gen, dev_points, rand_number);
        printf("rand number generated\n");
        cudaError err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
                break;
        } else {
                cerr << "No kernel error detected" << endl;
                
        }
        printf("Gill kernel called\n");
        // for each iteration, call a kernel
        // calculates state, X concentration, timestep, accumulate time
        cudaCallGillKernel(blocks, threadsPerBlock, dev_points, state, dev_X, dev_timestep, dev_accu_time, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
                break;
        } else {
                cerr << "No kernel error detected" << endl;
                
        }
        printf("Gill kernel called\n");
        // float* host_state = new float[N]();
        // float* host_X = new float[N](); // TODO destruct!!!!!!!!!!!!!!!
        // // float* host_timestep = new float[N]();
        // // cudaMemcpy(host_state, state, N * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(host_X, dev_X, N * sizeof(float), cudaMemcpyDeviceToHost);
        // // cudaMemcpy(host_timestep, dev_timestep, N * sizeof(float), cudaMemcpyDeviceToHost);
        // float* host_accu_time = new float[N]();
        // cudaMemcpy(host_accu_time, dev_accu_time, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaCallResampleKernel(blocks, threadsPerBlock, dev_resample_X, dev_is_resampled, dev_X, dev_accu_time, N, T);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
                break;
        } else {
                cerr << "No kernel error detected" << endl;
                
        }
        printf("resampled\n");
        // std::vector<float> v_X(std::begin(host_X), std::end(host_X)); // c++ 11
        // std::vector<float> v_accu_time(std::begin(host_accu_time), std::end(host_accu_time));

        // vector_X.push_back(v_X);
        // vector_accu_time.push_back(v_accu_time);

        // run a reduction kernel to find the minimum accumulate time       
        cudaCallFindMinKernel(blocks, threadsPerBlock, dev_accu_time, dev_min_time, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
                break;
        } else {
                cerr << "No kernel error detected" << endl;
                
        }
        printf("min get\n");
        cudaMemcpy(&host_min_time, dev_min_time, 1 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("%f\n", host_min_time);

        // delete[] host_X;
        // delete[] host_accu_time;
    }

    // FILE *outputFile = fopen("output.txt", "w");
    // for (int i = 0; i < T; ++i) {
    //     fprintf(outputFile, "%f ",host_X[i]);
    // }

    // //
    // fclose(outputFile);
    curandDestroyGenerator(gen);
    cudaFree(state);
    cudaFree(dev_X);
    cudaFree(dev_points);
    cudaFree(dev_timestep);
    cudaFree(dev_accu_time);
    cudaFree(dev_min_time);

    // delete[](host_X);

    return EXIT_SUCCESS;
}
