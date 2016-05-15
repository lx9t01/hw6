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

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);            \
    return EXIT_FAILURE;}} while(0)



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

    float* dev_points; // to determine the timestep
    float* dev_points_2; // to determine the reaction

    const int N = 100; // each iteration there is N simulations running

    cudaMalloc((void**)&dev_points, N * sizeof(float));
    cudaMalloc((void**)&dev_points_2, N * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    float* state;
    float* dev_concentration;
    float* dev_timestep;

    cudaMalloc((void**) &state, N * sizeof(float));
    cudaMalloc((void**) &dev_concentration, N * sizeof(float));
    cudaMalloc((void**) &dev_timestep, N * sizeof(float));

    cudaMemset(state, 0, N * sizeof(float));
    cudaMemset(dev_concentration, 0, N * sizeof(float));
    cudaMemset(dev_timestep, 0, N * sizeof(float));

    float* dev_accu_time;
    cudaMalloc((void**) &dev_accu_time, N * sizeof(float));
    cudaMemset(dev_accu_time, 0, N * sizeof(float));
    float* host_min_time = (float*)malloc(1 * sizeof(float));
    memset(host_min_time, 0, 1 * sizeof(float));

    float* dev_min_time;
    cudaMalloc((void**)&dev_min_time, 1 * sizeof(float));


    // resampling the data in vectors
    const int T = 1000; // the total time interval after resampling

    // the matrix for resampled data
    float* resamp_X = (float*)malloc(N * T * sizeof(float));

    float* dev_resample_X;
    cudaMalloc((void**)&dev_resample_X, N * T * sizeof(float));
    cudaMemset(dev_resample_X, 0.000, N * T * sizeof(float));

    const float final_time = 100;
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    cudaError err; 

    float* test = (float*)malloc(N * T * sizeof(float));
    // cudaMemcpy(test, dev_resample_X, N * T * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < N * T; ++i) {
    //     printf("%f ", test[i]);
    // }
    float* test_accu = (float*)malloc(N * sizeof(float));
    
    while (*host_min_time <= final_time) {
        curandGenerateUniform(gen, dev_points, N);
        curandGenerateUniform(gen, dev_points_2, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "curand No kernel error detected" << endl;
        }
        
        // for each iteration, call a kernel
        // calculates state, X concentration, timestep, accumulate time
        cudaCallGillKernel(blocks, threadsPerBlock, dev_points, dev_points_2, state, dev_concentration, dev_timestep, dev_accu_time, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "gill No kernel error detected" << endl;
        }
        // cudaMemcpy(test, dev_concentration, N * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(test_accu, dev_concentration, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += test_accu[i];
        }
        sum /= N;

        printf("after kernel, avg X: %f\n", sum);

        // run a reduction kernel to find the minimum accumulate time       
        cudaCallFindMinKernel(blocks, threadsPerBlock, dev_accu_time, dev_min_time, N);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "findmin No kernel error detected" << endl;
        }
        gpuErrchk(cudaMemcpy(host_min_time, dev_min_time, 1 * sizeof(float), cudaMemcpyDeviceToHost));

        // float new_min = 99999;
        // for (int i = 0; i < N; ++i) {
        //     if (test_accu[i] < new_min) {
        //         new_min = test_accu[i];
        //     }
        // }
        // *host_min_time = new_min;

        printf("min get ");
        printf("this min: %f\n", *host_min_time);

        cudaCallResampleKernel(blocks, threadsPerBlock, dev_resample_X, dev_concentration,dev_accu_time, N, T);
        err = cudaGetLastError();
        if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
        } else {
            cerr << "resemple No kernel error detected" << endl;
        }
        getchar();
    }
    free(test);
    free(test_accu);

    cudaMemcpy(resamp_X, dev_resample_X, N * T * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < T; ++j) {
    //         printf("%f ", resamp_X[i * T + j]);
    //     }
    //     printf("\n");
    // }
    float* dev_mean;
    float* dev_var;
    cudaMalloc((void**)&dev_mean, T * sizeof(float));
    cudaMalloc((void**)&dev_var, T * sizeof(float));

    float* host_mean = (float*)malloc(T * sizeof(float));
    float* host_var = (float*)malloc(T * sizeof(float));

    cudaCallMeanVarKernel(blocks, threadsPerBlock, dev_resample_X, dev_mean, dev_var, N, T);
    err = cudaGetLastError();
    if  (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
            // break;
    } else {
        cerr << "resemple No kernel error detected" << endl;
    }
    cudaMemcpy(host_mean, dev_mean, T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_var, dev_var, T * sizeof(float), cudaMemcpyDeviceToHost);


    FILE *gpumin = fopen("GPU_mean.txt", "w");
    for (int i = 0; i < T; ++i) {
        {
            fprintf(gpumin, "%f ", host_mean[i]);
        }
    }
    fclose(gpumin);

    FILE *gpuvar = fopen("GPU_var.txt", "w");
    for (int i = 0; i < T; ++i) {
        {
            fprintf(gpuvar, "%f ", host_var[i]);
        }
    }
    fclose(gpuvar);



    FILE *total_resample_file = fopen("resample.txt", "w");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < T; ++j) {
            fprintf(total_resample_file, "%f ", resamp_X[i * T + j]);
        }
        fprintf(total_resample_file, "\n");
    }
    fclose(total_resample_file);


    // find the mean and var
    float* mean = new float[T]();
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < N; ++j) {
            mean[i] += resamp_X[j * T + i];
        }
        mean[i] /= N;
    }





    FILE *outputFile = fopen("output_mean_CPU.txt", "w");
    for (int i = 0; i < T; ++i) {
        fprintf(outputFile, "%f ",mean[i]);
    }
    fclose(outputFile);


    delete mean;
    free(host_min_time);
    cudaFree(state);
    cudaFree(dev_concentration);
    cudaFree(dev_points);
    cudaFree(dev_points_2);
    cudaFree(dev_timestep);
    cudaFree(dev_accu_time);
    cudaFree(dev_min_time);
    free(resamp_X);

    cudaFree(dev_mean);
    cudaFree(dev_var);
    free(host_mean);
    free(host_var);


    return EXIT_SUCCESS;
}
