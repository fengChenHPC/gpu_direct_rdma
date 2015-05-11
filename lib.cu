#include <stdlib.h>
#include <stdio.h>

#define cutilSafeCall(err) {\
	if(cudaSuccess != err){\
		printf("%s\n", cudaGetErrorString(err));\
		exit(0);\
	}\
}

extern "C" void mallocOnGPU(float** ptr, size_t size){
	cutilSafeCall(cudaMalloc((void**)ptr, size*sizeof(float)));
}

extern "C" void freeOnGPU(float* ptr){
	cutilSafeCall(cudaFree(ptr));
}

extern "C" void mallocOnHost(float** ptr, size_t size){
	cutilSafeCall(cudaMallocHost((void**)ptr, size*sizeof(float)));
}

extern "C" void freeOnHost(float* ptr){
	cutilSafeCall(cudaFreeHost(ptr));
}

extern "C" void copyToHost(float* h_ptr, float *d_ptr, size_t size){
	cutilSafeCall(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
} 

extern "C" void copyToDevice(float *d_ptr, float *h_ptr, size_t size){
	cutilSafeCall(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
}

static __global__ void init(int rank, float *x, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		x[i] = (float)(rank + n + i);//expf((float)i + n) + sinf((float)i/(gridDim.x*blockDim.x));
}

extern "C" void initOnGPU(int rank, float *ptr, int n){
   	int block = 128;
	int grid = (n+block-1) / block;
	init<<<grid, block>>>(rank, ptr, n);
	cutilSafeCall(cudaDeviceSynchronize());
}

extern "C" void initOnHost(int rank, float *ptr, int n){
	for(int i = 0; i < n; i++){
		ptr[i] = i + n + rank;
	}
}

static __global__ void print(int rank, float *x, int n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < n)
		printf("process %d, thread %d data %.1f\n", rank, i, x[i]);
}

extern void printResult(int rank, float *x, int n){
   	int block = 128;
	int grid = (n+block-1) / block;

	print<<<grid, block>>>(rank, x, n);
	cutilSafeCall(cudaDeviceSynchronize());
}
