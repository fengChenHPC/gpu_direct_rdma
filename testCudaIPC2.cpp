#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>

#include <cuda_runtime_api.h>

#define checkCUDAError(err) {\
	cudaError_t cet = err;\
	if(cudaSuccess != cet) {\
		printf("%d %s\n", __LINE__, cudaGetErrorString(cet));\
	}\
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int numProcesses;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	cudaSetDevice(rank);

	int h_data;
	size_t bytes = sizeof(int);

	int *data;
	checkCUDAError(cudaMalloc((void**)&data, bytes));

	cudaIpcMemHandle_t handle;
	checkCUDAError(cudaIpcGetMemHandle(&handle, data));
	cudaIpcMemHandle_t *allHandles = (cudaIpcMemHandle_t*) malloc(sizeof(cudaIpcMemHandle_t)*numProcesses);
	MPI_Allgather(&handle, sizeof(handle), MPI_BYTE, allHandles, sizeof(handle), MPI_BYTE, MPI_COMM_WORLD);

	int* *ds = (int**) malloc(sizeof(int*)*numProcesses);
	for(int i = 0; i < numProcesses; i++) {
		//can't open handle created by itself
		if(i == rank) {
			ds[i] = data;	
		} else {
			checkCUDAError(cudaIpcOpenMemHandle((void**)(ds+i), allHandles[i], cudaIpcMemLazyEnablePeerAccess));
		}
	}

	for(int i = 0; i < numProcesses; i++) {
		int temp = (1+rank)*100+i;
		MPI_Barrier(MPI_COMM_WORLD);
		checkCUDAError(cudaMemcpy(data, &temp, sizeof(int), cudaMemcpyHostToDevice));
		MPI_Barrier(MPI_COMM_WORLD);
		for(int j = 0; j < numProcesses; j++) {
			int t;
			checkCUDAError(cudaMemcpy(&t, ds[j], sizeof(int), cudaMemcpyDeviceToHost));
			assert(t == (j+1)*100+i);
			//printf("%d %d %d %d\n", i, j, rank, t);
		}
	}
		
	for(int i = 0; i < numProcesses; i++) {
		if(i != rank) checkCUDAError(cudaIpcCloseMemHandle(ds[i]));
	}

	free(ds);
	free(allHandles);
	checkCUDAError(cudaFree(data));

	MPI_Finalize();

	return 0;
}

