#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "mpi.h"

#include "cuda_runtime_api.h"

extern "C" void mallocOnGPU(float**, size_t);
extern "C" void freeOnGPU(float*);
extern "C" void initOnGPU(int, float*, int);
extern void printResult(int, float *x, int n);

int main(int argc, char *argv[]){
	int nproc, rank;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int numGPU;
	cudaGetDeviceCount(&numGPU);
	printf("we have %d gpus\n", numGPU);
	cudaSetDevice(rank % numGPU);

	char name[8192];
	int len;
	MPI_Get_processor_name(name, &len);
	assert(len < 8192);

	printf("I am process %d in %d, I am on node %s\n", rank, nproc, name);

	int size = 256; 
	const int reps = 20;

	float *ptr[2*reps];

	do{	
		//malloc space	
		for(int i = 0; i < reps; i++){
			mallocOnGPU(&(ptr[2*i+0]), size);
			mallocOnGPU(&(ptr[2*i+1]), size);
		}

		//init ptr
		for(int i = 0; i < reps; i++){
		initOnGPU(rank, ptr[2*i+0], size);
		}

		MPI_Status status;
		MPI_Barrier(MPI_COMM_WORLD);
		double s = MPI_Wtime();	

		MPI_Request *request = (MPI_Request*) malloc(sizeof(MPI_Request)*reps);;
		for(int i = 0; i < reps; i++){		
			/*		
					if(0 == rank){
					MPI_Send(ptr[0], size, MPI_FLOAT, 1, 99, MPI_COMM_WORLD);
					MPI_Recv(ptr[1], size, MPI_FLOAT, 1, 99, MPI_COMM_WORLD, &status);
					}else{
					MPI_Recv(ptr[1], size, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, &status);
					MPI_Send(ptr[0], size, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
					}
					*/
			//MPI_Allreduce(ptr[0], ptr[1], size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			//* OpenMPI 1.8.5 doesn't support this
			MPI_Iallreduce(ptr[2*i+0], ptr[2*i+1], size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, request+i);

			//float *temp = ptr[0];
			//	ptr[0] = ptr[1];
			//	ptr[1] = temp;
		}
		//printf result	
		//	printResult(rank, ptr[1], size);
		assert(MPI_SUCCESS == MPI_Waitall(reps, request, MPI_STATUSES_IGNORE));

		double e = MPI_Wtime();	
		double et = (e - s)/reps;
		if(0 == rank){
			//printf("ping pong time = %lfs, data size %ld B, bandwidth = %.3f MB/s\n", et, size*sizeof(int), (float)(size*2*sizeof(int)/et/1024/1024));
			printf("%ld, %.3f\n", size*sizeof(int), (float)(size*2*sizeof(int)/et/1024/1024/1024));
		}

		free(request);

		for(int i = 0; i < reps; i++){
			freeOnGPU(ptr[2*i+0]);
			freeOnGPU(ptr[2*i+1]);
		}

		if(size < 256*1024){
			size += 1024;
		}else if(size < 64*1024*1024){
			size += 16*1024;
		}else{
			size += 4*1024*1024;
		}
	}while(size < 16*1024*1024);

	MPI_Finalize();

	return 0;
}
