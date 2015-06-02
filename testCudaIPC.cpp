#include <stdio.h>
#include <stdlib.h>

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
    cudaSetDevice(rank);

    cudaIpcMemHandle_t handle;

    const char *HELLO = "Hello, World!";
    size_t bytes = strlen(HELLO) + 1;
    if (rank == 0) {
        void *d;
        checkCUDAError(cudaMalloc(&d, bytes));
        checkCUDAError(cudaMemcpy(d, HELLO, bytes, cudaMemcpyHostToDevice));
        checkCUDAError(cudaIpcGetMemHandle(&handle, d));
        MPI_Send(&handle, sizeof(handle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        checkCUDAError(cudaFree(d));
    } else /*if (rank == 1)*/ {
        MPI_Status status;
        MPI_Recv(&handle, sizeof(handle), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        void* myData;
        checkCUDAError(cudaMalloc((void**)&myData, bytes));

        void *d;
        checkCUDAError(cudaIpcOpenMemHandle(&d, handle, cudaIpcMemLazyEnablePeerAccess));
        char *hello = new char[bytes];
        checkCUDAError(cudaMemcpyPeer(myData, rank, d, 0, bytes));
        checkCUDAError(cudaMemcpy(hello, myData, bytes, cudaMemcpyDeviceToHost));
        checkCUDAError(cudaIpcCloseMemHandle(d));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("%s\n", hello);

        checkCUDAError(cudaFree(myData));
        delete[] hello;

        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}

