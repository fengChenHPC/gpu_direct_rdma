#include"mpi.h"
#include<stdio.h>
//9 process
int main(int argc,char**argv){
	MPI_Comm localComm, rootComm;
	int rank, size, key, buff;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	key = rank/3;
	buff = rank;

	MPI_Comm_split(MPI_COMM_WORLD, key, rank, &localComm);
	MPI_Comm_split(MPI_COMM_WORLD, rank%3, rank, &rootComm);

    MPI_Allreduce(&rank, &buff, 1, MPI_INT, MPI_SUM, localComm);
    printf("%d, I am process %d, my sum is %d\n", __LINE__, rank, buff);
    MPI_Barrier(MPI_COMM_WORLD);

    if(0 == rank%3) {
        MPI_Allreduce(MPI_IN_PLACE, &buff, 1, MPI_INT, MPI_SUM, rootComm);
    } 
    
    printf("%d, I am process %d, my sum is %d\n", __LINE__, rank, buff);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&buff, 1, MPI_INT, 0, localComm);
    printf("%d, I am process %d, my sum is %d\n", __LINE__, rank, buff);

	 MPI_Finalize(); 

     return 0;
}

