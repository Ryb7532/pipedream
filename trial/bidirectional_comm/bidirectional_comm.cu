#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "mpi.h"


#define MPICHECK(cmd) do {                                  \
    int e = cmd;                                            \
    if (e != MPI_SUCCESS) {                                 \
        printf("Failed: MPI error %s:%d '%d'\n",            \
                __FILE__, __LINE__, e);                     \
    }                                                       \
} while (0)


#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
        printf("Failed: CUDA error %s:%d '%s'\n",           \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                       \
} while (0)


#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                   \
    if (r != ncclSuccess) {                                 \
        printf("Failed: NCCL error %s:%d '%s'\n",           \
                __FILE__, __LINE__, ncclGetErrorString(r)); \
    }                                                       \
} while (0)


static uint64_t getHostHash(const char* string) {
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i<maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}


int main(int argc, char* argv[]) {
    int size = 32*1024*1024;


    int myRank, nRanks, localRank = 0;


    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }


    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    int root = nRanks-1;


    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


    CUDACHECK(cudaSetDevice(localRank));
    if (myRank == root) {
        CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float) * (nRanks-1)));
        CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float) * (nRanks-1)));
    } else {
        CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
        CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    }
    CUDACHECK(cudaStreamCreate(&s));


    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


    NCCLCHECK(ncclGroupStart());
    if (myRank == root) {
        size_t offset = size;
        for (int r=0; r<nRanks-1; r++) {
            ncclRecv(recvbuff+r*offset, size, ncclFloat, r, comm, s);
        }
        for (int r=0; r<nRanks-1; r++) {
            ncclSend(sendbuff+r*offset, size, ncclFloat, r, comm, s);
        }
    } else {
        ncclRecv(recvbuff, size, ncclFloat, root, comm, s);
        ncclSend(sendbuff, size, ncclFloat, root, comm, s);
    }
    // if (myRank == root) {
    //     ncclBcast(recvbuff, size, ncclFloat, 0, comm, s);
    //     ncclBcast(sendbuff, size, ncclFloat, root, comm, s);
    // } else {
    //     ncclBcast(recvbuff, size, ncclFloat, root, comm, s);
    //     ncclBcast(recvbuff, size, ncclFloat, 0, comm, s);
    // }
    NCCLCHECK(ncclGroupEnd());

    CUDACHECK(cudaStreamSynchronize(s));


    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));


    ncclCommDestroy(comm);


    MPICHECK(MPI_Finalize());


    printf("[MPI Rank %d] Succcess\n", myRank);
    return 0;
}