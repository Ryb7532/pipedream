#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

using namespace std;

double get_elapsed_time(struct timeval *begin, struct timeval *end)
{
  return (end->tv_sec - begin->tv_sec) * 1000000
    + (end->tv_usec - begin->tv_usec);
}

int main(int argc, char *argv[])
{
  struct timeval t1, t2;
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  stringstream stm0,stm1;
  stm0 << rank;
  stm1 << size;

  string cmd = "./python.sh -m launch --nnodes ";
/*
  string cmd = "nvprof --profile-child-processes -o alexnet_4_hybrid_";
  cmd += stm0.str();
  cmd += "_%p.nvp python3 -m launch --nnodes ";
*/

  cmd += stm1.str();
  cmd += " --node_rank ";
  cmd += stm0.str();
  cmd += " --nproc_per_node 4 main_with_runtime.py -s --distributed_backend gloo -m models.alexnet.gpus=4 --epochs 6 -b 256 --config_path models/alexnet/gpus=4/hybrid_conf.json  --num_ranks_in_server 4 --master_addr ";
  cmd += (string)argv[1];

  cout << cmd << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  gettimeofday(&t1, NULL);
  int result = system(cmd.c_str());
  gettimeofday(&t2, NULL);

  if (result < 0) {
    cout << "Failed to run Rank " << rank << endl;
    MPI_Abort(MPI_COMM_WORLD,1);
    return 0;
  }

  double time = get_elapsed_time(&t1, &t2);
  printf("Rank %d : %.3lf seconds\n", rank, time/1000000.00);

  MPI_Finalize();
  return 0;
}
