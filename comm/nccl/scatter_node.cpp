#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

using namespace std;

template < typename T >
string to_string( const T& n ) {
  std::ostringstream stm ;
  stm << n ;
  return stm.str() ;
}

int main(int argc, char *argv[])
{
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  string cmd = "python3 -m launch --nnodes ";
/*
  string cmd = "nvprof --profile-child-processes -o alexnet_32_dp_";
  cmd += stm0.str();
  cmd += "_%p.nvp python3 -m launch --nnodes ";
*/

  cmd += to_string(size);
  cmd += " --node_rank ";
  cmd += to_string(rank);
  cmd += " --nproc_per_node ";
  cmd += (string)argv[2];
  cmd += " allreduce.py --master_addr ";
  cmd += (string)argv[1];

  MPI_Barrier(MPI_COMM_WORLD);

  int result = system(cmd.c_str());

  if (result < 0) {
    cout << "Failed to run Rank " << rank << endl;
    MPI_Abort(MPI_COMM_WORLD,1);
    return 0;
  }

  MPI_Finalize();
  return 0;
}
