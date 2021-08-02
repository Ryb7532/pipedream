#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[])
{
  if (argc < 2) {
    cout << "too few arguments\n";
    return 0;
  }

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  string cmd = "python all_to_all.py --backend gloo --master_addr ";
  cmd += (string)argv[1];
  cmd += " --rank ";
  ostringstream stm1, stm2;
  stm1 << rank;
  cmd += stm1.str();
  cmd += " --world_size ";
  stm2 << size;
  cmd += stm2.str();

  cout << "run in Node:" << rank << endl;
  int result = system(cmd.c_str());

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) {
    cout << "done!\n";
  }
  MPI_Finalize();
  return 0;
}
