#!/bin/bash

model=alexnet
bs=256

printf "config [ nodes gpus ]\n"
read nodes gpu

total_gpu=`expr $nodes \* $gpu`

printf "parallel\n"
printf "0: hybrid, 1: dp(inter-batch), 2: dp(intra-batch), 3: mp\n"
parallel=("hybrid" "dp" "dp_intra" "mp")
read idx
para=${parallel[$idx]}

backend=gloo
no_input_pipelining=""
if [ $idx = "0" ];then
    backend=gloo
elif [ $idx = "1" -o $idx = "2" ];then
    no_input_pipelining="--no_input_pipelining"
fi

printf "directory\n"
read machines

file="${model}_${machines}_${para}"

if [ $idx = "2" ]; then
    bs=`expr $bs / $total_gpu`
    para="dp"
fi

echo "#include <iostream>
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

  string cmd = \"./python.sh -m launch --nnodes \";
/*
  string cmd = \"nvprof --profile-child-processes -o ${file}_\";
  cmd += stm0.str();
  cmd += \"_%p.nvp python3 -m launch --nnodes \";
*/

  cmd += stm1.str();
  cmd += \" --node_rank \";
  cmd += stm0.str();
  cmd += \" --nproc_per_node ${gpu} main_with_runtime.py -s --distributed_backend ${backend} -m models.${model}.gpus=${machines} --epochs 6 -b ${bs} --config_path models/${model}/gpus=${machines}/${para}_conf.json ${no_input_pipelining} --num_ranks_in_server ${gpu} --master_addr \";
  cmd += (string)argv[1];

  cout << cmd << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  gettimeofday(&t1, NULL);
  int result = system(cmd.c_str());
  gettimeofday(&t2, NULL);

  if (result < 0) {
    cout << \"Failed to run Rank \" << rank << endl;
    MPI_Abort(MPI_COMM_WORLD,1);
    return 0;
  }

  double time = get_elapsed_time(&t1, &t2);
  printf(\"Rank %d : %.3lf seconds\n\", rank, time/1000000.00);

  MPI_Finalize();
  return 0;
}" > "cpp/${file}.cpp"

mpicxx -o cpp/$file "cpp/${file}.cpp"

echo "#!/bin/sh
#$ -cwd
#$ -l f_node=${nodes}
#$ -l h_rt=00:30:00
#$ -N ${file}
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn nccl

MASTER_ADDR=\$(hostname -i | awk '{print \$1}')

echo mpirun -x LD_LIBRARY_PATH -x PATH -n ${nodes} -npernode 1 --bind-to none "./cpp/${file}" \$MASTER_ADDR
mpirun -x LD_LIBRARY_PATH -x PATH -n ${nodes} -npernode 1 --bind-to none "./cpp/${file}" \$MASTER_ADDR" > job.sh

chmod u+x job.sh

printf "option\n"
read option

if [ "$option" != "n" ]; then
    qsub ${option} job.sh
fi
