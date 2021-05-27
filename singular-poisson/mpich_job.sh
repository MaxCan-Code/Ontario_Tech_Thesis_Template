#!/bin/bash
#SBATCH --job-name=mpich-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=45G
#SBATCH --time=0-0:15
# https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling/en#Whole_nodes

module load nixpkgs gmpich apptainer

cd /project/def-glewis/thesis/singular-poisson

srun --profile=task apptainer run -B /home -B /project -B /scratch \
    -H $(pwd)/newhome \
    -W $SLURM_TMPDIR \
    --writable-tmpfs \
    $(pwd)/FEniCS-pandas.sif \
    bash -c \
    'python3 -m mpi4py.bench helloworld && \
     set +o pipefail && \
     python3 /project/def-glewis/thesis/singular-poisson/run_demo.py'
