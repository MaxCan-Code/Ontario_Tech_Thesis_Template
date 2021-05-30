#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15	# number of MPI processes
#SBATCH --mem=45G		# memory; default unit is megabytes
#SBATCH --time=30		# time ("minutes", "minutes:seconds", "hours:minutes:seconds")
# ref https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling#Few_cores.2C_single_node

# git clone https://github.com/MaxCan-Code/thesis
# cd thesis && git fetch -a && git reset --hard origin/HEAD
# module load singularity/3.7 mpich/3.3a2
# module load singularity mpich

# run this line if you don't have FEniCS-pandas.sif:
# docker run --privileged -it --rm -v $(pwd):/root quay.io/singularity/singularity:v3.7.3-slim build root/FEniCS-pandas.{sif,def}

# singularity run --writable-tmpfs FEniCS-pandas.sif "python3 -VV"
# srun singularity exec -B ~/projects
# mpirun ~/singularity/builddir/singularity
singularity exec FEniCS-pandas.sif python3 run-hpc.py
