#!/bin/bash
#
#SBATCH --job-name=multi-mpi
#SBATCH --output=multi-mpi-%N-%J.out
#SBATCH --time=03:00:00
#SBATCH -e mpi-err.out
#SBATCH --ntasks=8
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=512G #pool for all cores

outdir=results
mkdir -p $outdir
indir=MC

module load python/3.11.1
module load openmpi/4.1.4
source python3111-openmpi414/bin/activate

srun python -m mpi4py multi_mpi_GS.py \
--n 18264 \
--p 51148 \
--g 22767 10165 18216\
--q 2 \
--itc 1 --diagnostics=True \
--iters 5 --burnin 1 \
--dir ${outdir} \
--y ${indir}/phenotype.txt \
--x ${indir}/genotype.zarr

