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

outdir="results/chr20-22/"
mkdir -p $outdir
indir=/nfs/scistore13/robingrp/human_data/GSM-preprocessed/

module load python/3.11.1
module load openmpi/4.1.4
source python3111-openmpi414/bin/activate

srun python -m mpi4py multi_mpi_GS_v1.py \
--n 18264 \
--p 51148 \
--g 22767 10165 18216\
--q 2 \
--itc 1 --diagnostics=True \
--iters 5 --burnin 1 \
--dir ${outdir} \
--y ${indir}/pheno/age_adjusted-bmi.txt \
--x ${indir}/final-zarr/std_methylation_chr20.zarr ${indir}/final-zarr/std_methylation_chr21.zarr ${indir}/final-zarr/std_methylation_chr22.zarr

