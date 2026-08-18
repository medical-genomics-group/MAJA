# -*- coding: utf-8 -*-
"""
Install dependencies:
```
pip install numpy loguru zarr dask pathlib
```
python get_mean_std_rmrsids.py 
--indir indir/ --outdir outputdir/ --p 1000 1000 1000 --pheno pheno
```
--indir directory where zarr files are stored without chr (required)
        File structure currently is indir/chrX/genotype.zarr
--outdir output directory (required)
--nchr  total number of chromsomes (default=22)
--k number of genetic components (k=2,3,4); default=4
"""
import os
import sys
import argparse
import numpy as np
from loguru import logger
import dask.array as da
import zarr

def main(indir, outdir, nchr, k, rmid, name):

    ### storage
    mean1 = []
    mean2 = []
    std1 = []
    std2 = []
    rmrsids = []
    p = np.zeros(nchr)

    ## open rmid file
    if rmid is not None:
        lines1 = list(np.loadtxt(rmid[0], delimiter=",").astype('int'))
        lines2 = list(np.loadtxt(rmid[1], delimiter=",").astype('int'))

    ## loop through chromosomes
    for i in range(1,nchr+1):
        ## open zarr files
        z = zarr.open(indir+"/genotype.zarr/", mode='r')
        xdata = da.from_zarr(z)
        ## first parent
        X1 = xdata[:,1::k].compute()
        X1 = X1.astype('float')
        X1 = np.where(np.equal(X1,9), np.nan, X1)
        n1, p[i-1] = X1.shape
        logger.info(f"{X1.shape=}")
        ## second parent
        X2 = xdata[:,2::k].compute()
        X2 = X2.astype('float')
        X2 = np.where(np.equal(X2,9), np.nan, X2)
        n2, _ = X2.shape
        logger.info(f"{X2.shape=}")
    
        ## delete rows where phenotype is na
        if rmid is not None:
            lines1 = [l for l in lines1 if l < n1]
            X1 = np.delete(X1, lines1, axis=0)
            n1 = len(X1)
            logger.info(f"{n1=}")
            lines2 = [l for l in lines2 if l < n2]
            X2 = np.delete(X2, lines2, axis=0)
            n2 = len(X2)
        
        # calculate mean and std and append
        s1 = np.nanstd(X1, ddof=1, axis=0)
        std1 = np.append(std1, s1)
        m1 = np.nanmean(X1, axis=0)
        mean1 = np.append(mean1, m1)
        logger.info(f"{len(s1)=}, {len(m1)=}")
        s2 = np.nanstd(X2, ddof=1, axis=0)
        std2 = np.append(std2, s2)
        m2 = np.nanmean(X2, axis=0)
        mean2 = np.append(mean2, m2)
        logger.info(f"{len(s2)=}, {len(m2)=}")

        ## drop indices with std==0 in both sets
        id = np.concatenate([np.where(s1==0)[0],np.where(s2==0)[0]])
        logger.info(f"{id=}")
        if len(id) > 0:
            id = np.unique(id)
            logger.info(f"{id=}")
            id += (np.sum(p[:i-1]))
            logger.info(f"{id=}")
            rmrsids = np.append(rmrsids, id)

    logger.info(f"{p=}")
    logger.info(f"{np.sum(p)=}")
    # save mean and std as zarr
    ## parent 1
    zm = zarr.array(mean1)
    logger.info(f"{zm=}")
    zarr.save(outdir+"/mean1.zarr", zm)
    zs = zarr.array(std1)
    logger.info(f"{zs=}")
    zarr.save(outdir+"/std1.zarr", zs)
    # parent2
    zm = zarr.array(mean2)
    logger.info(f"{zm=}")
    zarr.save(outdir+"/mean2.zarr", zm)
    zs = zarr.array(std2)
    logger.info(f"{zs=}")
    zarr.save(outdir+"/std2.zarr", zs)
    # save rmrsids and number of markers
    np.savetxt(outdir+"/rmrsids_ordered_"+name+".txt", rmrsids)
    np.savetxt(outdir+"/p_per_chr.txt", p)

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Getting means, std.dev. and rmrsids for all chromosomes.')
    parser.add_argument('--indir', type=str, help='path to input directory with zarr files (without chr)', required = True)
    parser.add_argument('--outdir', type=str, help='output directory', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family member incl. POO (2,3 or 4; default=4)')
    parser.add_argument('--nchr', type=int, default=22, help="number of chromosomes (default=22)")
    parser.add_argument('--rmid', type=str, nargs="+", help='list of ids to delete - 2 files (default is None)')
    parser.add_argument('--name', type=str, help='name of phenotype')
    args = parser.parse_args()
    logger.info(args)

    logger.remove()
    logger.add(
        sys.stderr,
        backtrace=True,
        diagnose=True,
        colorize=True,
        level=str("debug").upper(),
    )
    np.set_printoptions(precision=6, suppress=True)
    main(
        indir = args.indir,
        outdir = args.outdir,
        nchr=args.nchr,
        k=args.k,
        rmid=args.rmid,
        name=args.name,
        )
    logger.info("Done.")
