# -*- coding: utf-8 -*-
"""
converts vcf or plink file to zarr
drops markers with more than 5% missing values
replaces other NA values with column mean

Install dependencies:
```
pip install numpy loguru pandas bed_reader zarr pathlib scikit-allel
```
python convert_to_zarr.py --filename x.vcf --outdir dir/
```
--filename name of vcf or bed file
--outdir output directory where zarr files should be stored
"""
import os
import io
import sys
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from bed_reader import open_bed, sample_file
import allel
import zarr
import pathlib

def main(filename, outdir):

    ## input = zipped vcf file
    if filename.endswith('.vcf.gz'):
        # 1st dim  = variants, 2nd dim = samples
        callset = allel.read_vcf(filename, fields=['calldata/GT', 'samples', 'variants/CHROM', 'variants/POS', 'variants/ID'])
        ## get genotype data as genotype array (2 alleles)
        gt = allel.GenotypeArray(callset['calldata/GT'])
        #logger.info(f"{gt=}")
        ## replace -1 which represents missing values with nans
        gt = np.where(np.equal(gt, -1), np.nan, gt)
        X = (gt[:,:,0] + gt[:,:,1]).T
        logger.info(f"{X=}")
        n, p = X.shape
        ## get rsid and id information
        rsids = np.concatenate([callset['variants/CHROM'].reshape(p,1), callset['variants/POS'].reshape(p,1), callset['variants/ID'].reshape(p,1)], axis=1)
        logger.info(f"{rsids=}")
        ids = callset['samples']
        logger.info(f"{ids=}")

    ## input = bed file
    elif filename.endswith('.bed'):
        # open file
        bed = open_bed(filename)
        # get dimensions
        n = bed.iid_count
        p = bed.sid_count
        # get genotype
        X = bed.read()
        logger.info(f"{X=}")
        # get chrom, bp, rsid
        rsids = np.concatenate([bed.chromosome.reshape(p,1), bed.bp_position.reshape(p,1), bed.sid.reshape(p,1)], axis=1)
        logger.info(f"{rsids=}")
        # get list of samples
        ids = bed.iid
        logger.info(f"{ids=}")

    logger.info(f"{n=}, {p=}")
    ## convert X to pandas dataframe for easier handling of NAs
    X = pd.DataFrame(X)
    #logger.info(f"{X=}")
    index1 = X.columns
    #logger.info(f"{index1=}")
    ## drop markers where more than 5% of individuals have NA values
    X = X.dropna(axis=1, thresh=0.95*n)
    #logger.info(f"{X=}")

    ## replace NA with mean
    # applied Only on variables with NaN values
    for i in X.columns[X.isnull().any(axis=0)]:     
        X[i].fillna(X[i].mean(),inplace=True)
    #X=X.fillna(X.mean())
    logger.info(f"{X=}")

    ## standardize
    logger.info("Scaling...")
    Xnorm = (X-X.mean())/X.std()
    #logger.info(f"{Xnorm=}")

    ## drop columns with no variation which is set to NA by standardizing
    Xnorm = Xnorm.dropna(axis=1)
    logger.info(f"{Xnorm=}")

    ## get index and drop markers from rsids
    index2 = index1.difference(X.columns)
    if len(index2) > 0:
        logger.info(f"remove columns: {index2=}")
    #logger.info(f"{rsids=}")
    rsids = np.delete(rsids, index2, axis=0)
    rsids = pd.DataFrame(rsids, columns=['CHROM', 'POS', 'ID'])
    #logger.info(f"{rsids=}")

    # make sure output directory exists 
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    ## save rsids, pos, chr in order of the markers occuring
    rsids.to_csv(outdir+'/rsids.csv', index=None)
    ## save ids in the same order as X
    ids = pd.DataFrame(ids, columns=['ID'])
    ids.to_csv(outdir+'/ids.csv', index=None)

    # save genotype in zarr format
    n, p = X.shape
    z = zarr.array(Xnorm, chunks=(None,1000))
    logger.info(f"{z.info=}")
    logger.info(f"saved {n=} individuals, {p=} markers")
    zarr.save(outdir+'/genotype.zarr', z)

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting vcf or bed file to plink.')
    parser.add_argument('--filename', type=str, help='input file - can either be vcf.gz or bed', required=True)
    parser.add_argument('--outdir', type=str, help='path to output directory', required=True)
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
        filename=args.filename,
        outdir=args.outdir,
        ) 
    logger.info("Done.")
