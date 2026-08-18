#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Install dependencies:
```
pip install numpy loguru scikit-allel pandas
```
python get_rsids.py
--inputfiles path-to-file/chr1.vcf.gz path-to-file/chr2.vcf.gz path-to-file/chr3.vcf.gz etc
--dir path-to-output-directory/
````
"""
import sys
import argparse
import numpy as np
from loguru import logger
import allel
import pandas as pd
import pathlib

def main(inputfiles, dir):

    i = 0
    ##loop over multiple input files
    for l in inputfiles:
        logger.info(f"{l=}")
        #read in vcf file
        # 1st dim  = variants, 2nd dim = samples
        callset = allel.read_vcf(l, fields=['calldata/GT', 'variants/AF', 'variants/CHROM', 'variants/POS', 'variants/ID', 'variants/ALT', 'variants/REF',])
        gt = allel.GenotypeArray(callset['calldata/GT'])
        p = gt.n_variants
        ## get rsid and id information
        if i == 0:
            data = np.concatenate([callset['variants/CHROM'].reshape(p,1), callset['variants/POS'].reshape(p,1), callset['variants/ID'].reshape(p,1)], axis=1)
            data = np.concatenate([data, callset['variants/REF'].reshape(p,1), callset['variants/ALT'][:,0].reshape(p,1)], axis=1)
        else:
            temp = np.concatenate([callset['variants/CHROM'].reshape(p,1), callset['variants/POS'].reshape(p,1), callset['variants/ID'].reshape(p,1)], axis=1)
            temp = np.concatenate([temp, callset['variants/REF'].reshape(p,1), callset['variants/ALT'][:,0].reshape(p,1)], axis=1)
            data = np.append(data, temp, axis=0)
        i+=1

    #save rsids
    rsids = pd.DataFrame(data, columns=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
    logger.info(f"{rsids=}")
    # make sure output directory exists 
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    ## save rsids, pos, chr in order of the markers occuring
    rsids.to_csv(dir+'/rsids.csv', index=None, sep="\t")
       

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Getting rsids.')
    parser.add_argument('--inputfiles', type=str, nargs='+', help='name of input files', required=True)
    parser.add_argument('--dir', type=str, help='Path to output directory.', required=True)
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
    main(inputfiles = args.inputfiles, # inputfile
        dir = args.dir, # path to output directory
        ) 
    logger.info("Done.")

