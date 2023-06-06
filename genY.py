# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy zarr dask
```
generate MC:
python genY.py --n 18264 --p 18216 --p0 17216 --q 2 --dir MC --xfiles /nfs/scistore13/robingrp/human_data/GSM-preprocessed/final-zarr/std_methylation_chr22.zarr --scen 0
```
n = individuals
p = markers split per group
p0 = markers set to 0 per group
q = number of traits
xfiles = path to genotype matrix files (/nfs/scistore13/robingrp/human_data/GSM-preprocessed/final-zarr/std_methylation_chr22.zarr)
dir = path to directory where phenotype should be stored
scen = 0
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
import zarr
import dask.array as da
from loguru import logger


def main(n, groups, groups0, q, scen, xfiles):

    p = np.sum(groups)
    p0 = np.sum(groups0)
    assert p0 < p
    G = len(groups)
    assert G == len(groups0)
    logger.info(f"Problem has dimensions: {n=} individuals, {p=} markers with {p0=} effects set to 0, {q=} traits and {G=} groups.")

    # random generator
    rng = np.random.default_rng()
    # open genotype files via lazy loading
    for i in range(len(xfiles)):
        z = zarr.open(xfiles[i], mode='r')
        if i == 0:
            xdata = da.from_zarr(z)
        else:
            xdata = da.append(xdata,z, axis=1)
    Xnorm = xdata.compute()

    ### generate beta
    # covariances
    if q==2:
        var = np.array([
            [[0.3, 0],[0, 0.5]],
            [[0.3, -0.5*np.sqrt(0.3)*np.sqrt(0.5)],[-0.5*np.sqrt(0.3)*np.sqrt(0.5), 0.5]],
            [[0.3, +0.5*np.sqrt(0.3)*np.sqrt(0.5)],[+0.5*np.sqrt(0.3)*np.sqrt(0.5), 0.5]],
            ])
    if q==3:
        var = np.array([
            [[0.3, 0, 0],[0, 0.5, 0], [0, 0, 0.1]],
            [[0.3, -0.5*np.sqrt(0.3)*np.sqrt(0.5), -0.5*np.sqrt(0.3)*np.sqrt(0.1)],[0.3, -0.5*np.sqrt(0.3)*np.sqrt(0.5), 0.5, -0.5*np.sqrt(0.3)*np.sqrt(0.1)], [-0.5*np.sqrt(0.3)*np.sqrt(0.1), -0.5*np.sqrt(0.5)*np.sqrt(0.1), 0.1]],
            [[0.3, +0.5*np.sqrt(0.3)*np.sqrt(0.5), +0.5*np.sqrt(0.3)*np.sqrt(0.1)],[0.3, -0.5*np.sqrt(0.3)*np.sqrt(0.5), 0.5, -0.5*np.sqrt(0.3)*np.sqrt(0.1)], [+0.5*np.sqrt(0.3)*np.sqrt(0.1), +0.5*np.sqrt(0.5)*np.sqrt(0.1), 0.1]],
            ])

    for g in range(G):
        logger.info(f"{g=}, {scen=}, {var[scen]=}")
        # calculate number of non-zero effects
        p1 = (groups[g]-groups0[g])
        # generate beta
        if p1 == 0:
            beta = np.zeros((groups0[g], q))
        else:
            beta = np.concatenate([rng.multivariate_normal(np.zeros(q), var[scen]/p1, p1), np.zeros((groups0[g], q))])
        rng.shuffle(beta)
        V = np.matmul(beta.T, beta)
        if g == 0:
            true_beta = beta.copy()
            true_V = V.copy()
        else:
            true_beta = np.concatenate([true_beta, beta])
            true_V = np.concatenate([true_V, V])
      
    logger.info(f"{true_V=}")
    logger.info(f"{np.var(Xnorm@true_beta)=}")
    g = Xnorm@true_beta

    # generate epsilon
    sigma = np.eye(q) - np.cov(g, rowvar=False)
    epsilon = rng.multivariate_normal(np.zeros(q), np.diag(np.diag(sigma)), n)
    logger.info(f"calculated std for epsilon: {sigma=}")
    true_sigma2 = np.matmul(epsilon.T, epsilon)/n
    logger.info(f"{true_sigma2=}")

    # generate Y
    Y = g + epsilon
    Ynorm = stats.zscore(Y, axis=0, ddof=1)
 
    # save true values
    np.savetxt(dir+'/true_V.txt', true_V)
    np.savetxt(dir+'/true_sigma2.txt', true_sigma2)
    np.savetxt(dir+'/true_betas.txt', true_beta)
    np.savetxt(dir+'/true_epsilon.txt', epsilon)
    np.savetxt(dir+'/phenotype.txt', Ynorm)


#########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Simulation on top of data.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--q', type=int, default=2, help='number of traits; code set up for 2 or 3')
    parser.add_argument('--p', nargs='+', type=int, help='number of markers in each group, sums up to total number of markers', required = True)
    parser.add_argument('--p0', nargs='+', type=int, help='number of markers set to 0 in each group', required = True)
    parser.add_argument('--scen', type=int, help='different scenarios for variances - 0=no cov., 1=-0.5corr, 2=+0.5corr', required=True)
    parser.add_argument('--x', type=str, nargs='+', help='list of genotype matrix filenames (zarr files)', required = True)
    parser.add_argument('--dir', type=str, help='path to output directory', required = True)
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
    np.set_printoptions(precision=9, suppress=True)
    main(n = args.n, # number of individuals
        groups = np.array(args.p),  # number of markers
        groups0 = np.array(args.p0), # number of markers set to 0
        q = args.q, # number of traits
        scen = args.scen, # variance scenario
        xfiles = args.x, # path to xfiles
        dir = args.dir
    ) 
    logger.info("Done.")