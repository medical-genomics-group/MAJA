# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy zarr
```
generate MC:
python gen_data.py --n 10000 --p 4000 4000 2000 --p0 3250 3750 1800 --q 3 --dir MC
```
n = individuals
p = markers split per group
p0 = markers set to 0 per group
q = number of traits
dir = directory where MC is stored (mkdir beforehand)
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from loguru import logger


def main(n, groups, q, groups0, dir):

    p = np.sum(groups)
    p0 = np.sum(groups0)
    assert p0 < p
    G = len(groups)
    assert G == len(groups0)
    logger.info(f"Problem has dimensions: {n=} individuals, {p=} markers with {p0=} effects set to 0, {q=} traits and {G=} groups.")

    # random generator
    rng = np.random.default_rng()
    # generate genotpye matrices
    Xnorm = rng.normal(0, 1, size=(n, p))

    ### generate beta
    # covariances
    if q==1:
        var = np.array([0.2, 0.03, 0.09, 0.15, 0.1])

    if q==2:
        var = np.array([
            [[0.03, -0.5*np.sqrt(0.09)*np.sqrt(0.03)],[-0.5*np.sqrt(0.09)*np.sqrt(0.03), 0.09]],
            [[0.15, -0.5*np.sqrt(0.15)*np.sqrt(0.2)],[-0.5*np.sqrt(0.15)*np.sqrt(0.2), 0.2]],
            [[0.1, 0.5*np.sqrt(0.1)*np.sqrt(0.15)],[0.5*np.sqrt(0.1)*np.sqrt(0.15), 0.15]],
            ])
    if q==3:
        var = np.array([
            [[0.03, -0.5*np.sqrt(0.09)*np.sqrt(0.03), 0],[-0.5*np.sqrt(0.09)*np.sqrt(0.03), 0.15, 0], [0., 0., 0.05]],
            [[0.15, -0.5*np.sqrt(0.15)*np.sqrt(0.2), 0],[-0.5*np.sqrt(0.15)*np.sqrt(0.2), 0.2, 0], [0., 0., 0.1]],
            [[0.1, 0.5*np.sqrt(0.1)*np.sqrt(0.15), 0],[0.5*np.sqrt(0.1)*np.sqrt(0.15), 0.15, 0.5*np.sqrt(0.15)*np.sqrt(0.05)], [0, 0.5*np.sqrt(0.15)*np.sqrt(0.05), 0.05]],
            ])

    for g in range(G):
        # pick random beta variance (out of the 3 ones given above)
        r = np.random.randint(0,5) if q==1 else np.random.randint(0,3)
        logger.info(f"{g=}, {r=}, {var[r]=}")
        # calculate number of non-zero effects
        p1 = (groups[g]-groups0[g])
        # generate beta
        if p1 == 0:
            beta = np.zeros((groups0[g], q))
        else:
            beta = np.concatenate([rng.normal(0, np.sqrt(var[r]/p1), p1).reshape(p1,1), np.zeros((groups0[g],q))]) if q==1 else np.concatenate([rng.multivariate_normal(np.zeros(q), var[r]/p1, p1), np.zeros((groups0[g], q))])
        V = np.matmul(beta.T, beta)
        if g == 0:
            true_beta = beta.copy()
            true_V = V.copy()
        else:
            true_beta = np.concatenate([true_beta, beta])
            true_V = np.concatenate([true_V, V])

    #logger.info(f"{true_beta.shape=}")        
    logger.info(f"{true_V=}")
    logger.info(f"{np.var(Xnorm@true_beta)=}")
    g = Xnorm@true_beta

    # generate epsilon
    sigma = np.sqrt(1-np.var(g)) if q==1 else np.eye(q) - np.cov(g, rowvar=False)
    epsilon = rng.normal(0, sigma, n) if q==1 else rng.multivariate_normal(np.zeros(q), np.diag(np.diag(sigma)), n)
    logger.info(f"calculated std for epsilon: {sigma=}")
    #epsilon = rng.multivariate_normal(np.zeros(q), sigma, n)
    true_sigma2 = np.matmul(epsilon.T, epsilon)/n
    if q==1:
        true_sigma2 = true_sigma2.reshape(1,1)
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
    np.savetxt(dir+'/genotype.txt', Xnorm)


#########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Data simulation.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--q', type=int, default=2, help='number of traits; code set up for 2 or 3')
    parser.add_argument('--p', nargs='+', type=int, help='number of markers in each group, sums up to total number of markers', required = True)
    parser.add_argument('--p0', nargs='+', type=int, help='number of markers set to 0 in each group', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the MC is stored', required = True)
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
        q = args.q, # number of traits
        groups0 = np.array(args.p0), # number of markers set to 0
        dir = args.dir # path to results directory
    ) 
    logger.info("Done.")