# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy
```
python gen_data.py --n 20000 --p 60000 --p0 56000 --q 2 --dir MC
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from loguru import logger

def gen_data(n, p, q, p0, rng):

    # genotype matrix
    # prob = 0.05
    # x1 = rng.binomial(1, prob, size=(n, p))
    # x2 = rng.binomial(1, prob, size=(n, p))
    # x = x1 + x2
    # methylation data
    x = rng.normal(0., 1., size = (n,p))
    x_norm = stats.zscore(x, axis=0, ddof=1)
    
    # covariance matrices for effect sizes
    V1 = np.array([[0.49, -0.5*np.sqrt(0.49)*np.sqrt(0.3)],[-0.5*np.sqrt(0.49)*np.sqrt(0.3), 0.3]])
    V2 = np.array([[0.4, 0.5*np.sqrt(0.6)*np.sqrt(0.4)],[0.5*np.sqrt(0.4)*np.sqrt(0.6), 0.6]])
    # number of non-zero effects for each group
    p1 = int((p-p0)/2)
    b1 = rng.multivariate_normal(np.zeros(q), V1/p1, p1)
    b2 = rng.multivariate_normal(np.zeros(q), V2/p1, p1)
    b0 = np.zeros((int(p0/2), q))
    true_beta1 = np.concatenate((b1, b0))
    rng.shuffle(true_beta1)
    true_V1 = np.matmul(true_beta1.T, true_beta1)
    true_beta2 = np.concatenate((b2, b0))
    rng.shuffle(true_beta2)
    true_V2 = np.matmul(true_beta2.T, true_beta2)
    true_beta = np.concatenate([true_beta1, true_beta2])
    true_V = np.concatenate([true_V1, true_V2])
    logger.info(f"{true_V=}")
    # epsilon
    xb = np.matmul(x_norm, true_beta)
    s = np.eye(q) - np.cov(xb, rowvar=False)
    epsilon = rng.multivariate_normal(np.zeros(q), np.diag(np.diag(s)), n)
    true_sigma = np.cov(epsilon, rowvar = False)
    logger.info(f"{true_sigma=}")
    # phenotype
    y = xb + epsilon
    y_norm = stats.zscore(y, axis=0, ddof=1)

    return x_norm, y_norm, true_beta, true_V, epsilon, true_sigma


def main(n, p, p0, q, dir):

    logger.info(f"Problem has dimensions {n=}, {p=}, {q=} with {p0=} effects set to 0.")

    # random generator
    rng = np.random.default_rng()
    # generate data
    x, y, true_beta, true_V, true_epsilon, true_sigma = gen_data(n, p, q, p0, rng)
 
    # save true values
    np.savetxt(dir+'/true_V.txt', true_V)
    np.savetxt(dir+'/true_sigma.txt', true_sigma)
    np.savetxt(dir+'/true_betas.txt', true_beta)
    np.savetxt(dir+'/true_epsilon.txt', true_epsilon)
    np.savetxt(dir+'/phenotype.txt', y)
    np.savez_compressed(dir+'/genotype.npz', x)

#########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Data simulation.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--p0', type=int, help='number of markers set to 0', required = True)
    parser.add_argument('--q', type=int, help='number of traits', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the results are stored', required = True)
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
    main(n = args.n, # number of individuals
        p = args.p,  # number of markers
        p0 = args.p0, # number of markers set to 0
        q = args.q, # number of traits
        dir = args.dir # path to results directory
    ) 
    logger.info("Done.")