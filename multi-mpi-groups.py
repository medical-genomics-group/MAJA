# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy tqdm mpi4py welford matplotlib
```
run with 4 processes (given by -n):
mpiexec -n 4 python -m mpi4py multi-mpi-groups_11012023.py --n 20000 --p 30000 --q 2 --g 15000 15000 --iters 5000 --burnin 1000 --x MC/genotype.npz --y MC/phenotype.txt --dir results --true_dir MC
"""

import sys
import argparse
import welford
from mpi4py import MPI
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.linalg.blas as blas
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import trange


def sample_mvn(q, n, beta, xe, sigma_inv, V_inv, pi_ratio, rng):
    ## sample betas from multivariate normal distribution
    # calculate omega*
    omega_star_inv = (n-1)*sigma_inv + V_inv
    omega_star = linalg.inv(omega_star_inv)
    # calculate mu*
    mu_star = np.linalg.multi_dot([
            xe + (n-1)*beta,
            sigma_inv, 
            omega_star]).flatten("F") #flatten mean to draw from MVN
    # calculate exclusion probability
    # calc term in exponential (cut of at 100 so that exp does not overflow)
    f = np.minimum(
        np.linalg.multi_dot([mu_star.T, omega_star_inv, mu_star])/2, 
        100) # mu_star is transposed in calculation here due to flattening

    tau = calc_tau(f, pi_ratio, omega_star, V_inv)
    if tau > rng.uniform(0,1):
        tracker = 0
        beta = np.zeros((1,q))
    else:
        tracker = 1
        beta = rng.multivariate_normal(
            mu_star,
            omega_star,
            method="cholesky", 
            check_valid="ignore"
        )

    return tracker, beta


def calc_tau(f, pi_ratio, omega_star, V_inv):
    return (pi_ratio / (
        pi_ratio
        + np.linalg.det(V_inv)**(1/2)
        * np.linalg.det(omega_star)**(1/2)
        * np.exp(f) )
    )


def sample_V(beta, L, D, q, Z, a, b, s):
    # sample covariances according to
    # https://doi.org/10.1198/jcgs.2009.08095
    beta_2 = Z*np.linalg.multi_dot([beta.T, beta])
    ## all zero groups
    if np.all(beta_2 < 10e-15):
        V = np.zeros((q,q))
        Vinv = 10e+15*np.eye(q)
        L = np.eye(q)
        D = np.ones(q)
    ## non-zero groups
    else:
        ww = np.linalg.multi_dot([L, beta_2, L.T])

        for i in range(q):
            #sample elements of D
            D[i] = stats.invgamma.rvs(a= a/2 + Z, 
                scale=a*b/2 + ww[i, i]
                )
            #sample elements of L
            if i >= 1:
                si = np.linalg.inv((1 / D[i]) * beta_2[0:i, 0:i] + s*np.eye(i))
                mi = - (si / D[i]) @ beta_2[0:i, i]
                L[i, :i] = np.random.multivariate_normal(mi.flatten(), si).reshape((1, i))

        Vinv = np.linalg.multi_dot([L.T, np.diag(1/D), L])
        V = linalg.inv(Vinv)
    
    return V, Vinv, L, D


def main(n, p, q, iters, burnin, groups, itc, xfile, yfile, dir, true_dir):

    ## true values
    if true_dir == None:
        true_V = np.loadtxt('true_V.txt')
        true_sigma = np.loadtxt('true_sigma.txt')
    else:
        true_V = np.loadtxt(true_dir+'/true_V.txt')
        true_sigma = np.loadtxt(true_dir+'/true_sigma.txt')

    ## groups
    G = len(groups)
    group_idx = np.repeat(np.arange(G), groups)
    assert p == np.sum(groups)
    logger.info(f"Problem has dimensions {n=}, {p=}, {q=}, {G=}.")

    true_V = np.split(true_V, G, axis=0)

    # mpi initalization
    comm = MPI.COMM_WORLD
    worldSize = comm.Get_size()
    rank = comm.Get_rank()
    logger.info(f"There are {worldSize} processes running.")
    # size of data blocks
    p_split = int(p/worldSize)
    if p_split * worldSize < p:
        p_split += 1
    logger.info(f"Data is split in {worldSize} * {p_split} columns.")

    # random generator
    rng = np.random.default_rng()

    xdata = None
    beta = None
    Z_sum = np.ones(G, dtype='i')*p
    Z = np.zeros(G, dtype='i')
    epsilon = np.empty((n,q), dtype=np.float64)
    sigma_inv = np.empty((q,q), dtype=np.float64)
    V_inv = np.empty((G,q,q), dtype=np.float64)
    pi_ratio = np.ones(G)

    if rank == 0:
        # storage
        trace_V = np.zeros((iters,G,q,q))
        trace_sigma = np.zeros((iters,q,q))
        ## open phenotype file
        epsilon = np.loadtxt(yfile)
        # open genotype file
        xdata = np.load(xfile)
        xdata = xdata.f.arr_0
        if p_split*worldSize-p > 0:
            az = np.zeros((n, p_split*worldSize-p))
            logger.info(f"Added {p_split*worldSize-p} columns of zeros to x.")
            xdata = np.concatenate([xdata, az], axis=1)
        xdata = xdata.flatten('F')

        # initalize parameters
        init = {
            "beta": np.zeros((p_split*worldSize, q)),
            "V": np.repeat([0.5*np.eye(q)], G, axis=0),
            "sigma": 0.5*np.array(np.eye(q)),
            "pi": np.repeat(np.array([[0.5, 0.5]]), G, axis=0),
            "D": np.array(G*[np.ones(q)]),
            "L": np.array(G*[np.eye(q)]),
            "De": np.ones(q),
            "Le": np.eye(q),
            "mu": 0,
        }
        hypers = {
            "ae": 2,
            "be": 0.1,
            "se": 0.0001,
            "av": 2,
            "bv": 0.1,
            "sv": 0.0001
        }

        beta = init["beta"]
        beta = beta.flatten() #vectorize for sending data
        V = np.array(init["V"])
        for g in range(G):
            V_inv[g] = linalg.inv(V[g])
        sigma = np.array(init["sigma"])
        sigma_inv = linalg.inv(sigma)
        pi = init["pi"]
        mu = init["mu"]
        L = init["L"]
        D = init["D"]
        Le = init["Le"]
        De = init["De"]
        logger.info(f"initialize V as {V=}")
        logger.info(f"initialize sigma as {sigma=}")
        
        # generate storage using the Welford package
        w_beta = welford.Welford()
        w_V = welford.Welford()
        w_sigma = welford.Welford()
        trace_Z = np.zeros((iters, G))
    
    # send data
    # initialize recvbuf on all processes
    x = np.zeros(p_split*n)
    comm.Scatterv(sendbuf=[xdata, p_split*n, MPI.DOUBLE], recvbuf=x, root=0)
    x = x.reshape((n,p_split), order='F')

    beta_split = np.zeros(p_split*q)
    comm.Scatterv([beta, p_split*q, MPI.DOUBLE], beta_split, root=0)
    beta_split = beta_split.reshape(p_split, q)

    if rank == 0:
        del xdata

    # Loop through iterations
    logger.info(f"Running Gibbs with {iters=} and {burnin=}")
    for it in trange(iters, desc="Main loop"):

        comm.Barrier()
        if rank==0:            
            # ratio of probability(beta=0) and probability(beta != 0)
            pi_ratio = pi[:,0]/pi[:,1]

            # sample intercept term
            epsilon += mu
            mu = np.mean(epsilon, axis=0) if it == 0 else rng.multivariate_normal(np.mean(epsilon, axis=0), sigma/n)
            epsilon -= mu
            # flatten matrices to vectors for sending
            epsilon = epsilon.flatten()
            sigma_inv = sigma_inv.flatten()
            V_inv = V_inv.flatten()
        
        #send relevant information to all processes
        comm.Bcast([epsilon, MPI.DOUBLE], root=0)
        comm.Bcast([sigma_inv, MPI.DOUBLE], root=0)
        comm.Bcast([V_inv, MPI.DOUBLE], root=0)
        comm.Bcast([pi_ratio, MPI.DOUBLE], root=0)
        comm.Bcast([Z_sum, MPI.INT], root=0)
        # reshape flattened vectors
        if rank== 0:
            epsilon = epsilon.reshape((n,q))
            sigma_inv = sigma_inv.reshape((q,q))
            V_inv = V_inv.reshape((G,q,q))

        # loop trough all markers randomly
        rj = np.arange(0, p_split)
        rng.shuffle(rj)
        ## containers for differences in epsilon
        diff = np.zeros((n,q))
        diff_sum = np.zeros((n,q))
        ## set number of non-zero markers to 0 before each iteration
        Z = np.zeros(G, dtype='i')
        #keep track of number of processed markers
        counter = 0 
        for j in rj:
            # check if marker is outside of range
            gj = j + p_split*rank
            if gj >= p:
                beta_split[j] = np.zeros((1,q))
                counter += 1
            else:
                #get group index
                g = group_idx[gj]
                # calculate x.T@epsilon
                xe = blas.dgemm(1, a=x[:,j:j+1], b=epsilon, trans_a=1)
                prev_beta = beta_split[j:j+1].copy()
                # sample beta
                tracker, beta_split[j] = sample_mvn(
                    q,
                    n,
                    prev_beta.reshape(-1,q),
                    xe, 
                    sigma_inv, 
                    V_inv[g]*Z_sum[g],
                    pi_ratio[g],
                    rng
                    )
                # calculate difference in epsilon
                diff += blas.dgemm(1, a=x[:,j:j+1], b=(prev_beta - beta_split[j:j+1]))
                # udpate number of non-zero betas
                Z[g] += tracker
                counter += 1

            #receive and sum up diff after each process processed two markers
            if counter%itc==0 or counter == p_split:
                comm.Barrier()
                comm.Reduce(diff, diff_sum, MPI.SUM, root=0)
                                
                if rank==0:
                    epsilon = epsilon + diff_sum
                    diff_sum = np.zeros((n,q))
                comm.Bcast(epsilon, root=0)
                diff = np.zeros((n,q))
        
        comm.Barrier()
        # sum up number of non-zero effects
        comm.Reduce(Z, Z_sum, MPI.SUM, root=0)
        # pull together betas
        comm.Gatherv(sendbuf=beta_split, recvbuf=(beta, p_split*q), root = 0)        

        if rank == 0:
            beta = beta.reshape((p_split*worldSize, q))
            for g in range(G):
                # update pi for each group
                if Z_sum[g] == 0:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g]-1, 1))
                elif Z_sum[g] == groups[g]:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g]+1, Z_sum[g]-1))
                else:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g], Z_sum[g]))

                start_g = 0 if g == 0 else groups[g]
                end_g = p if g == G-1 else groups[g+1]
                #update V
                V[g], V_inv[g], L[g], D[g] = sample_V(
                    beta[start_g:end_g],
                    L[g], D[g],  
                    q, Z_sum[g],
                    hypers["av"],
                    hypers["bv"],
                    hypers["sv"]
                    )

            # update sigma
            sigma, sigma_inv, Le, De = sample_V(
                epsilon/((n)**(1/2)),
                Le, De,  
                q, n,
                hypers["ae"],
                hypers["be"],
                hypers["se"]
                )

            # store stuff
            trace_V[it] = V
            trace_sigma[it] = sigma
            trace_Z[it] = Z_sum
            if it >= burnin:
                w_beta.add(beta[:p])
                w_sigma.add(sigma)
                w_V.add(V.reshape(G*q,q))
                

    ## iterations finished
    ## print out numbers
    if rank == 0:
        mean_beta = np.array(w_beta.mean)
        var_beta = np.array(w_beta.var_s)
        mean_V = np.array(w_V.mean)
        var_V = np.array(w_V.var_s)
        mean_sigma = np.array(w_sigma.mean)
        var_sigma = np.array(w_sigma.var_s)
        logger.info(f"{np.mean(trace_Z[burnin:], axis=0)=}\n")
        logger.info(f"{true_V=}")
        logger.info(f"{mean_V=}")
        logger.info(f"{var_V=}")
        logger.info(f"{np.matmul(mean_beta.T, mean_beta)=}\n")
        logger.info(f"{true_sigma=}")
        logger.info(f"{mean_sigma=}")
        ### save
        np.savetxt(dir+'/mean_V.txt', mean_V)
        np.savetxt(dir+'/var_V.txt', var_V)
        np.savetxt(dir+'/mean_sigma.txt', mean_sigma)
        np.savetxt(dir+'/var_sigma.txt', var_sigma)
        np.savetxt(dir+'/mean_beta.txt', mean_beta)
        np.savetxt(dir+'/var_beta.txt', var_beta)
        np.savetxt(dir+'/trace_Z.txt', trace_Z)

        # Plotting sigma results
        logger.info("Plotting sigma results.")
        t = np.arange(iters)
        figS, axS = plt.subplots()
        for i in range(q):
            for j in range(q):
                axS.plot(t, trace_sigma[:,i,j])
        axS.set(ylabel='sigma', xlabel='iterations')
        axS.get_figure().savefig(dir+'trace_sigma.png')

        # Plotting V results
        logger.info("Plotting V results.")
        for g in range(G):
            figV, axV = plt.subplots()
            for i in range(q):
                for j in range(q):
                    axV.plot(t, trace_V[:,g,i,j])
            axV.set(ylabel='V', xlabel='iterations')
            axV.get_figure().savefig(dir+'trace_V_'+str(g)+'.png')



##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitrait Gibbs sampler.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--q', type=int, help='number of traits', required = True)
    parser.add_argument('--g', nargs='+', type=int, help='number of markers in each group', required=True)
    parser.add_argument('--iters', type=int, default=10000, help='number of iterations (default = 10000)')
    parser.add_argument('--burnin', type=int, default=1000, help='number of iterations in burnin (default = 1000)')
    parser.add_argument('--itc', type=int, default=2, help='counter for updating epsilon (default=2)')
    parser.add_argument('--x', type=str, help='genotype matrix filename in which file format?', required = True)
    parser.add_argument('--y', type=str, help='phenotype matrix filename in which file format?', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--true_dir', type=str, help='path to directory where the true values are stored')
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
        q = args.q, # number of traits
        iters = args.iters, # number of iterations
        burnin = args.burnin, # number of iterations in burnin period
        groups = np.array(args.g), # number of markers in each group
        itc = args.itc, # counter for updating epsilon (after number of processes times itc markers)
        xfile = args.x, # genotype file
        yfile = args.y, # phenotype file
        dir = args.dir, # path to results directory
        true_dir = args.true_dir # path to directory with true values
        ) 
    logger.info("Done.")
