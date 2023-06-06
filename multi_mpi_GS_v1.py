# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy tqdm mpi4py welford matplotlib pandas zarr dask
```
run with 4 processes (given by -n); 
either give --g or --gindex;

mpiexec -n 4 python -m mpi4py multi_mpi_180423.py --n 10000 --p 10003 --q 3
--iters 500 --burnin 100 
--x MC/ --y MC/phenotype.txt --dir results
--diagnostics True
--g 4000 4000 2000 
--gindex group_index.txt
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
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import zarr


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
    beta_2 = np.linalg.multi_dot([beta.T, beta])
    ## all zero groups
    if (np.sum(np.diag(beta_2)) <= 0.001) or (Z==0):
    #if np.all(beta_2 < 10e-09):
        V = np.zeros((q,q))
        Vinv = 10e+09*np.eye(q)
        L = np.eye(q)
        D = np.ones(q)
    ## non-zero groups
    else:
        beta_2 *= Z
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


def main(n, p, q, iters, burnin, groups, itc, xfiles, yfile, dir, gindex, diagnostics):

    if diagnostics:
        logger.info(f"Running with diagnostics: Saving traces of sigma, V and Z.")

    ## groups
    if gindex:
        group_idx = np.loadtxt(gindex, dtype='int8')
        _, groups = np.unique(group_idx, return_counts=True)
        G = int(np.amax(group_idx)+1)
        assert p == group_idx.shape[0]
    elif len(groups) > 0:
        G = len(groups)
        group_idx = np.repeat(np.arange(G), groups)
        group_idx = group_idx.astype(int)
        assert p == np.sum(groups)
    else:
        logger.info("Neither g nor gindex has been defined. One of them is needed for processing data.")
    logger.info(f"Problem has dimensions {n=}, {p=}, {q=}, {G=}.")


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

    beta = None
    tracker = None
    Z_sum = np.ones(G, dtype='i')*(p//G)
    Z = np.zeros(G, dtype='i')
    epsilon = np.empty((n,q), dtype=np.float64)
    sigma_inv = np.empty((q,q), dtype=np.float64)
    V_inv = np.empty((G,q,q), dtype=np.float64)
    pi_ratio = np.ones(G)

    # open genotype file via lazy loading
    # xdata = da.from_npy_stack(xfile)
    for i in range(len(xfiles)):
        z = zarr.open(xfiles[i], mode='r')
        if i == 0:
            xdata = da.from_zarr(z)
        else:
            xdata = da.append(xdata, z, axis=1)    
    logger.info(f"{xdata=}")
    # add columns of 0 for even split
    if p_split*worldSize-p > 0:
        z = p_split*worldSize-p
        az = np.zeros((n, p_split*worldSize-p))
        logger.info(f"Added {z} columns of zeros to x.")
        xdata = da.concatenate([xdata, az], axis=1)
        logger.info(f"{group_idx.shape=}")
        group_idx = np.append(group_idx, np.ones(z)*G)
        group_idx = group_idx.astype(int)
        logger.info(f"{group_idx.shape=}")
        logger.info(f"{G=}, {group_idx=}")
    # actually load only data that is needed in each process
    x = xdata[:,rank*p_split:(rank+1)*p_split].compute()
    logger.info(f"{rank=}, {x.shape=}")
    logger.info(f"{rank=}, {x=}")

    if rank == 0:
        ## open phenotype file
        epsilon = np.loadtxt(yfile)

        # initalize parameters
        init = {
            "beta": np.zeros((p_split*worldSize, q)),
            "V": np.repeat([0.5/G*np.eye(q)], G, axis=0),
            "sigma": 0.5*np.array(np.eye(q)),
            "pi": np.repeat(np.array([[0.9, 0.1]]), G, axis=0), #[0.5,0.5]
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
        tracker = np.zeros(p_split*worldSize)
        logger.info(f"{tracker=}")
        
        # generate storage using the Welford package
        w_beta = welford.Welford()
        w_V = welford.Welford()
        w_sigma = welford.Welford()
        w_tracker = welford.Welford()
        # storage
        if diagnostics:
            trace_V = np.zeros((iters,G,q,q))
            trace_sigma = np.zeros((iters,q,q))
            trace_Z = np.zeros((iters, G))

    # initializing "split" data
    tracker_split = np.zeros(p_split)
    beta_split = np.zeros(p_split*q)
    comm.Scatterv([beta, p_split*q, MPI.DOUBLE], beta_split, root=0)
    beta_split = beta_split.reshape(p_split, q)

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
                temp = V_inv[g] if Z_sum[g]==0 else V_inv[g]*Z_sum[g]
                tracker_split[j], beta_split[j] = sample_mvn(
                    q,
                    n,
                    prev_beta.reshape(-1,q),
                    xe, 
                    sigma_inv, 
                    temp, #V_inv[g]*Z_sum[g],
                    pi_ratio[g],
                    rng
                    )
                # calculate difference in epsilon
                diff += blas.dgemm(1, a=x[:,j:j+1], b=(prev_beta - beta_split[j:j+1]))
                # udpate number of non-zero betas
                Z[g] += tracker_split[j]
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
        comm.Gatherv(sendbuf=tracker_split, recvbuf=tracker, root = 0)        

        if rank == 0:
            beta = beta.reshape((p_split*worldSize, q))
            logger.info(f"{Z_sum=}")
            #logger.info(f"{tracker=}")

            for g in range(G):
                # update pi for each group
                if Z_sum[g] == 0:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g]-1, 1))
                elif Z_sum[g] == groups[g]:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g]+1, Z_sum[g]-1))
                else:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g], Z_sum[g]))
                
                #update V
                V[g], V_inv[g], L[g], D[g] = sample_V(
                    beta[group_idx==g],
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
            if diagnostics:
                trace_V[it] = V
                trace_sigma[it] = sigma
                trace_Z[it] = Z_sum
                if (it%500==0):
                    dfm = pd.DataFrame(beta[:p])
                    dfm.to_csv(dir+'/beta_'+str(it)+'.csv.zip', index=False, compression='zip', sep=',')
                    np.savetxt(dir+'/V_'+str(it)+'.txt', V.reshape(G*q,q))
                    np.savetxt(dir+'/sigma2_'+str(it)+'.txt', sigma)
                    np.savetxt(dir+'/Z_'+str(it)+'.txt', trace_Z[it])
                    np.savetxt(dir+'/prob_'+str(it)+'.txt', tracker[:p])
                    np.savetxt(dir+'/L_'+str(it)+'.txt', L.reshape(G*q,q))
                    np.savetxt(dir+'/epsilon_'+str(it)+'.txt', epsilon)
                    np.savetxt(dir+'/trace_sigma.txt', trace_sigma.diagonal(0,1,2))
                    np.savetxt(dir+'/trace_Z.txt', trace_Z)
                    for g in range(G):
                        np.savetxt(dir+'/trace_V'+str(g)+'.txt', trace_V[:,g].diagonal(0,1,2))

            if it >= burnin:
                w_beta.add(beta[:p])
                w_sigma.add(sigma)
                w_V.add(V.reshape(G*q,q))
                w_tracker.add(tracker[:p])
                

    ## iterations finished
    ## print out numbers
    if rank == 0:
        mean_beta = np.array(w_beta.mean)
        var_beta = np.array(w_beta.var_s)
        mean_V = np.array(w_V.mean)
        var_V = np.array(w_V.var_s)
        mean_sigma = np.array(w_sigma.mean)
        var_sigma = np.array(w_sigma.var_s)
        mean_prob = np.array(w_tracker.mean)
        var_prob = np.array(w_tracker.var_s)
        ### save
        dfm = pd.DataFrame(mean_beta)
        dfm.to_csv(dir+'/mean_beta.csv.zip', index=False, compression='zip', sep=',')
        dfv = pd.DataFrame(var_beta)
        dfv.to_csv(dir+'/var_beta.csv.zip', index=False, compression='zip', sep=',')
        #np.savetxt(dir+'/mean_beta.txt', mean_beta)
        #np.savetxt(dir+'/var_beta.txt', var_beta)
        np.savetxt(dir+'/mean_V.txt', mean_V)
        np.savetxt(dir+'/var_V.txt', var_V)
        np.savetxt(dir+'/mean_sigma.txt', mean_sigma)
        np.savetxt(dir+'/var_sigma.txt', var_sigma)
        np.savetxt(dir+'/mean_prob.txt', mean_prob)
        np.savetxt(dir+'/var_prob.txt', var_prob)

        if diagnostics:
            np.savetxt(dir+'/trace_sigma.txt', trace_sigma.diagonal(0,1,2))
            np.savetxt(dir+'/trace_Z.txt', trace_Z)
            for g in range(G):
                np.savetxt(dir+'/trace_V'+str(g)+'.txt', trace_V[:,g].diagonal(0,1,2))
        
            # Plotting sigma results
            logger.info("Plotting sigma results.")
            t = np.arange(iters)
            figS, axS = plt.subplots()
            for i in range(q):
                for j in range(q):
                    axS.plot(t, trace_sigma[:,i,j])
            axS.set(ylabel='sigma', xlabel='iterations')
            axS.get_figure().savefig(dir+'/trace_sigma.png')

            # Plotting V results
            logger.info("Plotting V results.")
            for g in range(G):
                figV, axV = plt.subplots()
                for i in range(q):
                    for j in range(q):
                        axV.plot(t, trace_V[:,g,i,j])
                axV.set(ylabel='V', xlabel='iterations')
                axV.get_figure().savefig(dir+'/trace_V'+str(g)+'.png')

            
            # Plotting Z
            logger.info("Plotting Z results.")
            figZ, axZ = plt.subplots()
            for g in range(G):
                axZ.plot(t, trace_Z[:,g])
            axZ.set(ylabel='Z', xlabel='iterations')
            axZ.get_figure().savefig(dir+'/trace_Z.png')


##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitrait Gibbs sampler.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--q', type=int, help='number of traits', required = True)
    parser.add_argument('--g', nargs='+', type=int, help='number of markers in each group')
    parser.add_argument('--iters', type=int, default=10000, help='number of iterations (default = 10000)')
    parser.add_argument('--burnin', type=int, default=1000, help='number of iterations in burnin (default = 1000)')
    parser.add_argument('--itc', type=int, default=2, help='counter for updating epsilon (default=2)')
    parser.add_argument('--x', type=str, nargs='+', help='directory in which genotype matrix (csv.zip) is stored', required = True)
    parser.add_argument('--y', type=str, help='phenotype matrix filename in txt file format', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--gindex', type=str, help='file with group index information')
    parser.add_argument('--diagnostics', type=bool, default=False, help='store traces for diagnostics; False by default')
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
        xfiles = args.x, # genotype file
        yfile = args.y, # phenotype file
        dir = args.dir, # path to results directory
        gindex = args.gindex, # path to directory with true values
        diagnostics = args.diagnostics # boolean for diagnostics
        ) 
    logger.info("Done.")
