# -*- coding: utf-8 -*-
# sex-specific MAJA: 
# difference to maja.py: 
# 1. input data - 2 zarr files for each sex needed and zarr files with mean and std. dev.
# 2. residual errors between sexes are assumed to be independent
"""
Install dependencies:
```
pip install numpy loguru scipy tqdm mpi4py welford matplotlib dask zarr pandas
```
run with 4 processes (given by -n); 
either give --g or --gindex;

mpiexec -n 4 python -m mpi4py maja_bee.py 
--n n1 n2   number of individuals, two numbers (separated by space) for father (parent1) and mother (parent2) (required) 
--p         number of markers (required)
--x         2 zarr files with genome data (separated by space) for father (parent1) and mother (parent2) (required)
--mfile     2 zarr files with means for each marker (separated by space) for father (parent1) and mother (parent2)  (required)
--sfile     2 zarr files with standard deviations for each marker (separated by space) for father (parent1) and mother (parent2)  (required)
--y         standardized phenotype file in txt format created in order_phenotype.py (required)
--dir       path to output directory (required)
--iters     total number of iterations (default=5000)
--burnin    number of iterations in the burnin (default=1000)
--rmid      2 files for ids to remove for father (parent1) and mother (parent2) created during preprocessing
--rmrsid    txt file with snps to remove (according to number of column in zarr file) created during preprocessing
--diagnostics   make trace plots for diagnostics (default=False)
--restart   bool to restart sampler on iteration 1000, i.e. after burnin (default=False); requires the correspoding epsilon_1000.txt file as y input; if another iteration, is used for restart, the files need to be changed in the code
--itc       counter for updating epsilon (default=1); can be changed for speed, but might cause divergence issues
--p_split   size of data blocks (default=20000); will help with memory if set correctly 
            must be divisible by chunk size in X zarr array
            must be set up such that (number of processes * p_split) > number of markers, but < (number of markers + p_split)
--g         number of markers within each group; if there is only 1 group, g=p (either give --g or --gindex)
--gindex    index file (.txt) with group index (starting from 0) for each marker in order (either give --g or --gindex)
"""
## turn off multithreading in numpy
import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
import pandas as pd
import zarr


def sample_mvn(q, nv, beta, xe, sigma_inv, V_inv, pi_ratio, rng):
    ## sample betas from multivariate normal distribution
    # calculate omega*
    omega_star_inv = (nv-1)*np.diag(sigma_inv) + V_inv
    omega_star = linalg.inv(omega_star_inv)
    # calculate mu*
    mu_star = np.matmul(omega_star, ((xe + (nv-1)*beta)*sigma_inv).reshape(q))
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
    if np.all(beta_2 < 10e-09):
        V = np.zeros((q,q))
        Vinv = 10e+09*np.eye(q)
        L = np.eye(q)
        D = np.ones(q)
    ## non-zero groups
    else:
        ww = np.linalg.multi_dot([L, beta_2, L.T])

        for i in range(q):
            #sample elements of D
            if Z > 5:
                D[i] = stats.invgamma.rvs(a= a/2 + Z, 
                    scale=a*b/2 + ww[i, i]
                    )
            # for small Z values, use different a and b
            ## changed
            else:
                D[i] = stats.invgamma.rvs(a=1/2+Z,
                                          scale=0.0001/2+ww[i,i]
                                          )

            #sample elements of L
            if i >= 1:
                si = np.linalg.inv((1 / D[i]) * beta_2[0:i, 0:i] + s*np.eye(i))
                mi = - (si / D[i]) @ beta_2[0:i, i]
                ## reset L only if values are reasonable
                ## changed
                if np.random.multivariate_normal(mi.flatten(),si).reshape((1,i)) < 2:
                    L[i, :i] = np.random.multivariate_normal(mi.flatten(), si).reshape((1, i))

        Vinv = np.linalg.multi_dot([L.T, np.diag(1/D), L])
        V = linalg.inv(Vinv)
    
    return V, Vinv, L, D


def main(nv, p, q, iters, burnin, groups, itc, xfiles, 
         yfile, dir, gindex, diagnostics, restart, mfile, sfile, rmid, rmrsid, p_split):

    if diagnostics:
        logger.info(f"Running with diagnostics: Saving traces of sigma, V and Z.")

    ## total number of individuals
    n = np.sum(nv)
    ## maximum n in nv
    nmax = int(np.amax(nv))
    n1 = nv[0]
    n2 = nv[1]

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
    logger.info(f"Problem has dimensions {n=}, {nv=}, {nmax=}, {p=}, {q=}, {G=}.")

    # mpi initalization
    comm = MPI.COMM_WORLD
    worldSize = comm.Get_size()
    rank = comm.Get_rank()
    if rank==0: logger.info(f"There are {worldSize} processes running.")
   
    # open genotype files via lazy loading
    z = zarr.open(xfiles[0], mode='r')
    xdata1 = da.from_zarr(z)
    xdata1 = xdata1.astype('int8')
    z = zarr.open(xfiles[1], mode='r')
    xdata2 = da.from_zarr(z)
    xdata2 = xdata2.astype('int8')
    if rank==0:
        logger.info(f"{xdata1=}")
        logger.info(f"{xdata2=}")
    ## delete rows where phenotype is na for first parent
    if rmid is not None:
        lines1 = np.loadtxt(rmid[0], delimiter=",").astype('int')
        lines1 = lines1[lines1 < n1]
        xdata1 = np.delete(xdata1, lines1, axis=0) 
        n1 -= len(lines1)
        ## delete rows where phenotype is na for second parent
        lines2 = np.loadtxt(rmid[1], delimiter=",").astype('int')
        lines2 = lines2[lines2 < n2]
        xdata2 = np.delete(xdata2, lines2, axis=0) 
        n2 -= len(lines2)
        n = n1 + n2
        nv = np.array([n1,n2])
        if rank==0:
            logger.info(f"trios: new {n1=}, {n2=} and {n=}, {nv=}")

    ## open mean and std file
    zm = zarr.open(mfile[0], mode='r')
    mean1 = da.from_zarr(zm)
    zm = zarr.open(mfile[1], mode='r')
    mean2 = da.from_zarr(zm)
    zs = zarr.open(sfile[0], mode='r')
    std1 = da.from_zarr(zs)
    zs = zarr.open(sfile[1], mode='r')
    std2 = da.from_zarr(zs)
    if rank==0:
        logger.info(f"{mean1.shape=}, {std1.shape=}, {mean2.shape=}, {std2.shape=}")
    ## delete rsids from x1, x2, means and stds
    if rmrsid is not None:
        lines = np.loadtxt(rmrsid, delimiter=",").astype('int')
        lines = lines[lines < p]
        if rank==0:
            logger.info(f"{lines=}")
        xdata1 = np.delete(xdata1, lines, axis=1)
        xdata2 = np.delete(xdata2, lines, axis=1) 
        if rank==0:
            logger.info(f"{xdata1=}")
            logger.info(f"{xdata2=}")
        mean1 = np.delete(mean1, lines, axis=0)
        mean2 = np.delete(mean2, lines, axis=0)
        std1 = np.delete(std1, lines, axis=0)
        std2 = np.delete(std2, lines, axis=0)
        group_idx = np.delete(group_idx, lines)
        if rank==0:
            logger.info(f"{group_idx=}")
            logger.info(f"after deleting lines: {mean1.shape=}, {std1.shape=}, {mean2.shape=}, {std2.shape=}")
        p -= len(lines)
        if rank==0:
            logger.info(f"new {p=}")

    #p_split = int(p/worldSize)
    if p_split * worldSize < p:
        p_split += 1
    if rank==0:
        logger.info(f"Data is split in {worldSize} * {p_split} columns.")
        
    # add columns of 0 for even split
    if p_split*worldSize-p > 0:
        c = p_split*worldSize-p
        if rank==0: logger.info(f"Added {c} columns of zeros to x.")
        xdata1 = da.concatenate([xdata1, np.zeros((n1, c))], axis=1)
        xdata2 = da.concatenate([xdata2, np.zeros((n2, c))], axis=1)
        if rank==0: logger.info(f"{group_idx.shape=}")
        group_idx = np.append(group_idx, np.ones(c)*G)
        group_idx = group_idx.astype(int)
        if rank==0: 
            logger.info(f"{group_idx.shape=}")
            logger.info(f"{G=}, {group_idx=}")
    # actually load only data that is needed in each process
    x1 = xdata1[:,rank*p_split:(rank+1)*p_split].compute()
    x2 = xdata2[:,rank*p_split:(rank+1)*p_split].compute()
    if rank==worldSize-1:
        logger.info(f"{rank=}, {x1.shape=}")
        logger.info(f"{rank=}, {x2.shape=}")
        logger.info(f"{rank=}, {np.unique(x1)=}")
        logger.info(f"{rank=}, {np.unique(x2)=}")
    ## split mean and std for each process
    mean1 = mean1[rank*p_split:(rank+1)*p_split].compute()
    std1 = std1[rank*p_split:(rank+1)*p_split].compute()
    mean2 = mean2[rank*p_split:(rank+1)*p_split].compute()
    std2 = std2[rank*p_split:(rank+1)*p_split].compute()
    if rank==0:
        logger.info(f"{np.unique(std1)=}")
        logger.info(f"{np.unique(std2)=}")

    # random generator
    rng = np.random.default_rng()
    ## initialize parameters
    beta = None
    tracker = None
    Z_sum = np.ones(G, dtype='i')*(p//G)
    Z = np.zeros(G, dtype='i')
    epsilon = np.empty(n, dtype=np.float64)
    sigma_inv = np.empty(q, dtype=np.float64)
    V_inv = np.empty((G,q,q), dtype=np.float64)
    pi_ratio = np.ones(G)

    if rank == 0:
        ## open phenotype file
        epsilon = np.loadtxt(yfile)

        # initalize parameters
        init = {
            "beta": np.zeros((p_split*worldSize, q)),
            "V": np.repeat([0.5*np.eye(q)], G, axis=0),
            "sigma": 0.5*np.ones(q),
            "pi": np.repeat(np.array([[0.90, 0.1]]), G, axis=0),
            "D": np.array(G*[np.ones(q)]),
            "L": np.array(G*[np.eye(q)]),
            "mu": np.zeros(q),
        }
        hypers = {
            "av": 2,
            "bv": 0.1,
            "sv": 0.0001
        }
        
        if restart==False:
            beta = init["beta"]
            V = np.array(init["V"])
            sigma = np.array(init["sigma"])
            pi = init["pi"]
            L = init["L"]
        else:
            V = np.loadtxt(dir+'/V_1000.txt').reshape(G,q,q)
            sigma = np.loadtxt(dir+'/sigma2_1000.txt')
            beta = pd.read_csv(dir+'/beta_1000.csv.zip', compression='zip').to_numpy()           
            beta = np.concatenate([beta, np.zeros((p_split*worldSize-p, q))], axis=0)
            L = np.loadtxt(dir+'/L_1000.txt').reshape(G,q,q)
            Z_sum = np.loadtxt(dir+'/Z_1000.txt').astype(np.int32).reshape(G)
            pi = init["pi"]
            for g in range(G):
                pi[g] = rng.dirichlet((groups[g]-Z_sum[g], Z_sum[g]))

        beta = beta.flatten() #vectorize for seding data
        mu = init["mu"]
        D = init["D"]
        tracker = np.zeros(p_split*worldSize)
        for g in range(G):
            V_inv[g] = linalg.inv(V[g])
        sigma_inv = 1/sigma
        if rank==0:
            logger.info(f"initialize V as {V=}")
            logger.info(f"initialize sigma as {sigma=}")

        # generate storage using the Welford package
        w_beta = welford.Welford()
        w_V = welford.Welford()
        w_sigma = welford.Welford()
        w_tracker = welford.Welford()
        # storage
        if diagnostics:
            trace_V = np.zeros((iters,G,q,q))
            trace_sigma = np.zeros((iters,q))
            trace_Z = np.zeros((iters, G))

    # initializing "split" data
    tracker_split = np.zeros(p_split)
    beta_split = np.zeros(p_split*q)
    comm.Scatterv([beta, p_split*q, MPI.DOUBLE], beta_split, root=0)
    beta_split = beta_split.reshape(p_split, q)

    # Loop through iterations
    if rank==0: logger.info(f"Running Gibbs with {iters=} and {burnin=}")
    for it in trange(iters, desc="Main loop"):

        comm.Barrier()
        if rank==0:            
            # ratio of probability(beta=0) and probability(beta != 0)
            pi_ratio = pi[:,0]/pi[:,1]

            # sample intercept term separately for each sex
            for i in range(q):
                start = 0 if i==0 else nv[i-1]
                end = n if i==q-1 else nv[i] 
                epsilon[start:end] += mu[i]
                mu[i] = np.mean(epsilon[start:end]) if it==0 else rng.normal(np.mean(epsilon[start:end]), sigma[i]/nv[i])
                epsilon[start:end] -= mu[i]
            # flatten matrices to vectors for sending
            V_inv = V_inv.flatten()
        
        #send relevant information to all processes
        comm.Bcast([epsilon, MPI.DOUBLE], root=0)
        comm.Bcast([sigma_inv, MPI.DOUBLE], root=0)
        comm.Bcast([V_inv, MPI.DOUBLE], root=0)
        comm.Bcast([pi_ratio, MPI.DOUBLE], root=0)
        comm.Bcast([Z_sum, MPI.INT], root=0)
        # reshape flattened vectors
        if rank== 0:
            V_inv = V_inv.reshape((G,q,q))

        # loop trough all markers randomly
        rj = np.arange(0, p_split)
        rng.shuffle(rj)
        ## containers for differences in epsilon
        diff = np.zeros(n)
        diff_sum = np.zeros(n)
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
                xe = np.zeros(q)
                # standardize x1 and substitute np.nan with mean
                xstd1 = x1[:,j:j+1]
                #logger.info(f"{xstd1=}")
                xstd1 = np.where(np.equal(xstd1,9), mean1[j:(j+1)], xstd1)
                #logger.info(f"{xstd1=}")
                xstd1 = (xstd1 - mean1[j:(j+1)])/std1[j:(j+1)]
                #logger.info(f"{n1=}, {xstd1.shape=}")
                xe[0] = (blas.dgemm(1, a=xstd1, b=epsilon[0:n1], trans_a=1))[0]
                #logger.info(f"{xe[0]=}")
                # standardize x2
                xstd2 = x2[:,j:j+1]
                xstd2 = np.where(np.equal(xstd2,9), mean2[j:(j+1)], xstd2)
                xstd2 = (xstd2 - mean2[j:(j+1)])/std2[j:(j+1)]
                #logger.info(f"{n2=}, {xstd2.shape=}")
                xe[1] = (blas.dgemm(1, a=xstd2, b=epsilon[n1:n], trans_a=1))[0]
                prev_beta = beta_split[j].copy()
                #logger.info(f"{prev_beta=}")
                # sample beta
                tracker_split[j], beta_split[j] = sample_mvn(
                    q,
                    nv,
                    prev_beta.reshape(-1,q),
                    xe, 
                    sigma_inv, 
                    V_inv[g]*Z_sum[g],
                    pi_ratio[g],
                    rng
                    )
                # calculate difference in epsilon
                #logger.info(f"{prev_beta[0]-beta_split[j,0]=}")
                diff[0:n1] += xstd1.reshape(n1)*(prev_beta[0]-beta_split[j,0])
                diff[n1:n] += xstd2.reshape(n2)*(prev_beta[1]-beta_split[j,1])
                # udpate number of non-zero betas
                Z[g] += tracker_split[j]
                counter += 1

            #receive and sum up diff after each process processed two markers
            if counter%itc==0 or counter == p_split:
                comm.Barrier()
                comm.Reduce(diff, diff_sum, MPI.SUM, root=0)
                                
                if rank==0:
                    epsilon = epsilon + diff_sum
                    diff_sum = np.zeros(n)
                comm.Bcast(epsilon, root=0)
                diff = np.zeros(n)
        
        comm.Barrier()
        # sum up number of non-zero effects
        comm.Reduce(Z, Z_sum, MPI.SUM, root=0)
        # pull together betas
        comm.Gatherv(sendbuf=beta_split, recvbuf=(beta, p_split*q), root = 0)
        comm.Gatherv(sendbuf=tracker_split, recvbuf=tracker, root = 0)        

        if rank == 0:
            beta = beta.reshape((p_split*worldSize, q))
            logger.info(f"{Z_sum=}")

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

            # update sigma2
            ## fast way
            sigma[0] = np.dot(epsilon[0:n1].T, epsilon[0:n1])/np.mean(stats.chi2.rvs(n1-2))
            sigma[1] = np.dot(epsilon[n1:n].T, epsilon[n1:n])/np.mean(stats.chi2.rvs(n2-2))
            sigma_inv = 1/sigma

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
                    np.savetxt(dir+'/trace_sigma.txt', trace_sigma)
                    np.savetxt(dir+'/trace_Z.txt', trace_Z)
                    for g in range(G):
                        np.savetxt(dir+'/trace_V'+str(g)+'.txt', trace_V[:,g].reshape(iters,4))
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
        logger.info(f"{mean_V=}")
        logger.info(f"{var_V=}")
        logger.info(f"{mean_sigma=}")
        logger.info(f"{var_sigma=}")
        ### save
        dfm = pd.DataFrame(mean_beta)
        dfm.to_csv(dir+'/mean_beta.csv.zip', index=False, compression='zip', sep=',')
        dfv = pd.DataFrame(var_beta)
        dfv.to_csv(dir+'/var_beta.csv.zip', index=False, compression='zip', sep=',')
        np.savetxt(dir+'/mean_V.txt', mean_V)
        np.savetxt(dir+'/var_V.txt', var_V)
        np.savetxt(dir+'/mean_sigma.txt', mean_sigma)
        np.savetxt(dir+'/var_sigma.txt', var_sigma)
        np.savetxt(dir+'/mean_prob.txt', mean_prob)
        np.savetxt(dir+'/var_prob.txt', var_prob)

        if diagnostics:
            np.savetxt(dir+'/trace_sigma.txt', trace_sigma)
            np.savetxt(dir+'/trace_Z.txt', trace_Z)
            for g in range(G):
                np.savetxt(dir+'/trace_V'+str(g)+'.txt', trace_V[:,g].reshape(iters,4))
        
            # Plotting sigma results
            logger.info("Plotting sigma results.")
            t = np.arange(iters)
            figS, axS = plt.subplots()
            for i in range(q):
                    axS.plot(t, trace_sigma[:,i])
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
    parser = argparse.ArgumentParser(description='Sex-specific Gibbs sampler.')
    parser.add_argument('--n', nargs='+', type=int, help='number of individuals for each sex', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    #parser.add_argument('--q', type=int, default=2, help='number of "traits" (default=2)')
    parser.add_argument('--g', nargs='+', type=int, help='number of markers in each group')
    parser.add_argument('--iters', type=int, default=5000, help='number of iterations (default = 5000)')
    parser.add_argument('--burnin', type=int, default=1000, help='number of iterations in burnin (default = 1000)')
    parser.add_argument('--itc', type=int, default=1, help='counter for updating epsilon (default=1)')
    parser.add_argument('--x', type=str, nargs='+', help='list of genotype matrix filenames (2 zarr files)', required = True)
    parser.add_argument('--y', type=str, help='phenotype matrix filename in txt file format, ordered according to "traits"', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--mfile', type=str, nargs="+", help='zarr files with means (2 files)', required = True)
    parser.add_argument('--sfile', type=str, nargs="+", help='zarr files with std. devs. (2 files)', required = True)
    parser.add_argument('--gindex', type=str, help='file with group index information')
    parser.add_argument('--diagnostics', type=bool, default=False, help='store traces for diagnostics; False by default')
    parser.add_argument('--restart', type=bool, default=False, help='restart with burnin values (default=False)')
    parser.add_argument('--rmid', type=str, nargs="+", help='list of ids to delete, 2 files (default is None)')
    parser.add_argument('--rmrsid', type=str, help='list of rsids to delete (default is None)')
    parser.add_argument('--p_split', type=int, default=20000, help='size of data blocks (default is 20000)')
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
    main(nv = np.array(args.n), # number of individuals
        p = args.p,  # number of markers
        q = 2, # number of sexes
        iters = args.iters, # number of iterations
        burnin = args.burnin, # number of iterations in burnin period
        groups = np.array(args.g), # number of markers in each group
        itc = args.itc, # counter for updating epsilon (after number of processes times itc markers)
        xfiles = args.x, # genotype file
        yfile = args.y, # phenotype file
        dir = args.dir, # path to results directory
        gindex = args.gindex, # path to directory with true values
        diagnostics = args.diagnostics, # boolean for diagnostics
        restart = args.restart, #boolean for restart
        mfile = args.mfile,
        sfile = args.sfile,
        rmid = args.rmid, # rows to remove from trios
        rmrsid = args.rmrsid, # columns to remove from genotype
        p_split = args.p_split,
        ) 
    logger.info("Done.")

