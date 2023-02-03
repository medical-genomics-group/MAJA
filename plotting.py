# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru matplotlib
```
python plotting.py --q 2 --g 2 --inputdir MC --resultsdir results
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def main(q, groups, inputdir, resultsdir):
    ## true values
    true_V = np.loadtxt(inputdir+'/true_V.txt')
    true_sigma = np.loadtxt(inputdir+'/true_sigma.txt')
    true_beta = np.loadtxt(inputdir+'/true_betas.txt')
    #estimated values
    mean_beta = np.loadtxt(resultsdir+'/mean_beta.txt')
    trace_V = np.loadtxt(resultsdir+'/mean_V.txt')
    trace_V_std = np.loadtxt(resultsdir+'/var_V.txt')
    trace_sigma = np.loadtxt(resultsdir+'/mean_sigma.txt')
    trace_sigma_std = np.loadtxt(resultsdir+'/var_sigma.txt')

    # improve plots by setting fontsizes and removing white space
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['ytick.labelsize']=15
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams["text.usetex"]=True
    plt.rcParams['legend.handlelength'] = 0 # remove errorbar from legend
 
    # estimated vs true beta
    fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    ax.scatter(x=true_beta[:,0], y=mean_beta[:,0], facecolors='none', edgecolors='grey', label="trait 1")
    ax.scatter(x=true_beta[:,1], y=mean_beta[:,1], marker="x", color="red", label="trait 2")
    ax.set(xlabel="true effect sizes", ylabel="estimated effect sizes")
    ax.legend(loc=4, fontsize=18, framealpha=0.)
    #plt.margins(0,0)
    fig.savefig(resultsdir+'/est_vs_true_beta.png')
    ax.set(xlim=(-0.02, 0.02), ylim=(-0.02, 0.02))
    fig.savefig(resultsdir+'/est_vs_true_beta_zoom.png')

    # estimated vs true V
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = -50  # pad is in points...
    figV, plotV = plt.subplots(ncols=2,figsize=(22,10), tight_layout=True)
    labels = [r'$\mathbf{V_{11}}$', r'$\mathbf{V_{12}}$', r'$\mathbf{V_{21}}$', r'$\mathbf{V_{22}}$',]
    for g in range(groups):
        z = 0
        for i in range(q):
            for j in range(q):
                if z==0:
                    plotV[g].errorbar(z, trace_V[g*q+i, j], yerr= np.sqrt(trace_V_std[g*q+i,j]),marker='o', color='red', mfc='None', label='est.')
                    plotV[g].errorbar(z, true_V[g*q+i, j], marker='x', color='black', label='true')
                else:
                    plotV[g].errorbar(z, trace_V[g*q+i, j], yerr= np.sqrt(trace_V_std[g*q+i,j]),marker='o', color='red', mfc='None')
                    plotV[g].errorbar(z, true_V[g*q+i, j], marker='x', color='black')
                z += 1
        title = 'group '+str(g)
        plotV[g].set(ylabel='values', title=title)
        plotV[g].legend(loc=9, fontsize=18, framealpha=0., bbox_to_anchor=[0.5, 0.9])
        plt.sca(plotV[g])
        plt.xticks(np.arange(q*q), labels, fontsize=15, weight='bold')
    figV.savefig(resultsdir+'/V.png')

    # estimated vs true sigma
    figS, plotS = plt.subplots(figsize=(10,9.6), tight_layout=True)
    labels = [r'$\mathbf{\Sigma_{11}}$', r'$\mathbf{\Sigma_{12}}$', r'$\mathbf{\Sigma_{21}}$', r'$\mathbf{\Sigma_{22}}$']
    plt.xticks(np.arange(q*q), labels, fontsize=18, weight='bold')
    z = 0
    for i in range(q):
        for j in range(q):
            if z == 0:
                plotS.errorbar(z, trace_sigma[i, j], yerr=np.sqrt(trace_sigma_std[i,j]), marker='o', color='red', mfc='None',label='est.')
                plotS.errorbar(z, true_sigma[i, j], marker='x', color='black', label='true')
            else:
                plotS.errorbar(z, trace_sigma[i, j], yerr=np.sqrt(trace_sigma_std[i,j]), marker='o', color='red', mfc='None')
                plotS.errorbar(z, true_sigma[i, j], marker='x', color='black')
            z += 1
    plotS.set(ylabel='values')
    figS.legend(loc=9, fontsize=18, framealpha=0.)
    figS.savefig(resultsdir+'/sigma.png')

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Plotting Gibbs results.')
    parser.add_argument('--q', type=int, help='number of traits', required = True)
    parser.add_argument('--g', type=int, help='number of groups', required = True)
    parser.add_argument('--inputdir', type=str, help='path to directory where true values are stored', required = True)
    parser.add_argument('--resultsdir', type=str, help='path to directory where the results are stored', required = True)
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
    main(q = args.q, # number of traits
        groups = args.g, # number of groups
        inputdir = args.inputdir, # path to directory where true values are stored
        resultsdir = args.resultsdir, # path to directory where results are stored
        )
    logger.info("Done.")