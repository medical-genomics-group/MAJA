# -*- coding: utf-8 -*-
"""
plot true and false positive rate for different posterior inclusion probabilities

Install dependencies:
```
pip install numpy loguru matplotlib
```
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def main(q, inputdir, resultsdir):

    ## true betas
    true_beta = np.loadtxt(inputdir+'/true_betas.txt')
    logger.info(f"{true_beta.shape=}")
    #estimated values
    mean_beta = np.loadtxt(resultsdir+'/mean_beta.txt')
    var_beta = np.loadtxt(resultsdir+'/var_beta.txt')
    mean_prob = np.loadtxt(resultsdir+'/mean_prob.txt')
    logger.info(f"{mean_prob.shape=}")

    # different posterior inclusion probability thresholds
    threshold = np.arange(0, 1.05, 0.05)
    lent = len(threshold)-1
    # containers for true and false postive rates
    tpr = np.zeros((lent,q))
    fpr = np.zeros((lent,q))
    tnr = np.zeros((lent,q))
    # pip == 0.95
    a = np.zeros(q)

    # loop through traits
    for i in range(q):

        tbeta = true_beta[:,i]
        # true number of non-zero markers (real positives)
        P = np.array(np.where(np.not_equal(tbeta, 0))).shape[1]
        # true number of zero markers (real negatives)
        N = np.array(np.where(np.equal(tbeta, 0))).shape[1]

        for j in range(lent):

            t = threshold[j]
            logger.info(f"{t=}")
            if np.isclose(t,0.25):
                a[i] = np.array(np.where(np.equal(tbeta, 0) & (mean_prob > t))).shape[1]/N
            # false positive: true_beta == 0 and mean_prob > t
            fpr[j,i] = np.array(np.where(np.equal(tbeta, 0) & (mean_prob > t))).shape[1]
            # true positive: true_beta != 0 and mean_prob > t
            tpr[j,i] = np.array(np.where(np.not_equal(tbeta, 0) & (mean_prob > t))).shape[1]
            # true negative: true_beta == 0 and mean_prob < t
            tnr[j,i] = np.array(np.where(np.equal(tbeta, 0) & (mean_prob < t))).shape[1]

    # divided obtained numbers by "real" ones to get rate
    logger.info(f"{P=}")
    logger.info(f"{N=}")
    logger.info(f"{tpr=}")
    logger.info(f"{fpr=}")
    logger.info(f"{tnr=}")
    tpr /= P
    fpr /= N
    tnr /= N
    logger.info(f"{tpr=}")
    logger.info(f"{fpr=}")
    logger.info(f"{tnr=}")

    stpr1=np.array([[0.999412, 0.999412],
       [0.661765, 0.661765],
       [0.460588, 0.460588],
       [0.392353, 0.392353],
       [0.356471, 0.356471],
       [0.327059, 0.327059],
       [0.301176, 0.301176],
       [0.283529, 0.283529],
       [0.264706, 0.264706],
       [0.25    , 0.25    ],
       [0.241176, 0.241176],
       [0.231765, 0.231765],
       [0.221765, 0.221765],
       [0.205882, 0.205882],
       [0.194118, 0.194118],
       [0.185294, 0.185294],
       [0.175882, 0.175882],
       [0.166471, 0.166471],
       [0.155294, 0.155294],
       [0.137059, 0.137059]])
    
    sfpr1=np.array([[0.999258, 0.999258],
       [0.207562, 0.207562],
       [0.058905, 0.058905],
       [0.029965, 0.029965],
       [0.017138, 0.017138],
       [0.011378, 0.011378],
       [0.00841 , 0.00841 ],
       [0.006254, 0.006254],
       [0.004876, 0.004876],
       [0.003993, 0.003993],
       [0.003286, 0.003286],
       [0.00265 , 0.00265 ],
       [0.002226, 0.002226],
       [0.001555, 0.001555],
       [0.000954, 0.000954],
       [0.000777, 0.000777],
       [0.00053 , 0.00053 ],
       [0.000389, 0.000389],
       [0.000283, 0.000283],
       [0.000141, 0.000141]])
    
    stpr2=np.array([[0.998235, 0.998235],
       [0.721765, 0.721765],
       [0.606471, 0.606471],
       [0.564118, 0.564118],
       [0.533529, 0.533529],
       [0.514706, 0.514706],
       [0.494706, 0.494706],
       [0.477647, 0.477647],
       [0.466471, 0.466471],
       [0.46    , 0.46    ],
       [0.447647, 0.447647],
       [0.44    , 0.44    ],
       [0.428824, 0.428824],
       [0.418235, 0.418235],
       [0.405882, 0.405882],
       [0.395294, 0.395294],
       [0.384118, 0.384118],
       [0.365294, 0.365294],
       [0.345882, 0.345882],
       [0.32    , 0.32    ]])
    
    sfpr2=np.array([[0.991343, 0.991343],
       [0.146219, 0.146219],
       [0.050989, 0.050989],
       [0.027633, 0.027633],
       [0.018481, 0.018481],
       [0.01265 , 0.01265 ],
       [0.009046, 0.009046],
       [0.006466, 0.006466],
       [0.005265, 0.005265],
       [0.004276, 0.004276],
       [0.003428, 0.003428],
       [0.002968, 0.002968],
       [0.002403, 0.002403],
       [0.001873, 0.001873],
       [0.001378, 0.001378],
       [0.000954, 0.000954],
       [0.000671, 0.000671],
       [0.000424, 0.000424],
       [0.000141, 0.000141],
       [0.000035, 0.000035]])

    # improve plots by setting fontsizes and removing white space
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['ytick.labelsize']=20
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams["text.usetex"]=True
    plt.rcParams['legend.handlelength'] = 0 # remove errorbar from legend

    ## plot ROC
    logger.info(f"{a=}")
    # plot
    fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    #ax.xaxis.set_ticks(np.arange(0, 1, 0.1))
    ax.plot(fpr[:,0], tpr[:,0], linewidth=1.0, color='red', marker='x', label='multi-trait')
    ax.plot(sfpr1[:,0], stpr1[:,0], linewidth=1.0, color='green', marker='x', label='trait 1')
    ax.plot(sfpr2[:,0], stpr2[:,0], linewidth=1.0, color='blue', marker='x', label='trait 2')
    #ax.plot(fpr[:,1], tpr[:,1], linewidth=1.0, color='green')
    ax.set(ylabel='true positive rate', xlabel='false positive rate')
    #ax.axvline(x=a[0], color='gray', linestyle='--')
    #ax.axvline(x=0.05, color='gray', linestyle='dotted')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='gray', ls="--", linewidth=1.0)
    fig.legend(loc=4, fontsize=20, framealpha=1.0, bbox_to_anchor=[0.95, 0.1], edgecolor='black')
    fig.savefig(resultsdir+'/roc.png')

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Checks.')
    parser.add_argument('--q', type=int, help='number of traits', required = True)
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
        inputdir = args.inputdir, # path to directory where true values are stored
        resultsdir = args.resultsdir, # path to directory where results are stored
        )
    logger.info("Done.")