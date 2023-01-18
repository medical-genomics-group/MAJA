# multitrait

## gen_data.py:

Generate data

### Command line options:

--n number of individuals

--p number of markers / CpG sites

--p0 number of markers / CpG sites set to 0

--q number of traits

--dir path to directory where the generated data should be stored. The directory has to be created beforehand.

### Output:

genotype.npz

phenotype.txt

true_betas.txt

true_epsilon.txt

true_V.txt

true_sigma.txt


## multi-mpi-groups.py:

Gibbs sampler with groups

### Command line options:

--n number of individuals

--p number of markers / CpG sites

--q number of traits

--iters number of iterations

--burnin number of iterations in the burnin period

--x name of file with methylation data (incl. path)

--y name of phenotype file (incl. path)

--dir output directory. Needs to be created beforehand

--true_dir path to directory where the true values are stored (this is just for testing)

### Output:

mean_beta.txt

var_beta.txt

mean_V.txt

var_V.txt

mean_sigma.txt

var_sigma.txt

## plotting.py:

Some output plots

### Command line options:

--q number of traits
--g number of groups
--inputdir path to directory where true values are stored
--resultsdir path to directory where results are stored

### Output plots:

estimated vs true effect sizes

comparison of estimated and true covariances
