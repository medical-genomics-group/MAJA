# MAJA: MultivAriate Joint bAyesian model for two sexes instead of multiple traits

maja_bee.py is an adatped version of maja.py which is set up to estimate covariances between two sexes and assumes independent residual errors.

Setup is the same as for maja.py. The preparation and input for the Gibbs sampler varies slightly, as explained below.

## 3. Prepare data for maja_bee.py
The phenotypic data is required to be stored as txt format, one column only, with measures of sex 1 and then measures of sex2, standardized for each sex. This can be done with order_phenotype.py. Not available measures need to be imputed or removed. <br/>
The genomic data needs to be saved in zarr format (https://zarr.readthedocs.io/) and not standardized. zarr only stores the genomic values, thus one needs to keep track of variant or probe ids separately (for example, using get_rsids.py to get the rsids from the vcf files). Several files can be given as input, but the total amount of files needs to fit into RAM. 

zarr files of the genomic data (not standardized) for each sex should be prepared beforehand. The needed zarr files for mean and standard deviations can be created with get_mean_std_rmrsids.py.

## 4. Run maja_bee on data
Similarly to maja.py, but with slightly different input for n, x, mfile, sfile

### Command line options:
```
--n n1 n2   number of individuals, two numbers (separated by space) 
            for father (parent1) and mother (parent2) (required) 
--p         number of markers (required)
--x         2 zarr files with genome data (separated by space) 
            for father (parent1) and mother (parent2) (required)
--mfile     2 zarr files with means for each marker (separated by space) 
            for father (parent1) and mother (parent2); created during preprocessing  (required)
--sfile     2 zarr files with standard deviations for each marker (separated by space) 
            for father (parent1) and mother (parent2); created during preprocessing  (required)
--y         standardized phenotype file in txt format created in order_phenotype.py (required)
--dir       path to output directory (required)
--iters     total number of iterations (default=5000)
--burnin    number of iterations in the burnin (default=1000)
--rmid      2 files for ids to remove for father (parent1) and mother (parent2) created during preprocessing
--rmrsid    txt file with snps to remove (according to number of column in zarr file) created during preprocessing
--diagnostics   make trace plots for diagnostics (default=False)
--restart   bool to restart sampler on iteration 1000, i.e. after burnin (default=False); 
            requires the correspoding epsilon_1000.txt file as y input; if another iteration, is used for restart, the files need to be changed in the code
--itc       counter for updating epsilon (default=1); can be changed for speed, but might cause divergence issues
--p_split   size of data blocks (default=20000); will help with memory if set correctly 
            must be divisible by chunk size in X zarr array
            must be set up such that (number of processes * p_split) > number of markers, but < (number of markers + p_split)
--g         number of markers within each group; if there is only 1 group, g=p (either give --g or --gindex)
--gindex    index file (.txt) with group index (starting from 0) for each marker in order 
            (either give --g or --gindex)
```