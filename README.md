# MAJA: MultivAriate Joint bAyesian model 

MAJA is a multivariate joint Bayesian method (Gibbs sampler) that is able to (i) estimate the unique contribution of individual loci, genes, or molecular pathways, to variation in one or more traits; (ii) determine genetic covariances and correlations; and (iii) find shared and distinct associations between multiple traits, while allowing for sparsity and correlations within the genomic data. It is suitable for high-dimensional data and flexible in the provided number of traits.

The code is written in python using MPI and tested on python/3.11.1 with openmpi/4.1.4 and python/3.12 with openmpi/4.1.6, run on a high performance computing cluster using slurm.
Information about which input parameters a program requires and how to run it is alos given in the first few lines of each program. 

## 1. Set up python environment:
```
module load python/3.11.1
module load openmpi/4.1.4
python -m venv *nameofyourenv*
source *nameofyourenv*/bin/activate
pip install -U pip
pip install numpy scipy matplotlib loguru mpi4py welford zarr dask pandas tqdm bed_reader scikit-allel
deactivate
```

## 2. Get code:
```
git clone https://github.com/medical-genomics-group/MAJA
```

## 3. Prepare data
The phenotypic data is required to be stored as txt format with one trait per column, standardized per column. Not available measures need to be imputed. <br/>
The genomic data needs to be standardized per columns and saved in zarr format (https://zarr.readthedocs.io/). zarr only stores the genomic values, thus one needs to keep track of variant or probe ids separately. Several files can be given as input, but the total amount of files needs to fit into RAM. 

plink or vcf files can be converted to zarr format using convert_to_zarr.py. Probes with more than 5% missing values will be removed and remaining NA values will be replaced with the column mean. Other data cleaning has to be done beforehand.

Load modules and source pyenv:
```
module load python/3.11.1
module load openmpi/4.1.4
source *nameofyourenv*/bin/activate
```
Run conversion code:
```
python convert_to_zarr.py --filename filename --outdir dir/
```
### Command line options:
```
--filename         name of vcf or bed file (required)
--outdir           output directory (required)
```
### Output:
**genotype.zarr**: genomic data in zarr format <br/>
**rsids.csv**: chromosome, bp position and id of the varinats saved in the same order as in genotype.zarr <br/>
**ids.csv**: sample IDs in the same order as in genotype.zarr <br/>

## 4. Run MAJA on data
Load modules and source pyenv (see step 3)

Run interactivley with 4 processes (given by -n): 
```
mpiexec -n 4 python -m mpi4py maja.py 
--n 10000 --p 80000 --q 3 
--iters 5000 --burnin 1000 
--x xinput.zarr --y phenotype.txt --dir results
--diagnostics True
--g 40000 40000
```

For slurm submission, the number of processes needs to be set by
```
#SBATCH --ntasks=number of processes
```

```
srun python -m mpi4py maja.py 
--n 10000 --p 80000 --q 3 
--iters 5000 --burnin 1000 
--x xinput1.zarr xinput2.zarr --y phenotype.txt --dir results
--diagnostics True
--g 40000 40000
```
The speed of MAJA will increase with the number of processes. Be aware that MAJA might not converge if this number is too high and the data is very correlated. The speed of MAJA can also be increased with the option --itc which is a counter for updating the residuals (see Command line options). Use this option with care, as setting itc to a higher number will increase the speed, but might also hinder convergence.

### Command line options:
```
--n           number of individuals (required)
--p           number of markers (required)
--q           number of traits (required)
--iters       total number of iterations (default=5000)
--burnin      number of iterations in the burnin (default=1000)
--x           genomic data files in zarr format; needs to be standardized per column (required); 
              can be several files with different probes which will be added sequentially
--y           phenotype file in txt format, 1 trait per column; needs to be standardized per column (required)
--dir         path to output directory (required)
--g           number of markers in each group (either give --g or --gindex)
--gindex      index file (.txt) with group index for each marker in order (either give --g or --gindex)
--diagnostics   turn on diagnostics: traces of errors and included markers are saved and plotted (default=False)
--restart     bool to restart sampler on iteration 999, i.e. after burnin (default=False); 
              requires the correspoding epsilon_999.txt file as y input; 
              if another iteration, is used for restart, the files need to be changed in the code
--itc         counter for updating the residuals/epsilon (default=1)
```

### Output:
The mean and variance of the posterior estimates across iterations (excluding the burnin) are saved for effects, variance, residual variance and inclusion probability.

**mean_betas.csv.zip:** posterior mean of effects where columns correspond to the genetic components and rows to the markers<br/>
**var_betas.csv.zip:** variance of effects sizes<br/>
**mean_prob.txt:** posterior inclusion probability for each marker (how often has the marker been included in the model); the marker is either included for all genetic components or not, thus only one column<br/>
**var_prob.txt:** variance of posterior inclusion probability<br/>
**mean_sigma2.txt:** posterior mean of residual variance<br/>
**var_sigma2.txt:** variance of residual variance<br/>
**mean_V.txt:** posterior mean of variance of effects (if the variance is estimated for different groups, the variances of each group are given one after the other)<br/>
**var_V.txt:** variance of variance of effects<br/>

During the burnin, the current estimates of every 500 iterations and iteration 999 (which is the last iteration in the burnin) is saved to be able to restart MAJA if necesseary. This includes:

**beta_XX.csv.zip:** effects <br/>
**epsilon_XX.txt:** residual errors (should be used as phenotype file when restarting) <br/>
**L_XX.txt:** part of covariance matrix V = LDL.T<br/>
**Le_XX.txt:** part of residual covariance matrix sigma2 = LeDeLe.T<br/>
**prob_XX.txt:** tracker if marker is included in model (1) or not (0) for iteration XX<br/>
**sigma2_XX.txt:** residual variance of iteration XX<br/>
**V_XX.txt:** variance of effects of iteration XX<br/>
**Z_XX.txt:** number of markers included in the model at iteration XX    <br/>

If the option --diagnostics is set to True, the following files will be stored:

**trace_sigma2.txt:** residual variance for each iteration (rows = iterations, columns = variance for each trait); only variances are saved<br/>
**trace_Vg.txt:** variance of effects for group g for each iteration (rows = iterations, columns = variance for each trait); only variances are saved<br/>
**trace_Z.txt:** number of included markers for each iteration (rows = iterations); the marker is either included for all genetic components or not, thus only one column<br/>
**trace_sigma2.png:** residual variance as function of iteration<br/>
**trace_V.png:** (co)variances as function of iterations<br/>
**trace_Z.png:** number of included markers as function of iterations<br/>


## Simulating effects on real data
The script *genY.py* generates phenotypes and effects using real genomic data, assuming different effects covariances. The residual variances are generated assuming independence between the traits. Currently, one or two groups can be generated.

### Command line options:
```
--n       number of individuals
--p       total number of markers, variants or probes split per group
--p0      number of markers set to 0 per group
--q       number of traits (currently only supports 2 or 3)
--xfiles  path to genomic data files in zarr format
--dir     path to output directory
--scen    covariance scenarios (0=indepedent traits, 1=negative correlations, 2=positive correlations); see code for exact defintion
```

### Output:
**true_V.txt**: true simulated variance of effects; each q times q block represents a group<br/>
**true_beta.txt**: true simulated effects<br/>
**true_sigma2.txt**: true simulated residual variance<br/>
**true_epsilon.txt**: true simulated residuals<br/>
**phenotype.txt**: generated phenotypes<br/>

The *phenotype.txt* and the corresponding xfiles can then be used as input for step 3.


## Test MAJA on example MC
The directory MC contains simulated genomic and phenotypic data for two traits in the correct file formats for n=1,000 individuals. The genomic data (genotype.zarr) emulates methylation data, i.e. each of the p=2,000 simulated probes is drawn indpendently from a standard normal distribution. The 500 causal effects (true_betas.txt) are drawn from a multivariate normal distribution with V = [[0.5, 0.5 * sqrt(0.3 * 0.5)], [0.5 * sqrt(0.3 * 0.5), 0.3]] and randomly assigned to probes. The rest of the effects are set to 0. Residual errors (true_epsilon.txt) are drawn from a normal distribution with mean 0 and standard deviation sqrt(1-var(X betas)). The penotypes (phenotype.txt) are calculated as genotypic data times effects plus residual errors.

Run interactively with 2 processes on simulated data using the command:
```
mpiexec -n 2 python -m mpi4py maja.py 
--n 1000 --p 2000 --q 2 
--iters 2000 --burnin 1000 
--x MC/genotype.zarr --y MC/phenotype.txt --dir results/
--diagnostics True
--g 2000
```
In case you are having troubles running this example with openmpi on a Mac, try the option --pmixmca ptl_tcp_if_include lo0

Since it is a very small dataset meant for testing, you can also run without MPI.
```
python maja.py 
--n 1000 --p 2000 --q 2 
--iters 2000 --burnin 1000 
--x MC/genotype.zarr --y MC/phenotype.txt --dir results/
--diagnostics True
--g 2000
```

## Association studies
The model is set up so that markers are either included in the model for all genetic components or not included at all. Therefore, if a marker is included with a high posterior inclusion probability, one needs to check for each trait if the effect size +/- standard deviation includes 0. If 0 is covered by effect size +/- standard deviation, there is no association.


In case of questions or problems, please contact ilse.kraetschmer@ist.ac.at
