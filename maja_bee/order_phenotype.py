#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Install dependencies:
```
pip install numpy loguru pandas
```
python order_phenotype.py 
--y path to phenotype file with phenotype childID, parentID, pheno (required)
--index pedigree/index file with child, father and mother ids, delimiter=" " (required)
--name output file name: will just contain standardized phenotype, ordered first for all fathers (parent 1), then all mothers (parent 2)
--odir output directory
--include_duos if True, duos will be included (default = False, i.e. only trios)
"""
import numpy as np
import sys
import argparse
from loguru import logger
import pandas as pd

def main(yfile, index, odir, name, duos):

    # read in phenotype files, drop NA
    ## familyID, parental ID, value
    y1 = pd.read_csv(yfile[0], delimiter="\t")
    y1.columns = ["ID", "PID", "VALUE"]
    y1["ID"] = y1["ID"].astype("str")
    #, names = ["ID", "PID", "VALUE"], dtype={"ID": str, "PID": str, "VALUE": float}, usecols=[0, 1, 2])
    y1 = y1.dropna()
    logger.info(f"{y1=}")
    y2 = pd.read_csv(yfile[1], delimiter="\t")
    y2.columns = ["ID", "PID", "VALUE"]
    y2["ID"] = y2["ID"].astype("str")
    #, names = ["ID", "PID", "VALUE"], dtype={"ID": str, "PID": str, "VALUE": float}, usecols=[0, 1, 2])
    y2 = y2.dropna()
    logger.info(f"{y2=}")

    # read in indices for child
    id = pd.read_csv(index, delimiter=" ", names = ["ID", "FID", "MID"], dtype={"ID": str, "FID": str, "MID": str})
    logger.info(f"{id=}")
    # re-order: trios, father-child duos, mother-child duos
    mask_trios = id.isna().any(axis=1)
    mask_duos1 = id["MID"].isna() ## mother is na
    mask_duos2 = id["FID"].isna() ## father is na

    ## FATHERS: drop mother-child duos
    ## assign values to parental id
    logger.info(f"first parent")
    if duos:
        id1 = pd.concat((id[~mask_trios], id[mask_duos1]), axis=0).reset_index(drop=True)
    else:
        id1 = id[~mask_trios].reset_index(drop=True)
    result1 = id1.merge(y1, left_on='ID', right_on='ID', how='left')
    logger.info(f"{id1=}")
    logger.info(f"{result1=}")

    ## MOTHERS: drop father-child duos for mothers
    logger.info(f"second parent")
    if duos:
        id2 = pd.concat((id[~mask_trios], id[mask_duos2]), axis=0).reset_index(drop=True)
    else:
        id2 = id1
    result2 = id2.merge(y2, left_on='ID', right_on='ID', how='left')
    logger.info(f"{id2=}")
    logger.info(f"{result2=}")

    ## keep track of indices where phenotype is NA
    id_na1 = np.where(result1['VALUE'].isna())
    id_na2 = np.where(result2['VALUE'].isna())
    logger.info(f"{id_na1=}")
    logger.info(f"{id_na2=}")
    logger.info(f"{result1['VALUE'].dropna().values.shape=}")
    logger.info(f"{result2['VALUE'].dropna().values.shape=}")

    ## save phenotype and dropped indices
    if odir is not None:
        odir=odir+"/"
    else:
        odir=""
    if duos:
        duos="trios_duos_"
    else:
        duos=""
    ## standardize
    result1 = result1["VALUE"].dropna()
    result1 = (result1-result1.mean())/result1.std()
    result2 = result2["VALUE"].dropna()
    result2 = (result2-result2.mean())/result2.std()
    ## save
    np.savetxt(odir+"ordered_"+duos+name, np.concatenate([result1, result2]))
    np.savetxt(odir+"rmid_ordered_parent1_"+duos+name, id_na1, fmt='%i', delimiter=",")
    np.savetxt(odir+"rmid_ordered_parent2_"+duos+name, id_na2, fmt='%i', delimiter=",")
    
##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing.')
    parser.add_argument('--y', type=str, nargs="+", help='name of input files, one for each parent, order should match pedigree file', required=True)
    parser.add_argument('--index', type=str, help='list of indices', required=True)
    parser.add_argument('--include_duos', type=bool, default=False, help='include duos (default=False)')
    parser.add_argument('--odir', type=str, help='name of output directory')
    parser.add_argument('--name', type=str, help='name of output file', required=True)
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
    main(yfile = args.y, # inputfile
        index = args.index, # list of indices
        odir = args.odir, # ouput directory
        name = args.name, # output filename
        duos =args.include_duos,
        ) 
    logger.info("Done.")


