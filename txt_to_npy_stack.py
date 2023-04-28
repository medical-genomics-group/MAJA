# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru  dask
```
python txt_to_npy_stack.py --inputfile xfilename.txt --dir X
"""

import sys
import argparse
import numpy as np
from loguru import logger
import dask.array as da
#import datatable as dt

def main(xfile, dir):
    xdata = np.loadtxt(xfile)
    logger.info(f"{xdata=}")
    #xdata = dt.fread(xfile, sep = " ")
    #logger.info("Data loaded by fread.")
    #xdata = xdata.to_pandas()
    #xdata = xdata.to_numpy()
    x = da.from_array(xdata)
    logger.info(f"{x=}")
    logger.info(f"{dir=}")
    da.to_npy_stack(dir, x)

    xdata = da.from_npy_stack(dir)
    logger.info(f"{xdata=}")
    logger.info(f"{xdata.compute()=}")

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting txt to npy stack.')
    parser.add_argument('--inputfile', type=str, help='name of input file', required=True)
    parser.add_argument('--dir', type=str, help='name of output directory', required=True)
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
    main(xfile = args.inputfile,
        dir = args.dir
        ) 
    logger.info("Done.")