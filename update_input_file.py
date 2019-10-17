"""
Update an existing mcmcfit input file using the results of a
previous chain.
"""
import argparse

import configobj
import numpy as np

from mcmc_utils import flatchain, readchain_dask as readchain


def update_entry(name, input_dict, newval):
    fields = input_dict[name].split(' ')
    fields = [v for v in fields if v != '']
    fields[0] = str(newval)
    input_dict[name] = '    '.join(fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--thin", action="store", type=int,
                        default=10, help="thinning for chain (default=10)")
    parser.add_argument('input_file', help='mcmc input file')
    parser.add_argument('chain_file', help='chain output file')
    args = parser.parse_args()

    input_dict = configobj.ConfigObj(args.input_file)
    print("Reading in the chain")
    chain = readchain(args.chain_file)
    nwalkers, nsteps, npars = chain.shape
    fchain = flatchain(chain, npars, thin=args.thin)

    # get param names from chain file
    firstLine = open(args.chain_file).readline()
    # first entry is walker number - strip
    fields = firstLine.strip().split()[1:]
    # strip "core" from core params
    for i in range(len(fields)):
        if 'core' in fields[i]:
            fields[i] = fields[i].split("_")[0]

    # get median values
    vals = np.median(fchain, axis=0)

    # update
    for n, v in zip(fields, vals):
        if n in input_dict:
            print("\nUpdating the entry:")
            print("n: {}".format(n))
            print("v: {}".format(v))
            update_entry(n, input_dict, v)

    input_dict.write()
