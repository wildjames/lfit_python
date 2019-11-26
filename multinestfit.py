
import argparse
import multiprocessing as mp
import os
from pprint import pprint
from shutil import rmtree

import configobj
import emcee
import numpy as np

import mcmc_utils as utils
import plot_lc_model as plotCV
from CVModel import construct_model, extract_par_and_key


if __name__ in '__main__':

    # Set up the parser.
    parser = argparse.ArgumentParser(
        description='''Execute an MCMC fit to a dataset.'''
    )

    parser.add_argument(
        "input",
        help="The filename for the MCMC parameters' input file.",
        type=str,
    )
    parser.add_argument(
        '--notify',
        help="The script will email a summary results of the MCMC to this address",
        type=str,
        default=''
    )
    parser.add_argument(
        '--debug',
        help='Enable the debugging flag in the model',
        action='store_true'
    )

    parser.add_argument(
        "--quiet",
        help="Do not plot the initial conditions",
        action="store_true"
    )

    args = parser.parse_args()
    input_fname = args.input
    dest = args.notify
    debug = args.debug
    quiet = args.quiet

    if debug:
        if os.path.isdir("DEBUGGING"):
            rmtree("DEBUGGING")

    # I want to pre-check that the details have been supplied.
    if dest != '':
        location = __file__.split('/')[:-1] + ["email_details.json"]
        details_loc = '/'.join(location)
        if not os.path.isfile(details_loc):
            print("Couldn't find the file {}! Creating it now.")
            with open(details_loc, 'w') as f:
                s = '{\n  "user": "Bot email address",\n  "pass": "Bot email password"\n}'
                f.write(s)

        # Check that the details file has been filled in.
        # If it hasn't, ask the user to get it done.
        with open(details_loc, 'r') as f:
            details = f.read()
        if "Bot email address" in details:
            print("The model will continue for now, but there are no")
            print("email credentials supplied and the code will fail")
            print("when it tries to send it.")
            print("Don't panic, just complete the JSON file here:")
            print("{}".format(details_loc))

    # Build the model from the input file
    model = construct_model(input_fname, debug, True)

    print("\nStructure:")
    pprint(model.structure)

    input_dict = configobj.ConfigObj(input_fname)

    # Read in information about mcmc
    nthreads = int(input_dict['nthread'])
    nwalkers = int(input_dict['nwalkers'])
    scatter_1 = float(input_dict['first_scatter'])
    to_fit = int(input_dict['fit'])
    comp_scat = bool(int(input_dict['comp_scat']))

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
    try:
        neclipses = int(input_dict['neclipses'])
    except KeyError:
        neclipses = len(model.search_node_type("Eclipse"))
        print("The model has {} eclipses.".format(neclipses))

    # Wok out how many degrees of freedom we have in the model
    # How many data points do we have?
    dof = np.sum([ecl.lc.n_data for ecl in model.search_node_type('Eclipse')])
    # Subtract a DoF for each variable
    dof -= len(model.dynasty_par_names)
    # Subtract one DoF for the fit
    dof -= 1
    dof = int(dof)

    print("\n\nInitial guess has a chisq of {:.3f} ({:d} D.o.F.).".format(model.chisq(), dof))
    print("\nFrom the wrapper functions with the above parameters, we get;")
    pars = model.dynasty_par_vals
    print("a ln_prior of {:.3f}".format(ln_prior(pars, model)))
    print("a ln_like of {:.3f}".format(ln_like(pars, model)))
    print("a ln_prob of {:.3f}".format(ln_prob(pars, model)))
    print()
    if np.isinf(model.ln_prior()):
        print("ERROR: Starting position violates priors!")
        print("Offending parameters are:")

        pars, names = model.__get_descendant_params__()
        for par, name in zip(pars, names):
            print("{:>15s}_{:<5s}: Valid?: {}".format(
                par.name, name, par.isValid))

            if not par.isValid:
                print("  -> {}_{}".format(par.name, name))

        # Calculate ln_prior verbosely, for the user's benefit
        model.ln_prior(verbose=True)
        exit()

    # If we're not running the fit, plot our stuff.
    if not quiet:
        plotCV.nxdraw(model)
        plotCV.plot_model(model, True, save=True, figsize=(11, 8), save_dir='Initial_figs/')
    if not to_fit:
        exit()

    # # # # # # # # # # # # # # # # # # # # # # # #
    # #       Run the PyMultiNest sampler       # #
    # # # # # # # # # # # # # # # # # # # # # # # #

    ndims = model.n_dim
    n_params = model.n_params

