
import argparse
import multiprocessing as mp
import os
from pprint import pprint
from shutil import rmtree

import configobj
import emcee
import numpy as np
from scipy.special import erf, erfinv
from pymultinest.solve import Solver
from pymultinest.analyse import Analyzer

import mcmc_utils as utils
import plot_lc_model as plotCV
from CVModel import construct_model, extract_par_and_key


class CubeConverter(object):
    '''
    For each prior function in the Prior class, there must be
    a corresponding <cube2func> function on this one for it to be
    usable with simulated annealing.

    The <cube2func> function must take a value, u, between 0:1, and
    transform it according to this equation:

    ln_prob(\theta) d(theta) = d(u)

    for a parameter theta.

    Ideally, solve this analytically. Failing that, numerical integration
    is likely the best way to go.
    '''
    def convert(self, u_i, prior):
        func = getattr(self, "cube2{}".format(prior.type))
        theta_i = func(u_i, prior.p1, prior.p2)
        return theta_i

    def cube2gauss(self, u_i, p1, p2):
        '''Gaussian has a mean, mu, and a deviation of sigma.'''
        theta_i = p2 * np.sqrt(2) * erfinv(2*u_i - 1) + p1
        return theta_i

    def cube2uniform(self, u, p1, p2):
        theta = p1 + (u*np.abs(p2 - p1))
        return theta

    def cube2log_uniform(self, u, p1, p2):
        ln_theta = np.log(p1) + (u * (np.log(p2) - np.log(p1)))
        theta = np.exp(ln_theta)
        return theta

    def cube2gaussPos(self, u, p1, p2):
        raise NotImplementedError("GaussPos has not been implimented with MultiNest!")

    def cube2mod_jeff(self, u, p1, p2):
        raise NotImplementedError("Modified Jefferies prior not implemented with MultiNest!")



class HierarchicalModelSolver(Solver):
    DEBUG = True
    def __init__(self, model, *args, **kwargs):
        print("args passed to me:")
        print(args)
        print("Kwargs passed to me:")
        for k, v in kwargs.items():
            print("{}: {}".format(k, v))


        self.model = model
        self.priors = [par.prior for par in model.dynasty_par_list[0]]

        self.convert = CubeConverter().convert

        super().__init__(*args, **kwargs)

    def chisq(self, vector):
        self.model.dynasty_par_vals = vector

        return self.model.chisq()

    def Prior(self, cube):
        '''Take a cube vector, and return the correct thetas'''
        vect = []

        if self.DEBUG:
            print("Entered prior with this cube:")
            print(cube)
        for u, prior in zip(cube, self.priors):
            if self.DEBUG:
                print("--------")
                print("  u = {:.2f}".format(u))
            theta = self.convert(u, prior)

            vect.append(theta)

            if self.DEBUG:
                print("  theta = {:.2f}".format(theta))
                print("  lnp = {:.2f}".format(prior.ln_prob(theta)))

        if self.DEBUG:
            input("> ")

        return vect

    def LogLikelihood(self, vect):
        '''Take a parameter vector, convert to desired parameters, and calculate the ln(like) of that vector

        This will be maximized
        '''
        if self.DEBUG:
            print("~~~~~~~~~~~")
            print("Entered the LogLikelihood with this vector:")
            print(vect)

        self.model.dynasty_par_vals = vect
        ln_like = self.model.ln_like()

        if self.DEBUG:
            print("Got a ln(like) of {:.3f}".format(ln_like))

        return ln_like


if __name__ in '__main__':

    # Set up the parser.
    parser = argparse.ArgumentParser(
        description='''Execute an MultiNest fit to a dataset.'''
    )

    parser.add_argument(
        "input",
        help="The filename for the MultiNest and CV Model parameters' input file.",
        type=str,
    )
    parser.add_argument(
        '--notify',
        help="The script will email a summary results of the fit to this address",
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
    model = construct_model(input_fname, debug)

    print("\nStructure:")
    pprint(model.structure)

    input_dict = configobj.ConfigObj(input_fname)

    # Read in information about mcmc
    nthreads = int(input_dict['nthread'])
    nlive = int(input_dict['nlive'])
    to_fit = int(input_dict['fit'])

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
    print("\nFrom the model's functions with the above parameters, we get;")
    print("a ln_prior of {:.3f}".format(model.ln_prior()))
    print("a ln_like of {:.3f}".format(model.ln_like()))
    print("a ln_prob of {:.3f}".format(model.ln_prob()))
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

    ndim = 0
    nparams = 0
    for par in model.dynasty_par_list[0]:
        nparams += 1
        if par.isVar:
            ndim += 1

    print("ndim: {}\nnparams: {}".format(ndim, nparams))

    solution = HierarchicalModelSolver(
        model,
        outputfiles_basename='./out/',
        n_dims=ndim, n_params=nparams,
        n_live_points=nlive,
        verbose=True
    )

    ## Analysis
    # create analyzer object
    a = Analyzer(ndim, outputfiles_basename="./out/")

    # get a dictionary containing information about
    #   the logZ and its errors
    #   the individual modes and their parameters
    #   quantiles of the parameter posteriors
    stats = a.get_stats()

    # get the best fit (highest likelihood) point
    bestfit_params = a.get_best_fit()

    pos = bestfit_params['parameters']


    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(solution)
    print("Multinest best fit parameters:")
    print(pos)
