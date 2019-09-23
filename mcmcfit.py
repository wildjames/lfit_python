'''
This script will run the actual fitting procedure.
Requires the input file, and data files defined in that.
Supplied at the command line, via:

    python3 mcmcfit.py mcmc_input.dat

Can also notify the user of a completed chain with the --notify flag.
'''

import argparse
import multiprocessing as mp
import os
from pprint import pprint
from shutil import rmtree
from sys import exit

import configobj
import emcee
import numpy as np

import mcmc_utils as utils
from CVModel import construct_model, extract_par_and_key


# I need to wrap the model's ln_like, ln_prior, and ln_prob functions
# in order to pickle them :(
def ln_prior(param_vector, model):
    model.dynasty_par_vals = param_vector
    val = model.ln_prior()
    return val

def ln_prob(param_vector, model):
    model.dynasty_par_vals = param_vector
    val = model.ln_prob()
    return val

def ln_like(param_vector, model):
    model.dynasty_par_vals = param_vector
    val = model.ln_like()
    return val

if __name__ in '__main__':

    np.random.seed = 432

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

    args = parser.parse_args()
    input_fname = args.input
    dest = args.notify
    debug = args.debug

    if debug:
        if os.path.isdir("DEBUGGING"):
            rmtree("DEBUGGING")

    # I want to pre-check that the details have been supplied.
    if dest is not '':
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
    nburn          = int(input_dict['nburn'])
    nprod          = int(input_dict['nprod'])
    nthreads       = int(input_dict['nthread'])
    nwalkers       = int(input_dict['nwalkers'])
    ntemps         = int(input_dict['ntemps'])
    scatter_1      = float(input_dict['first_scatter'])
    scatter_2      = float(input_dict['second_scatter'])
    to_fit         = bool(int(input_dict['fit']))
    use_pt         = bool(int(input_dict['usePT']))
    double_burnin  = bool(int(input_dict['double_burnin']))
    comp_scat      = bool(int(input_dict['comp_scat']))

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
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
    if not to_fit:
        import plotCV

        plotCV.nxdraw(model)

        plotCV.plot_model(model, True, save=True, figsize=(11, 8), save_dir='Initial_figs/')

        exit()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  MCMC Chain sampler, handled by emcee.                      #
    #  The below plugs the above into emcee's relevant functions  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # How many parameters do I have to deal with?
    npars = len(model.dynasty_par_vals)

    print("\n\nThe MCMC has {:d} variables and {:d} walkers".format(
        npars, nwalkers))
    print("(It should have at least 2*npars, {:d} walkers)".format(2*npars))
    if nwalkers < 2*npars:
        exit()

    # p_0 is the initial position vector of the MCMC walker
    p_0 = model.dynasty_par_vals

    # We cant to scatter that, so create an array of our scatter values.
    # This will allow us to tweak the scatter value for each individual
    # parameter.
    p0_scatter_1 = np.array([scatter_1 for _ in p_0])

    # If comp_scat is asked for, each value wants to be scattered differently.
    # Some want more, some less.
    if comp_scat:
        # scatter factors. p0_scatter_1 will be multiplied by these:
        scat_fract = {
            'q':      2,
            'rwd':    1,
            'dphi':   0.2,
            'dFlux':  2,
            'sFlux':  2,
            'wdFlux': 2,
            'rsFlux': 2,
            'rdisc':  2,
            'ulimb':  1e-6,
            'scale':  3,
            'fis':    3,
            'dexp':   3,
            'phi0':   20,
            'az':     2,
            'exp1':   5,
            'exp2':   5,
            'yaw':    10,
            'tilt':   2,
        }

        for par_i, name in enumerate(model.dynasty_par_names):
            # Get the parameter of this parName, striping off the node encoding
            key, _ = extract_par_and_key(name)

            # Skip the GP params
            if key.startswith('ln'):
                continue

            # Multiply it by the relevant factor
            p0_scatter_1[par_i] *= scat_fract[key]

        # Create another array for second burn-in
        p0_scatter_2 = p0_scatter_1*(scatter_2/scatter_1)


    # Initialise the sampler. If we're using parallel tempering, do that.
    # Otherwise, don't.
    mp.set_start_method("forkserver")
    pool = mp.Pool(nthreads)

    if use_pt:
        print("MCMC using parallel tempering at {} levels, for {} total walkers.".format(ntemps, nwalkers*ntemps))
        # Create the initial ball of walker positions
        p_0 = utils.initialise_walkers_pt(p_0, p0_scatter_1,
                                          nwalkers, ntemps, ln_prior, model)
        # Create the sampler
        # TODO: The emcee PTSampler is deprecated. Use this package instead:
        # https://github.com/willvousden/ptemcee
        sampler = emcee.PTSampler(ntemps, nwalkers, npars,
                                  ln_like, ln_prior,
                                  loglargs=(model,),
                                  logpargs=(model,),
                                  pool=pool)
                                #   threads=nthreads)
    else:
        # Create the initial ball of walker positions
        p_0 = utils.initialise_walkers(p_0, p0_scatter_1, nwalkers,
                                       ln_prior, model)
        # Create the sampler
        sampler = emcee.EnsembleSampler(nwalkers, npars,
                                        ln_prob,
                                        args=(model,),
                                        pool=pool)
                                        # threads=nthreads)


    # Run the burnin phase
    print("\n\nExecuting the burn-in phase...")
    pos, prob, state = utils.run_burnin(sampler, p_0, nburn)

    # Do we want to do that again?
    if double_burnin:
        # If we wanted to run a second burn-in phase, then do. Scatter the
        # position about the first burn
        print("Executing the second burn-in phase")

        # Get the Get the most likely step of the first burn-in
        p_0 = pos[np.argmax(prob)]
        # And scatter the walker ball about that position
        p_0 = utils.initialise_walkers(p_0, p0_scatter_2, nwalkers,
                                       ln_prior, model)

        # Run that burn-in
        pos, prob, state = utils.run_burnin(sampler, p_0, nburn)


    # Now, reset the sampler. We'll use the result of the burn-in phase to
    # re-initialise it.
    sampler.reset()
    print("Starting the main MCMC chain. Probably going to take a while!")

    # Get the column keys. Otherwise, we can't parse the results!
    col_names = "walker_no " + ' '.join(model.dynasty_par_names) + ' ln_prob'

    if use_pt:
        # Run production stage of parallel tempered mcmc
        sampler = utils.run_ptmcmc_save(sampler, pos, nprod,
                                        "chain_prod.txt", col_names=col_names)

        # get chain for zero temp walker. Higher temp walkers DONT sample the
        # right landscape!
        # chain shape = (ntemps,nwalkers*nsteps,ndim)
        chain = sampler.flatchain[0, ...]
    else:
        # Run production stage of non-parallel tempered mcmc
        sampler = utils.run_mcmc_save(sampler, pos, nprod, state,
                                      "chain_prod.txt", col_names=col_names)

        # lnprob is in sampler.ln(probability) and is shape (nwalkers, nsteps)
        # sampler.chain has shape (nwalkers, nsteps, npars)

        # Collect results from all walkers
        chain = utils.flatchain(sampler.chain, npars, thin=10)

    with open('modparams.txt', 'w') as f:
        f.write("parName,mean,84th percentile,16th percentile\n")
        lolim, result, uplim = np.percentile(chain, [16, 50, 84], axis=0)
        labels = model.dynasty_par_names

        for n, m, u, l in zip(labels, result, uplim, lolim):
            s = "{} {} {} {}\n".format(n, m, u, l)
            f.write(s)
        f.write('\n')

    from plotCV import fit_summary
    fit_summary('chain_prod.txt', input_fname, destination=dest,
                automated=True)
