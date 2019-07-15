'''This script will run the actual fitting procedure.
Requires the input file, and data files defined in that.
Supplied at the command line, via:

    python3 mcmcfit.py mcmc_input.dat
'''

from pprint import pprint
import sys
import numpy as np
import configobj
import matplotlib.pyplot as plt

import emcee

from model import Lightcurve, Eclipse, Band, LCModel, GPLCModel
from mcmc_utils import Param
import mcmc_utils as utils

from func_timeout import func_timeout, FunctionTimedOut

def extract_par_and_key(key):
    '''As stated. For example,
    extract_par_and_key("wdFlux_long_complex_key_label)
    >>> ("wdFlux", "long_complex_key_label")
    '''

    if key.startswith("ln_"):
        key = key.split("_")

        par = "_".join(key[:3])
        label = "_".join(key[3:])

    else:
        par, label = key.split('_')[0], '_'.join(key.split('_')[1:])

    return par, label

def construct_model(input_file):
    '''Takes an input filename, and parses it into a model tree.
    Returns that model tree. '''

    input_dict = configobj.ConfigObj(input_file)

    for key, item in input_dict.items():
        if key in ['ampin_gp', 'ampout_gp', 'tau_gp']:
            print("OI! Don't use {0}, use ln_{0}!!".format(key))
            input_dict['ln_'+key] = item
            input("As punishment, you have to hit enter to continue:\n> ")

    # Do we use the complex model? Do we use the GP?
    is_complex = bool(int(input_dict['complex']))
    use_gp = bool(int(input_dict['useGP']))

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
    try:
        neclipses = int(input_dict['neclipses'])
    except KeyError:
        # Read in all the available eclipses
        neclipses = 9e99

    # # # # # # # # # # # # # # # # #
    # Get the initial model setup # #
    # # # # # # # # # # # # # # # # #

    # Start by creating the overall Model. Gather the parameters:
    if use_gp:
        core_par_names = GPLCModel.node_par_names
        core_pars = [Param.fromString(name, input_dict[name])
                     for name in core_par_names]

        # and make the model object with no children
        model = GPLCModel('core', core_pars)
    else:
        core_par_names = LCModel.node_par_names
        core_pars = [Param.fromString(name, input_dict[name])
                     for name in core_par_names]

        # and make the model object with no children
        model = LCModel('core', core_pars)

    # # # # # # # # # # # # # # # # #
    # # # Now do the band names # # #
    # # # # # # # # # # # # # # # # #

    # Collect the bands and their params. Add them total model.
    band_par_names = Band.node_par_names

    # Use the Eclipse class to find the parameters we're interested in
    ecl_pars = Eclipse.node_par_names
    if is_complex:
        ecl_pars += ('exp1', 'exp2', 'yaw', 'tilt')

    # I care about the order in which eclipses and bands are defined.
    # Collect that order here.
    defined_bands = []
    defined_eclipses = []
    with open(input_file, 'r') as input_file_obj:
        for line in input_file_obj:
            line = line.strip().split()

            if len(line):
                key = line[0]

                # Check that the key starts with any of the band pars
                if np.any([key.startswith(par) for par in band_par_names]):
                    # Strip off the variable part, and keep only the label
                    _, key = extract_par_and_key(key)
                    if key not in defined_bands:
                        defined_bands.append(key)

                # Check that the key starts with any of the eclipse pars
                if np.any([key.startswith(par) for par in ecl_pars]):
                    # Strip off the variable part, and keep only the label
                    _, key = extract_par_and_key(key)
                    if key not in defined_eclipses:
                        defined_eclipses.append(key)

    # Collect the band params into their Band objects.
    for label in defined_bands:
        band_pars = []

        # Build the Param objects for this band
        for par in band_par_names:
                # Construct the parameter key and retrieve the string
                key = "{}_{}".format(par, label)
                string = input_dict[key]

                # Make the Param object, and save it
                band_pars.append(Param.fromString(par, string))

        # Define the band as a child of the model.
        Band(label, band_pars, parent=model)

    # # # # # # # # # # # # # # # # #
    # # Finally, get the eclipses # #
    # # # # # # # # # # # # # # # # #

    lo = float(input_dict['phi_start'])
    hi = float(input_dict['phi_end'])

    for label in defined_eclipses[:neclipses]:
        # Get the list of parameters, and their priors
        params = []
        for par in ecl_pars:
            key = "{}_{}".format(par, label)
            param = Param.fromString(par, input_dict[key])

            params.append(param)

        # Get the observational data
        lc_fname = input_dict['file_{}'.format(label)]
        lc = Lightcurve.from_calib(lc_fname)
        lc.trim(lo, hi)

        # Get the band object that this eclipse belongs to
        my_band = input_dict['band_{}'.format(label)]
        my_band = model.search_Node('Band', my_band)

        Eclipse(lc, is_complex, label, params, parent=my_band)

    # Make sure that all the model's Band have eclipses. Otherwise, prune them
    model.children = [band for band in model.children if len(band.children)]

    return model


if __name__ in '__main__':

    try:
        input_fname = sys.argv[1]
    except:
        print("No input file supplied!!")
        exit()

    model = construct_model(input_fname)

    print("\nStructure:")
    pprint(model.structure)
    # Get the model's graph
    model.draw()

    # I need to wrap the model's ln_like, ln_prior, and ln_prob functions
    # in order to pickle them :(
    def ln_prior(param_vector):
            try:
            model.dynasty_par_vals = param_vector
            val = func_timeout(
                60,
                model.ln_prior
            )
        except FunctionTimedOut as e:
            print("Model Parameters:")
            model.report()
            val = -np.inf
            print(e)
        return val

    def ln_prob(param_vector):
        try:
            model.dynasty_par_vals = param_vector
            val = func_timeout(
                60,
                model.ln_prob
            )
        except FunctionTimedOut as e:
            print("Model Parameters:")
            model.report()
            val = -np.inf
            print(e)
        return val

    def ln_like(param_vector):
        try:
            model.dynasty_par_vals = param_vector
            val = func_timeout(
                60,
                model.ln_like
            )
        except FunctionTimedOut as e:
            print("Model Parameters:")
            model.report()
            val = -np.inf
            print(e)
        return val

    input_dict = configobj.ConfigObj(input_fname)

    for key, item in input_dict.items():
        if key in ['ampin_gp', 'ampout_gp', 'tau_gp']:
            input_dict['ln_'+key] = item

    # Read in information about mcmc
    nburn = int(input_dict['nburn'])
    nprod = int(input_dict['nprod'])
    nthreads = int(input_dict['nthread'])
    nwalkers = int(input_dict['nwalkers'])
    ntemps = int(input_dict['ntemps'])
    scatter_1 = float(input_dict['first_scatter'])
    scatter_2 = float(input_dict['second_scatter'])
    to_fit = int(input_dict['fit'])
    use_pt = bool(int(input_dict['usePT']))
    use_gr = bool(int(input_dict['gelman_rubin_burn']))
    gr_thresh = float(input_dict['gelman_rubin_thresh'])
    double_burnin = bool(int(input_dict['double_burnin']))
    comp_scat = bool(int(input_dict['comp_scat']))

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
    try:
        neclipses = int(input_dict['neclipses'])
    except KeyError:
        neclipses = 0
        while model.search_Node('Eclipse', str(neclipses)) is not None:
            neclipses += 1
        print("The model has {} eclipses.".format(neclipses))

    # Wok out how many degrees of freedom we have in the model
    eclipses = model.search_node_type('Eclipse')
    # How many data points do we have?
    dof = np.sum([ecl.lc.x.size for ecl in eclipses])
    # Subtract a DoF for each variable
    dof -= len(model.dynasty_par_names)
    # Subtract one DoF for the fit
    dof -= 1

    print("\n\nInitial guess has a chisq of {:.3f} ({:d} D.o.F.).".format(model.chisq(), dof))
    print("\nFrom the wrapper functions, we get;")
    pars = model.dynasty_par_vals
    print("a ln_prior of {:.3f}".format(ln_prior(pars)))
    print("a ln_like of {:.3f}".format(ln_like(pars)))
    print("a ln_prob of {:.3f}".format(ln_prob(pars)))
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
    model.plot_data()


    if not to_fit:
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
    if use_pt:
        # Create the initial ball of walker positions
        p_0 = utils.initialise_walkers_pt(p_0, p0_scatter_1,
                                          nwalkers, ntemps, ln_prior)
        # Create the sampler
        sampler = emcee.PTSampler(ntemps, nwalkers, npars,
                                  ln_like, ln_prior, threads=nthreads)
    else:
        # Create the initial ball of walker positions
        p_0 = utils.initialise_walkers(p_0, p0_scatter_1, nwalkers,
                                       ln_prior)
        # Create the sampler
        sampler = emcee.EnsembleSampler(nwalkers, npars,
                                        ln_prob, threads=nthreads)


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
        p_0 = utils.initialise_walkers(p_0, p0_scatter_2, nwalkers, ln_prior)

        # Run that burn-in
        pos, prob, state = utils.run_burnin(sampler, p_0, nburn)


    # Now, reset the sampler. We'll use the result of the burn-in phase to
    # re-initialise it.
    sampler.reset()
    print("Starting the main MCMC chain. Probably going to take a while!")

    # Get the column keys. Otherwise, we can't parse the results!
    col_names = ','.join(model.dynasty_par_names) + ',ln_prob'

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


    # Save flattened chain
    np.savetxt('chain_flat.txt', chain, delimiter=' ')


    print("Model parameters:")
    with open('model_parameters.txt', 'w') as file_obj:
        file_obj.write('parName,mean,84th percentile,16th percentile\n')

        # Save the results for later
        chain_results = []
        for i, name in enumerate(model.dynasty_par_names):
            # Get the results of each parameter
            par = chain[:, i]
            lolim, best, uplim = np.percentile(par, [16, 50, 84])

            # Save the middle value, for later setting to the model
            chain_results.append(best)

            # Report
            print("{:>15s}: {:>7.3f} +{:<7.3f} -{:<7.3f}".format(
                name, best, uplim, lolim))
            file_obj.write("{},{},{},{}\n".format(name, best, uplim, lolim))

        # Set the model parameters to the results of the chain.
        model.dynasty_par_vals = chain_results

        # Evaluate the final model. Save to file.
        print("\n\nFor this model;\n")
        print("  Chisq             = {:.3f}".format(model.chisq()))
        file_obj.write("\n\n  Chisq             = {:.3f}\n".format(
            model.chisq()))

        print("  ln prior          = {:.3f}".format(model.ln_prior()))
        file_obj.write("  ln prior          = {:.3f}\n".format(
            model.ln_prior()))

        print("  ln like           = {:.3f}".format(model.ln_like()))
        file_obj.write("  ln like           = {:.3f}\n".format(
            model.ln_like()))

        print("  ln prob           = {:.3f}".format(model.ln_prob()))
        file_obj.write("  ln prob           = {:.3f}\n".format(
            model.ln_prob()))

    # Plot the data and the final fit.
    model.plot_data(save=True)
