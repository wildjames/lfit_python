'''This script will run the actual fitting procedure.
Requires the input file, and data files defined in that.
Supplied at the command line, via:

    python3 mcmcfit.py mcmc_input.dat
'''

from pprint import pprint
import sys
import numpy as np
import configobj

import emcee

from model import Eclipse, Band, LCModel, Lightcurve
from mcmc_utils import Param
import mcmc_utils as utils


if __name__ in '__main__':
    input_file = sys.argv[1]
    input_dict = configobj.ConfigObj(input_file)

    # Read in information about mcmc, neclipses, use of complex/GP etc.
    nburn = int(input_dict['nburn'])
    nprod = int(input_dict['nprod'])
    nthreads = int(input_dict['nthread'])
    nwalkers = int(input_dict['nwalkers'])
    ntemps = int(input_dict['ntemps'])
    scatter_1 = float(input_dict['first_scatter'])
    scatter_2 = float(input_dict['second_scatter'])
    to_fit = int(input_dict['fit'])
    is_complex = bool(int(input_dict['complex']))
    use_gp = bool(int(input_dict['useGP']))
    use_pt = bool(int(input_dict['usePT']))
    corner = bool(int(input_dict['corner']))
    double_burnin = bool(int(input_dict['double_burnin']))
    comp_scat = bool(int(input_dict['comp_scat']))

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
    try:
        neclipses = int(input_dict['neclipses'])
    except KeyError:
        neclipses = -1

    if use_gp:
        # TODO: Impliment the GP version of the Eclipses.
        # Read in GP params using fromString function from mcmc_utils.py
        ampin_gp = Param.fromString('ampin_gp', input_dict['ampin_gp'])
        ampout_gp = Param.fromString('ampout_gp', input_dict['ampout_gp'])
        tau_gp = Param.fromString('tau_gp', input_dict['tau_gp'])

    # Start by creating the overall Model. Gather the parameters:
    core_par_names = ['rwd', 'dphi', 'q']
    core_pars = [Param.fromString(name, s) for name, s in input_dict.items()
                 if name in core_par_names]
    # and make the model object with no children
    model = LCModel('core', core_pars)

    # Collect the bands and their params. Add them total model.
    band_par_names = ['wdFlux', 'rsFlux']

    # Get a sub-dict of only band parameters
    band_dict = {}
    for key, string in input_dict.items():
        if np.any([key.startswith(par) for par in band_par_names]):
            band_dict[key] = string

    # Get a set of the bands we have in the input file
    defined_bands = [key.split('_')[-1] for key in band_dict]
    defined_bands = set(defined_bands)
    print("I found definitions of the following bands: {}".format(
        defined_bands))

    for band in defined_bands:
        band_pars = []
        for key, string in band_dict.items():
            if key.endswith("_{}".format(band)):
                name = key.split("_")[0]
                band_pars.append(Param.fromString(name, string))

        band = Band(band, band_pars, parent=model)

    # These are the entries to ignore.
    descriptors = ['file', 'plot', 'band']
    descriptors += band_par_names
    descriptors += core_par_names
    complex_desc = ['exp1', 'exp2', 'yaw', 'tilt']
    if not is_complex:
        print("Using the complex BS model. ")
        descriptors.extend(complex_desc)

    ecl_i = -1
    while True:
        ecl_i += 1

        # The user can limit the number if eclipses to fit.
        if ecl_i == neclipses:
            break

        # Collect this eclipses' parameters.
        ecl_exists = [key.endswith("_{}".format(ecl_i)) for key in input_dict]
        if np.any(ecl_exists):
            # Initialise this eclipses's stuff.
            eclipse_pars = []

            # What band are we going to be looking at?
            band = input_dict['band_{}'.format(ecl_i)]
            # Retrieve the band object, so we can request it as a parent later
            band = model.search_Node('Band', band)

            # print("Eclipse {} belongs to the {}".format(ecl_i, band.name))

            # Loop through the input dict, searching for keys that have a tail
            # matching this eclipse
            for key, string in input_dict.items():
                if key.endswith("_{}".format(ecl_i)):

                    # Make sure we don't create a parameter from any of the
                    # descriptors. Check none of the forbidden keys are in this
                    test = [d in key for d in descriptors]
                    if np.any(test):
                        continue

                    # Construct the name of the parameter,
                    # i.e. strip off the tail code
                    name = key.replace("_{}".format(ecl_i), '')

                    # print("{} has the parameter {}, calling it {}".format(
                    #     ecl_i, key, name))

                    # Make the Param object from the string, and add it to
                    # our list of pars.
                    param = Param.fromString(name, string)
                    eclipse_pars.append(param)

            # Read in the datafile associated with this eclipse
            fname = input_dict['file_{}'.format(ecl_i)]
            eclipse_data = Lightcurve.from_calib(fname)

            # Trim the eclipse down to our desired range.
            start = float(input_dict['phi_start'])
            end = float(input_dict['phi_end'])
            eclipse_data.trim(start, end)

            # Construct the eclipse object
            Eclipse(eclipse_data, is_complex, str(ecl_i), eclipse_pars,
                    parent=band)

            # print("\n\n")
        else:
            break

    neclipses = ecl_i
    print("Fitting {} eclipses.".format(neclipses))

    #################################
    # The model is now fully built. #
    #################################

    print("\nStructure:")
    pprint(model.structure)
    # Get the model's graph
    # model.draw()
    # model.plot_data(save=True)

    print("\n\nInitial guess has a chisq of {:.3f},".format(
        model.chisq(False)))
    print("a ln_prob of {:.3f},".format(model.ln_prob(verbose=True)))
    print("and a ln_prior of {:.3f}".format(model.ln_prior()))
    print()
    if np.isinf(model.ln_prior()):
        print("ERROR: Starting position violates priors!")
        print("Offending parameters are:")

        pars, names = model.__get_descendant_Params__()
        for par, name in zip(pars, names):
            print("{:>15s}_{:<5s}: Valid?: {}".format(
                par.name, name, par.isValid))

            if not par.isValid:
                print("  -> {}_{}".format(par.name, name))
        print("\n\n")

        # Calculate ln_prior verbosley, for the user's benefit
        model.ln_prior(verbose=True)

    if not to_fit:
        model.draw()
        print("Model parameters:")
        with open('model_parameters.txt', 'w') as file_obj:
            # Evaluate the final model.
            print("\n\nFor this model;\n")
            print("  Chisq             = {:.3f}".format(model.chisq()))
            file_obj.write("  Chisq             = {:.3f}\n".format(
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
        # model.plot_data(save=True)
        for iecl in range(neclipses):
            eclipse = model.search_Node('Eclipse', str(iecl))
            eclipse.plot_data(save=True)

        exit()

    ##############################################################
    # MCMC Chain sampler, handled by emcee.                      #
    # The below plugs the above into emcee's relevant functions. #
    ##############################################################

    # How many parameters do I have to deal with?
    npars = len(model.par_val_list)

    print("The MCMC has {:d} variables and {:d} walkers".format(
        npars, nwalkers))
    print("(It should have at least 2*npars, {:d} walkers)".format(2*npars))

    # p_0 is the initial position vector of the MCMC walker
    p_0 = model.par_val_list

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

        for par_i, name in enumerate(model.par_names):
            # Get the parameter of this parName, striping off the node encoding
            key = name.split("_")[0]

            # Multiply it by the relevant factor
            p0_scatter_1[par_i] *= scat_fract[key]

        # Create another array for second burn-in
        p0_scatter_2 = p0_scatter_1*(scatter_2/scatter_1)

    # I need to wrap the model's ln_like, ln_prior, and ln_prob functions
    # in order to pickle them :(
    def ln_prior(par_val_list):
        model.par_val_list = par_val_list
        return model.ln_prior()

    def ln_prob(par_val_list):
        model.par_val_list = par_val_list
        return model.ln_prob()

    def ln_like(par_val_list):
        model.par_val_list = par_val_list
        return model.ln_like()

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
    print("Executing the burn-in phase...")
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

    col_names = model.par_names

    if use_pt:
        # Run production stage of parallel tempered mcmc
        sampler = utils.run_ptmcmc_save(sampler, pos, nprod,
                                        "chain_prod.txt", col_names)

        # get chain for zero temp walker. Higher temp walkers DONT sample the
        # right landscape!
        # chain shape = (ntemps,nwalkers*nsteps,ndim)
        chain = sampler.flatchain[0, ...]

        # Save flattened chain
        np.savetxt('chain_flat.txt', chain, delimiter=' ')
    else:
        # Run production stage of non-parallel tempered mcmc
        sampler = utils.run_mcmc_save(sampler, pos, nprod, state,
                                      "chain_prod.txt", col_names)

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
        for i, name in enumerate(model.par_names):
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
        model.par_val_list = chain_results

        # Evaluate the final model. Save to file.
        print("\n\nFor this model;\n")
        print("  Chisq             = {:.3f}".format(model.chisq()))
        file_obj.write("  Chisq             = {:.3f}\n".format(
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
