'''
Helper functions to aid the MCMC nuts and bolts.
'''
import warnings

import emcee
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
# lightweight progress bar
from tqdm import tqdm

try:
    import triangle
    # This triangle should have a method corner
    # There are two python packages with conflicting names
    getattr(triangle, "corner")
except (AttributeError, ImportError):
    # We want the other package
    # print("Could not import package, `triangle`, falling back on `corner.triangle`")
    import corner as triangle

try:
    import dask.dataframe as dd
    use_dask = True
except ImportError:
    use_dask = False

TINY = -np.inf



def fracWithin(pdf, val):
    return pdf[pdf >= val].sum()


def thumbPlot(chain, labels, **kwargs):
    seaborn.set(style='ticks')
    seaborn.set_style({"xtick.direction": "in","ytick.direction": "in"})
    fig = triangle.corner(chain, labels=labels, bins=50,
                          label_kwargs=dict(fontsize=18), **kwargs)
    return fig


def initialise_walkers(p, scatter, nwalkers, ln_prior, model):
    # Create starting ball of walkers with a certain amount of scatter
    p0 = emcee.utils.sample_ball(p, scatter*p, size=nwalkers)
    # Make initial number of invalid walkers equal to total number of walkers
    numInvalid = nwalkers
    print('Initialising walkers...')
    print('Number of walkers currently invalid:')
    # All invalid params need to be resampled
    while numInvalid > 0:
        # Create a mask of invalid params
        isValid = np.array([np.isfinite(ln_prior(p, model)) for p in p0])
        bad = p0[~isValid]
        # Determine the number of good and bad walkers
        nbad = len(bad)
        print(nbad)
        ngood = len(p0[isValid])
        # Choose nbad random rows from ngood walker sample
        replacement_rows = np.random.randint(ngood, size=nbad)
        # Create replacement values from valid walkers
        replacements = p0[isValid][replacement_rows]
        # Add scatter to replacement values
        replacements += 0.5*replacements*scatter*np.random.normal(
            size=replacements.shape)
        # Replace invalid walkers with new values
        p0[~isValid] = replacements
        numInvalid = len(p0[~isValid])
    return p0


def initialise_walkers_pt(p, scatter, nwalkers, ntemps, ln_prior, model):
    # Create starting ball of walkers with a certain amount of scatter
    p0 = np.array([emcee.utils.sample_ball(p, scatter*p, size=nwalkers) for
                   i in range(ntemps)])
    orig_shape = p0.shape
    # Re-shape p0 array
    p0 = p0.reshape(nwalkers*ntemps, len(p))
    # Make initial number of invalid walkers equal to total number of walkers
    numInvalid = nwalkers*ntemps
    print('Initialising walkers...')
    print('Number of walkers currently invalid:')
    # All invalid params need to be resampled
    while numInvalid > 0:
        # Create a mask of invalid params
        isValid = np.array([np.isfinite(ln_prior(p, model)) for p in p0])
        bad = p0[~isValid]
        # Determine the number of good and bad walkers
        nbad = len(bad)
        print(nbad)
        ngood = len(p0[isValid])
        # Choose nbad random rows from ngood walker sample
        replacement_rows = np.random.randint(ngood, size=nbad)
        # Create replacement values from valid walkers
        replacements = p0[isValid][replacement_rows]
        # Add scatter to replacement values
        replacements += 0.5*replacements*scatter*np.random.normal(
            size=replacements.shape)
        # Replace invalid walkers with new values
        p0[~isValid] = replacements
        numInvalid = len(p0[~isValid])
    p0 = p0.reshape(orig_shape)
    return p0


def run_burnin(sampler, startPos, nSteps, storechain=False, progress=True):
    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)
    for pos, prob, state in sampler.sample(startPos,
                                           iterations=nSteps,
                                           storechain=storechain):
        iStep += 1
        if progress:
            bar.update()
    if progress:
        bar.close()
    return pos, prob, state


def run_mcmc_save(sampler, startPos, nSteps, rState, file, col_names='',
                  progress=True, **kwargs):
    '''runs an MCMC chain with emcee, and saves steps to a file'''
    # open chain save file
    if file:
        with open(file, "w") as f:
            f.write(col_names)
            if col_names:
                f.write("\n")

    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)

    for pos, prob, state in sampler.sample(startPos,
                                           iterations=nSteps, rstate0=rState,
                                           storechain=True, **kwargs):

        iStep += 1
        if progress:
            bar.update()

        for k in range(pos.shape[0]):
            # loop over all walkers and append to file
            thisPos = pos[k]
            thisProb = prob[k]

            with open(file, 'a') as f:
                f.write("{0:4d} {1:s} {2:f}\n".format(
                    k, " ".join(map(str, thisPos)), thisProb))

    if progress:
        bar.close()
    return sampler


def run_ptmcmc_save(sampler, startPos, nSteps, file,
                    progress=True, col_names='', **kwargs):
    '''runs PT MCMC and saves zero temperature chain to a file'''
    if file:
        with open(file, "w") as f:
            f.write(col_names)
            if col_names:
                f.write("\n")

    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)

    for pos, prob, like in sampler.sample(startPos,
                                          iterations=nSteps,
                                          storechain=True, **kwargs):

        iStep += 1
        if progress:
            bar.update()
        # pos is shape (ntemps, nwalkers, npars)
        # prob is shape (ntemps, nwalkers)
        # loop over all walkers for first temp and append to file
        zpos = pos[0, ...]
        zprob = prob[0, ...]

        for k in range(zpos.shape[0]):
            thisPos = zpos[k]
            thisProb = zprob[k]

            with open(file, 'a') as f:
                f.write("{0:4d} {1:s} {2:f}\n".format(k, " ".join(
                map(str, thisPos)), thisProb))

    if progress:
        bar.close()
    return sampler


def flatchain(chain, npars=None, nskip=0, thin=1):
    '''flattens a chain (i.e collects results from all walkers),
    with options to skip the first nskip parameters, and thin the chain
    by only retrieving a point every thin steps - thinning can be useful when
    the steps of the chain are highly correlated'''
    if npars is None:
        npars = chain.shape[2]
    return chain[:, nskip::thin, :].reshape((-1, npars))


def readchain(file, **kwargs):
    '''Reads in the chain file in a single thread.
    Returns the chain in the shape (nwalkers, nprod, npars)
    '''
    data = pd.read_csv(file, header=0, compression=None,
                       delim_whitespace=True, **kwargs)
    data = np.array(data)

    # Figure out what shape the result should have.
    nwalkers = int(np.amax(data[:, 0])+1)
    nprod = int(data.shape[0] / nwalkers)
    npars = int(data.shape[1] - 1)

    # empty array to fill. Make it nans to be safe?
    chain = np.zeros((nwalkers, nprod, npars))
    chain[:, :, :] = np.nan

    for i in range(nwalkers):
        index = np.where(data[:, 0] == float(i))
        chain[i] = data[index, 1:]

    return chain


def readchain_dask(file, **kwargs):
    '''Reads in the chain file using threading.
    Returns the chain in the shape (nwalkers, nprod, npars).'''

    if not use_dask:
        return readchain(file, **kwargs)

    data = dd.io.read_csv(file, engine='c', header=0, compression=None,
                          na_filter=False, delim_whitespace=True, **kwargs)
    data = data.compute()
    data = np.array(data)

    # Figure out what shape the result should have.
    nwalkers = int(np.amax(data[:, 0])+1)
    nprod = int(data.shape[0] / nwalkers)
    npars = int(data.shape[1] - 1)

    # empty array to fill. Make it nans to be safe?
    chain = np.zeros((nwalkers, nprod, npars))
    chain[:, :, :] = np.nan

    for i in range(nwalkers):
        index = np.where(data[:, 0] == float(i))
        chain[i] = data[index, 1:]

    return chain


def readflatchain(file):
    data = pd.read_csv(file, header=None, compression=None,
                       delim_whitespace=True)
    data = np.array(data)
    return data


def plotchains(chain, npar, alpha=0.2):
    nwalkers, nsteps, npars = chain.shape
    fig = plt.figure()
    for i in range(nwalkers):
        plt.plot(chain[i, :, npar], alpha=alpha, color='k')
    return fig


def GR_diagnostic(sampler_chain):
    '''Gelman & Rubin check for convergence.'''
    m, n, ndim = np.shape(sampler_chain)
    R_hats = np.zeros((ndim))
    samples = sampler_chain[:, :, :].reshape(-1, ndim)
    for i in range(ndim):  # iterate over parameters

        # Define variables
        chains = sampler_chain[:, :, i]

        flat_chain = samples[:, i]
        psi_dot_dot = np.mean(flat_chain)
        psi_j_dot = np.mean(chains, axis=1)
        psi_j_t = chains

        # Calculate between-chain variance
        between = sum((psi_j_dot - psi_dot_dot)**2) / (m - 1)

        # Calculate within-chain variance
        inner_sum = np.sum(np.array([(psi_j_t[j, :] - psi_j_dot[j])**2
                                     for j in range(m)]), axis=1)
        outer_sum = np.sum(inner_sum)
        W = outer_sum / (m*(n-1))

        # Calculate sigma
        sigma2 = (n-1)/n * W + between

        # Calculate convergence criterion (potential scale reduction factor)
        R_hats[i] = (m + 1)*sigma2/(m*W) - (n-1)/(m*n)
    return R_hats


def rebin(xbins, x, y, e=None, weighted=True, errors_from_rms=False):
    digitized = np.digitize(x, xbins)
    xbin = []
    ybin = []
    ebin = []
    for i in range(0, len(xbins)):
            bin_y_vals = y[digitized == i]
            bin_x_vals = x[digitized == i]
            if e is not None:
                bin_e_vals = e[digitized == i]
            if weighted:
                if e is None:
                    raise Exception('Cannot compute weighted mean without errors')
                weights = 1.0/bin_e_vals**2
                xbin.append(np.sum(weights*bin_x_vals) / np.sum(weights))
                ybin.append(np.sum(weights*bin_y_vals) / np.sum(weights))
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append(np.sqrt(1.0/np.sum(weights)))
            else:
                xbin.append(bin_x_vals.mean())
                ybin.append(bin_y_vals.mean())
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append(np.sqrt(np.sum(bin_e_vals**2)) / len(bin_e_vals))
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    return (xbin, ybin, ebin)
