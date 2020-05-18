import multiprocessing as mp
import os
# import warnings

import ptemcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from astropy.stats import sigma_clipped_stats

from mcmc_utils import (flatchain, initialise_walkers_pt, readchain,
                        readflatchain, run_burnin, run_ptmcmc_save, thumbPlot)
from model import Param

# Location of data tables
ROOT, _ = os.path.split(__file__)

# Define helper functions for the MCMC fit
def ln_prior(vect, model):
    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(model.npars):
        model[i] = vect[i]

    lnp = 0.0

    # teff, (usually uniform between allowed range - 6 to 90,000)
    param = model.teff
    lnp += param.prior.ln_prob(param.currVal)

    # logg, uniform between allowed range (7.01 to 8.99), or Gaussian from constraints
    param = model.logg
    lnp += param.prior.ln_prob(param.currVal)

    # Parallax, gaussian prior of the gaia value.
    param = model.plax
    lnp += param.prior.ln_prob(param.currVal)

    # reddening, cannot exceed galactic value (should estimate from line of sight)
    # https://irsa.ipac.caltech.edu/applications/DUST/
    param = model.ebv
    lnp += param.prior.ln_prob(param.currVal)
    return lnp

def ln_likelihood(vect, model):
    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(model.npars):
        model[i] = vect[i]

    errs = []
    for obs in model.obs_fluxes:
        errs.append(obs.err)
    errs = np.array(errs)

    chisq = model.chisq()

    return -0.5*(np.sum(np.log(2.0*np.pi*errs**2)) + chisq)

def ln_prob(vect, model):
    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(model.npars):
        model[i] = vect[i]

    lnp = ln_prior(vect, model)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(vect, model)
    else:
        return lnp

def parseInput(file):
    ''' reads in a file of key = value entries and returns a dictionary'''
    # Reads in input file and splits it into lines
    blob = np.loadtxt(file, dtype='str', delimiter='\n')
    input_dict = {}
    for line in blob:
        # Each line is then split at the equals sign
        k, v = line.split('=')
        input_dict[k.strip()] = v.strip()
    return input_dict

def sdss2kg5(g, r):
    KG5 = g  - 0.2240*(g-r)**2 - 0.3590*(g-r) + 0.0460
    return KG5
sdss2kg5_vect = np.vectorize(sdss2kg5)

def sdssmag2flux(mag):
    return 3631e3*np.power(10, -0.4*mag)


class wdModel():
    '''wd model
    can be passed to MCMC routines for calculating model and chisq, and prior prob

    also behaves like a list, of the current values of all parameters
    this enables it to be seamlessly used with emcee

    Note that parallax should be provided in MILLIarcseconds.'''

    # arguments are Param objects (see mcmc_utils)
    def __init__(self, teff, logg, plax, ebv, fluxes, debug=False):
        self.DEBUG = debug

        self.teff = teff
        self.logg = logg
        self.plax = plax
        self.ebv = ebv

        # initialise list bit of object with parameters
        self.variables = [self.teff, self.logg, self.plax, self.ebv]

        # Observed data
        self.obs_fluxes = fluxes

        # Teff, logg to model SDSS magnitudes tables
        self.DA = pd.read_csv(os.path.join(ROOT, 'Bergeron/Table_DA'), delim_whitespace=True, skiprows=0, header=1)
        self.loggs = np.unique(self.DA['log_g'])
        self.teffs = np.unique(self.DA['Teff'])
        self.nlogg = len(self.loggs)
        self.nteff = len(self.teffs)

        # Extinction coefficient dictionary
        self.extinction_coefficients = {
            'u_s': 5.155,
            'g_s': 3.793,
            'r_s': 2.751,
            'i_s': 2.086,
            'z_s': 1.479,
            'kg5': 3.5,
        }

    # these routines are needed so object will behave like a list
    def __getitem__(self, ind):
        return self.variables[ind].currVal

    def __setitem__(self, ind, val):
        self.variables[ind].currVal = val

    def __delitem__(self, ind):
        self.variables.remove(ind)

    def __len__(self):
        return len(self.variables)

    def insert(self, ind, val):
        self.variables.insert(ind, val)

    @property
    def npars(self):
        return len(self.variables)

    @property
    def dist(self):
        if self.plax.currVal <= 0.0:
            if self.DEBUG:
                print("Warning! Parallax, {} <= 0.0".format(self.plax.currVal))
            return np.inf
        else:
            return 1000./self.plax.currVal

    def __str__(self):
        return "<wdModel with teff {:.3f} || logg {:.3f} || plax {:.3f} || ebv {:.3f} || Debugging {}>".format(self.teff.currVal, self.logg.currVal, self.plax.currVal, self.ebv.currVal, self.DEBUG)

    def gen_absolute_mags(self):
        '''
        Take my Teff and logg, and interpolate a model absolute magnitude corresponding to each of my observations.
        Returns a magnitude observed in Super SDSS, with HCAM, on the GTC.
        '''
        t, g = self.teff.currVal, self.logg.currVal

        abs_mags = []
        for obs in self.obs_fluxes:
            if self.DEBUG:
                print("\n Interpolating Bergeron model magnitude, with observing band (HCAM, Super) {}".format(obs.band))
            band = obs.band

            # Get the Bergeron magnitude for this Teff, logg in this band on the GTC/HCAM
            if band == 'kg5':
                if self.DEBUG:
                    print("This is a kg5 band, so I will infer the model magnitude from g and r")
                # KG5 mags must be inferred
                gmags = np.array(self.DA['g_s'])
                rmags = np.array(self.DA['r_s'])

                z = sdss2kg5_vect(gmags, rmags)
                z = z.reshape((self.nlogg,self.nteff))

            else:
                z = np.array(self.DA[band])
                z = z.reshape((self.nlogg,self.nteff))

            # cubic bivariate spline interpolation on <z>
            func = interp.RectBivariateSpline(self.loggs,self.teffs,z,kx=3,ky=3)
            mag = func(g,t)[0,0]

            abs_mags.append(mag)

            if self.DEBUG:
                print("Interpolated a magnitude of {:.3f} from Bergeron table".format(func(g,t)[0,0]))
                print("After applying color corrections, the magnitude is {:.3f}".format(mag))
                print("\n------------------------------------\n")

        return np.array(abs_mags)

    def gen_apparent_mags(self):
        '''Apply distance modulus and extinction to my generated magnitudes.
        Observed above the atmosphere, from earth, in super SDSS, with HCAM, on the GTC.
        '''
        # Get absolute magnitudes
        abs_mags = self.gen_absolute_mags()

        # Apply distance modulus
        d = self.dist
        dmod = 5.0*np.log10(d/10.0)

        if self.DEBUG:
            print("Model holds parallax = {:.3f}".format(self.plax.currVal))
            print("            distance = {:.3f}".format(d))
            print("    Distance modulus = {:.3f}".format(dmod))
        mags = abs_mags + dmod

        # Apply extinction coefficients. At the same time, collect errors
        for i, obs in enumerate(self.obs_fluxes):
            band = obs.band
            ex = self.extinction_coefficients[band]
            ex *= self.ebv.currVal

            if self.DEBUG:
                print("Band {}: Extinction: {:.3f}".format(band, ex))

            mags[i] += ex

        if self.DEBUG:
            print("Got apparent magnitudes.")
            for obs, mag in zip(self.obs_fluxes, mags):
                band = obs.band
                print(" {}: {:.3f}".format(band, mag))

        if self.DEBUG:
            print("\n------------------------------------\n")

        return mags

    def chisq(self):
        '''Set internal teff and logg, calculate the model WD fluxes for that,
        and compute chisq from the obervations'''
        mags = self.gen_apparent_mags()
        flux_errs = np.zeros_like(mags)
        fluxes = sdssmag2flux(mags)

        # collect errors
        for i, obs in enumerate(self.obs_fluxes):
            flux_errs[i] = obs.err

        # Collect observed GTC/HCAM magnitudes.
        teff, logg = self.teff.currVal, self.logg.currVal
        obs_mags = np.array([obs.bergeron_mag(teff, logg) for obs in self.obs_fluxes])
        # Convert to fluxes
        obs_fluxes = sdssmag2flux(obs_mags)

        # Chisquared
        chisq = np.power(((fluxes - obs_fluxes)/flux_errs), 2)
        chisq = np.sum(chisq)

        return chisq


class Flux(object):
    BANDS = ['u', 'g', 'r', 'i', 'z']

    LAMBDAS = {
        'u':    355.7,
        'g':    482.5,
        'r':    626.1,
        'i':    767.2,
        'z':    909.7,
        'kg5':  507.5,
        'u_s':  352.6,
        'g_s':  473.2,
        'r_s':  619.9,
        'i_s':  771.1,
        'z_s':  915.6,
        'us':  352.6,
        'gs':  473.2,
        'rs':  619.9,
        'is':  771.1,
        'zs':  915.6,
    }

    def __init__(self, val, err, band, syserr=0.03, debug=False):
        self.DEBUG = debug

        self.flux = val
        self.err = np.sqrt(err**2 + (val*syserr)**2)

        # This is the actual band observed with.
        self.orig_band = band

        self.cent_lambda = self.LAMBDAS[band]

        self.mag = 2.5*np.log10(3631e3 / self.flux)
        self.magerr = 2.5*0.434*(self.err / self.flux)


        ## Get the correction I need from the user
        # Valid telescopes, and their instruments
        instruments = {
            'ntt': ['ucam'],
            'gtc': ['hcam'],
            'wht': ['hcam', 'ucam'],
            'tnt': ['uspec'],
            'none': ['']
        }
        filters = {
            'ucam': ['u', 'g', 'r', 'i', 'z', 'u_s', 'g_s', 'r_s', 'i_s', 'z_s'],
            'hcam': ['u', 'g', 'r', 'i', 'z', 'u_s', 'g_s', 'r_s', 'i_s', 'z_s'],
            'uspec': ['u', 'g', 'r', 'i', 'z']
        }

        if 'y' in input("Apply correction? y/n: ").lower():
            self.correct_me = True
            print("\nWhat telescope was band {} observed with? {}".format(band, instruments.keys()))
            tel = input("> ")
            while tel not in instruments.keys():
                print("\nThat telescope is not supported! ")
                tel = input("> ")
            if tel == 'none':
                print("Not performing a color correction on this filter")
                self.correct_me = False
                return

            print("What instrument was band {} observed with? {}".format(band, instruments[tel]))
            inst = input("> ")
            while inst not in instruments[tel]:
                inst = input("That is not a valid instrument for this telescope!\n> ")

            print("\nWhat filter was used for this observation? Labelled as {}".format(band))
            print("Options: {}".format(filters[inst]))
            filt = input("> ")
            while filt not in filters[inst]:
                print("That is not available on that instrument!")
                filt = input("Enter a filter: ")

            print("This is a 'super' filter, so I need to do some colour corrections. Using the column {0}, which is the magnitude in (HCAM/GTC/super filter - {0})".format(filt))



            # Save the correction table for this band here
            correction_table_fname = 'color_corrections_HCAM-GTC-super_minus_{}_{}.csv'.format(tel, inst)
            script_loc = os.path.split(__file__)[0]
            correction_table_fname = os.path.join(script_loc, 'color_correction_tables', correction_table_fname)
            print("Table is stored at {}".format(correction_table_fname))

            # Create an interpolater for the color corrections
            correction_table = pd.read_csv(correction_table_fname)

            # Model table teffs
            teffs = np.unique(correction_table['Teff'])
            loggs = np.unique(correction_table['logg'])

            # Color Correction table contains regular - super color, sorted by Teff, then logg
            corrections = np.array(correction_table[filt])
            corrections = corrections.reshape(len(teffs),len(loggs))

            self.orig_band = filt
            self.correction_func = interp.RectBivariateSpline(teffs, loggs, corrections, kx=3, ky=3)
        else:
            self.correct_me = False

        # This is the HCAM-equivalent band
        if '_s' in self.orig_band:
            self.band = self.orig_band
        else:
            self.band = self.orig_band + '_s'

        print("Finished setting up this flux!\n\n")


    def __str__(self):
        return "Flux object with band {} (HCAM equivalent: {}), flux {:.5f}, magnitude {:.3f}. I{} need to be color corrected to HCAM/GTC, super SDSS!".format(
            self.orig_band, self.band, self.flux, self.mag, "" if self.correct_me else " DON'T"
        )


    def color_correct_GTC_minus_obs(self, teff, logg):
        correction = 0.0

        # Interpolate the correction for this teff, logg
        if self.DEBUG:
            print("\nInterpolating color correction for band {} T: {:.0f} || logg: {:.3f}".format(self.band, teff, logg))

        if self.correct_me:
            correction = self.correction_func(teff, logg)[0,0]
        else:
            correction = 0.0

        if self.DEBUG:
            print("Got a correction of {:.3f} mags for {}".format(correction, self.band))
            print(" (I have a natively observed magnitude of {:.6f})".format(self.mag))

        return correction


    def bergeron_mag(self, teff, logg):
        '''Returns the calculated magnitude of this WD, as if it was observed
        with HiPERCAM on the GTC'''
        corr = self.color_correct_GTC_minus_obs(teff, logg)
        corrmag = self.mag + corr

        if self.DEBUG:
            print("Band {} || Magnitude: {:.6f} || Correction: {:.5f} || GTC/HCAM magnitude: {:.6f}".format(self.band, self.mag, corr, corrmag))

        return corrmag



def plotColors(model):
    print("\n\n-----------------------------------------------")
    print("Creating color plots...")
    _, ax = plt.subplots(figsize=(6,6))

    # OBSERVED DATA
    flux_u = [obs for obs in model.obs_fluxes if 'u' in obs.band][0]
    flux_g = [obs for obs in model.obs_fluxes if 'g' in obs.band][0]
    flux_r = [obs for obs in model.obs_fluxes if 'r' in obs.band][0]
    print("Observations:\n    {}\n    {}\n    {}".format(flux_u, flux_g, flux_r))

    obs_ug_err = np.sqrt((flux_u.magerr**2) + (flux_g.magerr**2))
    obs_gr_err = np.sqrt((flux_g.magerr**2) + (flux_r.magerr**2))

    # Correct magnitudes to the Bergeron frame
    t, g = model.teff.currVal, model.logg.currVal
    u_mag = flux_u.bergeron_mag(t, g)
    g_mag = flux_g.bergeron_mag(t, g)
    r_mag = flux_r.bergeron_mag(t, g)

    if model.DEBUG:
        print("Observation bergeron magnitudes (GTC/HCAM), uncorrected for extinction:")
        print("   Magnitudes:\n     u: {}\n     g: {}\n     r: {}".format(u_mag, g_mag, r_mag))

    # subtract interstellar extinction
    ex = model.ebv
    u_mag -= model.extinction_coefficients['u_s'] * ex.currVal
    g_mag -= model.extinction_coefficients['g_s'] * ex.currVal
    r_mag -= model.extinction_coefficients['r_s'] * ex.currVal

    print("After correcting to GTC/HCAM/Super, and removing IS extinction:")
    print("   Magnitudes:\n     u: {}\n     g: {}\n     r: {}".format(u_mag, g_mag, r_mag))

    ug_mag = u_mag - g_mag
    gr_mag = g_mag - r_mag

    print("Observed Colors in the HCAM/GTC/super lightpath (corrected for IS extinction):")
    print("u-g = {:> 5.3f}+/-{:< 5.3f}".format(ug_mag, obs_ug_err))
    print("g-r = {:> 5.3f}+/-{:< 5.3f}".format(gr_mag, obs_gr_err))


    # Generate the model's apparent magnitudes (no atmosphere, no IS extinction), and plot that color too
    # Get absolute magnitudes
    abs_mags = model.gen_absolute_mags()
    # Apply distance modulus
    dmod = 5.0*np.log10(model.dist/10.0)
    modelled_mags = abs_mags + dmod

    # Calculate the colours
    bands = [obs.orig_band for obs in model.obs_fluxes]
    u_index = bands.index(flux_u.orig_band)
    g_index = bands.index(flux_g.orig_band)
    r_index = bands.index(flux_r.orig_band)
    if model.DEBUG:
        print("Bergeron model interpolations for T: {:.0f}, log(g): {:.3f}...".format(model.teff.currVal, model.logg.currVal))
        print("Observed bands: {}".format(bands))
        print("Modelled mags: {}".format(modelled_mags))
        print("Indexes|| u: {} || g: {} || r: {}\n".format(u_index, g_index, r_index))

    model_ug = modelled_mags[u_index] - modelled_mags[g_index]
    model_gr = modelled_mags[g_index] - modelled_mags[r_index]

    # bergeron model magnitudes, will be plotted as tracks
    bergeron_umags = np.array(model.DA['u_s'])
    bergeron_gmags = np.array(model.DA['g_s'])
    bergeron_rmags = np.array(model.DA['r_s'])

    # calculate colours
    ug = bergeron_umags-bergeron_gmags
    gr = bergeron_gmags-bergeron_rmags

    # make grid of teff, logg from the bergeron table
    teffs = np.unique(model.DA['Teff'])
    loggs = np.unique(model.DA['log_g'])
    nteff = len(teffs)
    nlogg = len(loggs)
    # reshape colours onto 2D grid of (logg, teff)
    ug = ug.reshape((nlogg, nteff))
    gr = gr.reshape((nlogg, nteff))


    # Plotting
    # Bergeron cooling tracks and isogravity contours
    for a in range(nlogg):
        ax.plot(ug[a, :], gr[a, :], 'k-')
    for a in range(0, nteff, 4):
        ax.plot(ug[:, a], gr[:, a], 'r--')

    # Observed color
    ax.errorbar(
        x=ug_mag, y=gr_mag,
        xerr=obs_ug_err,
        yerr=obs_gr_err,
        fmt='o', ls='none', color='darkred', capsize=3,
        label='Observed (GTC/HCAM calculated)'
    )

    # Modelled color
    ax.errorbar(
        x=model_ug, y=model_gr,
        fmt='o', ls='none', color='blue', capsize=3,
        label='Modelled - T: {:.0f} | logg: {:.2f}'.format(t, g)
    )

    # annotate for teff
    xa = ug[0, 4] + 0.03
    ya = gr[0, 4]
    val = teffs[4]
    t = ax.annotate(
        'T = %d K' % val, xy=(xa, ya), color='r',
        horizontalalignment='left',
        verticalalignment='center', size='small'
    )
    t.set_rotation(0.0)

    xa = ug[0, 8] + 0.03
    ya = gr[0, 8]
    val = teffs[8]
    t = ax.annotate(
        'T = %d K' % val, xy=(xa, ya), color='r',
        horizontalalignment='left',
        verticalalignment='center', size='small'
    )
    t.set_rotation(0.0)

    xa = ug[0, 20] + 0.01
    ya = gr[0, 20] - 0.01
    val = teffs[20]
    t = ax.annotate(
        'T = %d K' % val, xy=(xa, ya), color='r',
        horizontalalignment='left',
        verticalalignment='top', size='small'
    )
    t.set_rotation(0.0)

    xa = ug[0, 24] + 0.01
    ya = gr[0, 24] - 0.01
    val = teffs[24]
    t = ax.annotate(
        'T = %d K' % val, xy=(xa, ya), color='r',
        horizontalalignment='left',
        verticalalignment='top', size='small'
    )
    t.set_rotation(0.0)

    ax.set_xlabel('{}-{}'.format(flux_u.orig_band, flux_g.orig_band))
    ax.set_ylabel('{}-{}'.format(flux_g.orig_band, flux_r.orig_band))
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.5, 0.5])
    ax.legend()

    plt.savefig("colorPlot.pdf")
    plt.show()

    print("Done!")
    print("-----------------------------------------------\n")

def plotFluxes(model):
    '''Plot the colors, and the theoretical WD cooling tracks'''
    print("\n\n-----------------------------------------------")
    print("Creating flux plots...")
    print("model is:")
    print(model)

    # Get modelled WD fluxes for this T, G.
    # Includes distance modulus and interstellar reddening.
    # Flux as seen through HCAM/GTC/Super
    model_mags = model.gen_apparent_mags()
    model_flx = sdssmag2flux(model_mags)
    # Central wavelengths for the bands
    lambdas = np.array([obs.cent_lambda for obs in model.obs_fluxes])

    print("Modelled magnitudes:")
    for obs, m, f in zip(model.obs_fluxes, model_mags, model_flx):
        band = obs.orig_band
        print("Band {:>4s}: Mag: {:> 7.3f}  || Flux: {:<.3f}".format(band, m, f))

    # Grab the observed magnitudes, and convert them to HCAM/GTC fluxes -- NOT their native flux!
    # Includes distance and interstellar reddenning
    teff, logg = model.teff.currVal, model.logg.currVal
    obs_mags = np.array([obs.bergeron_mag(teff, logg) for obs in model.obs_fluxes])

    obs_flx = sdssmag2flux(obs_mags)
    obs_flx_err = [obs.err for obs in model.obs_fluxes]

    # Do the actual plotting
    _, ax = plt.subplots(figsize=(5,5))
    ax.errorbar(
        lambdas, model_flx,
        xerr=None, yerr=None,
        fmt='o', ls='none', color='darkred', label='Modelled apparent flux',
        markersize=6, linewidth=1, capsize=None
    )
    ax.errorbar(
        lambdas, obs_flx,
        xerr=None, yerr=obs_flx_err,
        fmt='o', ls='none', color='blue', label='Observed app. flux (GTC/HCAM)',
        markersize=6, linewidth=1, capsize=None
    )
    # ax.set_title("Observed and modelled fluxes")
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Flux, mJy")
    ax.legend()

    plt.tight_layout()
    plt.savefig("fluxPlot.pdf")
    plt.show()

    print("Done!")
    print("-----------------------------------------------\n")


if __name__ == "__main__":
    # Allows input file to be passed to code from argument line
    import argparse
    parser = argparse.ArgumentParser(description='Fit WD Fluxes')
    parser.add_argument('file', action='store', help="input file")
    parser.add_argument('--summarise', dest='summarise', action='store_true',
                        help='Summarise existing chain file without running a new fit.')
    parser.add_argument('--no-chain', dest='nochain', action='store_true', help='No chain file is being used')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debugging.')

    args = parser.parse_args()

    # Use parseInput function to read data from input file
    input_dict = parseInput(args.file)
    summarise = args.summarise
    if summarise:
        print("I will NOT run a fit, but just re-create the output figures!")
    nochain = args.nochain
    debug = args.debug
    print(debug)

    # Read information about mcmc, priors, neclipses, sys err
    nburn = int(input_dict['nburn'])
    nprod = int(input_dict['nprod'])
    nthread = int(input_dict['nthread'])
    nwalkers = int(input_dict['nwalkers'])
    scatter = float(input_dict['scatter'])
    thin = int(input_dict['thin'])
    toFit = int(input_dict['fit'])

    # Grab the variables
    teff = Param.fromString('teff', input_dict['teff'])
    logg = Param.fromString('logg', input_dict['logg'])
    plax = Param.fromString('plax', input_dict['plax'])
    ebv  = Param.fromString('ebv', input_dict['ebv'])

    syserr = float(input_dict['syserr'])
    if not nochain:
        chain_file = input_dict['chain']
    flat = int(input_dict['flat'])

    # # # # # # # # # # # #
    # Load in chain file  #
    # # # # # # # # # # # #
    if nochain:
        colKeys = []
        fchain = []
        filters = []
    else:
        print("Reading in the chain file,", chain_file)
        if flat:
            with open(chain_file, 'r') as f:
                colKeys = f.readline().strip().split()[1:]
            fchain = readflatchain(chain_file)
        else:
            with open(chain_file, 'r') as f:
                colKeys = f.readline().strip().split()[1:]
            chain = readchain(chain_file)
            print("The chain has the {} walkers, {} steps, and {} pars.".format(*chain.shape))
            fchain = flatchain(chain, thin=thin)
        print("Done!")

    # Extract the fluxes from the chain file, and create a list of Fux objects from that
    chain_bands = [key for key in colKeys if 'wdflux' in key.lower()]# and 'kg5' not in key.lower()]
    print("I found the following bands in the chain file:")
    for band in chain_bands:
        print("--> {}".format(band))
    print('\n\n\n')

    fluxes = []
    for band in chain_bands:
        print("Doing band {}".format(band))

        #TODO: Fix this.
        if 'kg5' in band.lower():
            print("KG5 BANDS ARE CURRENTLY UNUSED!!! SKIPPING")
            input("> ")
        else:
            index = colKeys.index(band)
            mean, _, std = sigma_clipped_stats(fchain[:, index])

            flx = Flux(mean, std, band.lower().replace("wdflux_", ""), syserr=syserr, debug=debug)
            fluxes.append(flx)

    while True:
        print("Would you like to add another flux? I currently have {}".format([obs.orig_band for obs in fluxes]))
        cont = input("y/n: ")
        if cont.lower() == 'y':
            print("Enter a band:")
            band = input("> ")
            print("Enter a Flux, in mJy")
            flx = input("> ")
            print("Enter an error on flux, mJy")
            fle = input("> ")

            flx = float(flx)
            fle = float(fle)

            flux = Flux(flx, fle, band, syserr=syserr, debug=debug)
            fluxes.append(flux)
        else:
            print("Done!")
            break

    # Create the model object
    myModel = wdModel(teff, logg, plax, ebv, fluxes, debug=debug)
    npars = myModel.npars


    mags = myModel.gen_apparent_mags()
    chisq = myModel.chisq()
    print("\n\n\nFor a Teff, logg = {:.0f}, {:.3f}".format(myModel.teff.currVal, myModel.logg.currVal))
    print("I generated these magnitudes: {}".format(mags))
    print("This corresponds to the fluxes: {}".format(sdssmag2flux(mags)))
    print("My chisq is {:.3f}".format(chisq))
    print("I'm using the filters:")
    for obs in myModel.obs_fluxes:
        print("{:>4s}: Flux {:.3f}+/-{:.3f}".format(obs.orig_band, obs.flux, obs.err))

    # Just summarise a previous chain, then stop
    if summarise:
        chain = readchain('chain_wd.txt')
        nameList = ['Teff', 'log g', 'Parallax', 'E(B-V)']

        likes = chain[:, :, -1]

        # Plot the mean likelihood evolution
        likes = np.mean(likes, axis=0)
        steps = np.arange(len(likes))
        std = np.std(likes)

        # Make the likelihood plot
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.fill_between(steps, likes-std, likes+std, color='red', alpha=0.4)
        ax.plot(steps, likes, color="green")

        ax.set_xlabel("Step")
        ax.set_ylabel("ln_like")

        plt.tight_layout()
        plt.savefig('likelihoods.png')
        plt.show()

        # Flatten the chain for the thumbplot. Strip off the ln_prob, too
        flat = flatchain(chain[:, :, :-1])
        bestPars = []
        for i in range(npars):
            par = flat[:, i]
            lolim, best, uplim = np.percentile(par, [16, 50, 84])
            myModel[i] = best

            print("%s = %f +%f -%f" % (nameList[i], best, uplim-best, best-lolim))
            bestPars.append(best)

        print("Creating corner plots...")
        fig = thumbPlot(flat, nameList)
        fig.savefig('cornerPlot.pdf')
        fig.show()

        toFit = False

    if toFit:
        guessP = np.array(myModel)
        nameList = ['Teff', 'log_g', 'Parallax', 'E(B-V)']
        # p0 = emcee.utils.sample_ball(guessP, scatter*guessP, size=nwalkers)
        # sampler = emcee.EnsembleSampler(
        #     nwalkers,
        #     npars,
        #     ln_prob,
        #     args=(myModel,),
        #     threads=nthread
        # )

        mp.set_start_method("forkserver")
        pool = mp.Pool()
        ntemps = 10
        p0 = initialise_walkers_pt(
            guessP, scatter,
            nwalkers, ntemps, ln_prior,
            myModel
        )
        sampler = ptemcee.sampler.Sampler(
            nwalkers, npars,
            ln_likelihood, ln_prior,
            ntemps=ntemps,
            loglargs=(myModel,),
            logpargs=(myModel,),
            pool=pool
        )

        # burnIn
        pos, prob, state = run_burnin(sampler, p0, nburn)

        # production
        sampler.reset()
        col_names = "walker_no " + ' '.join(nameList) + ' ln_prob'
        sampler = run_ptmcmc_save(
            sampler,
            pos, nprod,
            "chain_wd.txt",
            col_names=col_names
        )
        chain = []
        for i in range(ntemps):
            chain.append(sampler.flatchain[0, i::ntemps, ...])
        chain = np.array(chain)
        print(chain.shape)

        # Plot the likelihoods
        likes = chain[:, :, -1]

        # Plot the mean likelihood evolution
        likes = np.mean(likes, axis=0)
        steps = np.arange(likes.shape[0])
        std = np.std(likes)

        # Make the likelihood plot
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.fill_between(steps, likes-std, likes+std, color='red', alpha=0.4)
        ax.plot(steps, likes, color="green")

        ax.set_xlabel("Step")
        ax.set_ylabel("ln_like")

        plt.tight_layout()
        plt.savefig('likelihoods.png')
        plt.show()

        bestPars = []
        print(chain.shape)
        for i in range(npars):
            par = chain[:, :, i]
            lolim, best, uplim = np.percentile(par, [16, 50, 84])
            myModel[i] = best

            print("%s = %f +%f -%f" % (nameList[i], best, uplim-best, best-lolim))
            bestPars.append(best)
        print("Creating corner plots...")
        fig = thumbPlot(chain[0], nameList)
        fig.savefig('cornerPlot.pdf')
        fig.show()
    else:
        bestPars = [par for par in myModel]

    print("Done!")
    print("Chisq = {:.3f}".format(myModel.chisq()))

    # Plot measured and model colors and fluxes
    print("Model: {}".format(myModel))
    plotColors(myModel)
    plotFluxes(myModel)
