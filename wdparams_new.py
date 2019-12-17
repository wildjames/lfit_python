import os
import warnings

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from astropy.stats import sigma_clipped_stats
import seaborn
from past.utils import old_div

from mcmc_utils import (flatchain, readchain_dask, readflatchain,
                        run_burnin, run_mcmc_save, thumbPlot)
from model import Param

# Location of data tables
ROOT, _ = os.path.split(__file__)
DA = pd.read_csv(os.path.join(ROOT, 'Bergeron/Table_DA'), delim_whitespace=True, skiprows=0, header=1)

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


class wdModel():
    '''wd model
    can be passed to MCMC routines for calculating model and chisq, and prior prob

    also behaves like a list, of the current values of all parameters
    this enables it to be seamlessly used with emcee

    Note that parallax should be provided in MILLIarcseconds.'''

    # arguments are Param objects (see mcmc_utils)
    def __init__(self, teff, logg, plax, ebv, fluxes):
        self.teff = teff
        self.logg = logg
        self.plax = plax
        self.ebv = ebv

        # initialise list bit of object with parameters
        self.variables = [self.teff, self.logg, self.plax, self.ebv]

        # Observed data
        self.obs_fluxes = fluxes

        self.DA = pd.read_csv(os.path.join(ROOT, 'Bergeron/Table_DA'), delim_whitespace=True, skiprows=0, header=1)

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
        if self.plax.currVal < 0.0:
            return 0.0
        else:
            return 1000./self.plax.currVal

    def gen_fluxes(self):
        '''
        Take my parameters, and interpolate a model flux corresponding to each of
        my observations. Return this list
        '''

        for obs in self.obs_fluxes:
            print(obs)


class Flux(object):
    BANDS = ['u', 'g', 'r', 'i', 'z']

    def __init__(self, val, err, band):
        self.val = val
        self.err = err
        self.band = band.replace("_s", "")

        if band.endswith('_s'):
            print("\n\nObserved flux {} is super SDSS! I will perform color corrections to transform it to regular SDSS.".format(band))
            self.correct_me = True

            ## Get the correction I need from the user
            # Valid telescopes, and their instruments
            telescopes = ['ntt', 'gtc', 'wht', 'vlt', 'none']
            instruments = {
                'ntt': ['ucam'],
                'gtc': ['hcam'],
                'wht': ['hcam', 'ucam'],
                'vlt': ['ucam'],
                'none': ['']
            }

            print("\nWhat telescope was band {} observed with? {}".format(band, telescopes))
            tel = input("> ")
            while tel not in telescopes:
                print("\nThat telescope is not supported! ")
                tel = input("> ")

            print("What instrument was band {} observed with? {}".format(band, instruments[tel]))
            inst = input("> ")


            print("\nWhat filter was used for this observation? Labelled as {}".format(band))
            print("Options: (u_s, g_s, r_s, i_s, z_s)")
            filt = input("> ")
            conv_filt = filt.replace("_s", "")

            print("This is a 'super' filter, so I need to do some colour corrections.")
            # Save the correction table for this band here
            correction_table_fname = 'calculated_mags_{}_{}.csv'.format(tel, inst)
            script_loc = os.path.split(__file__)[0]
            correction_table_fname = os.path.join(script_loc, 'color_correction_tables', correction_table_fname)
            print("Table is stored at {}".format(correction_table_fname))

            # Create an interpolater for the color corrections
            correction_table = pd.read_csv(correction_table_fname)

            # Model table teffs
            teffs = np.unique(correction_table['Teff'])
            loggs = np.unique(correction_table['logg'])
            corrections = np.array(correction_table["{}-{}".format(conv_filt, filt)]).reshape(len(loggs), len(teffs))
            self.correction_func = interp.RectBivariateSpline(loggs, teffs, corrections, kx=3, ky=3)
        else:
            self.correct_me = False

    def __str__(self):
        return "Flux object with band {}, flux {}, magnitude {} (no color correction, if applicable)".format(self.band, self.val, self.absmag())

    def absmag(self, teff=None, logg=None):
        mag = 2.5*np.log10(3631000 / self.val)
        magerr = 2.5*0.434*(self.err / self.val)

        print("Magnitude from my Flux, band {}: {}")

        if self.correct_me and teff is not None and logg is not None:
            # Interpolate the correction for this teff, logg

            print("Interpolating color correction for T: {:d} || logg: {:.3f}".format(teff, logg))
            correction = self.correction_func(logg, teff)[0,0]
            print("Got a correction of {:.3f} mags".format(correction))

            mag += correction

        return mag, magerr




if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # Allows input file to be passed to code from argument line
    import argparse
    parser = argparse.ArgumentParser(description='Fit WD Fluxes')
    parser.add_argument('file', action='store', help="input file")
    parser.add_argument('--summarise', dest='summarise', action='store_true',
                        help='Summarise existing chain file without running a new fit.')
    parser.add_argument('--no-chain', dest='nochain', action='store_true', help='No chain file is being used')

    args = parser.parse_args()

    # Use parseInput function to read data from input file
    input_dict = parseInput(args.file)
    summarise = args.summarise
    if summarise:
        print("I will NOT run a fit, but just re-create the output figures!")
    nochain = args.nochain

    # Read information about mcmc, priors, neclipses, sys err
    nburn = int(input_dict['nburn'])
    nprod = int(input_dict['nprod'])
    nthread = int(input_dict['nthread'])
    nwalkers = int(input_dict['nwalkers'])
    scatter = float(input_dict['scatter'])
    thin = int(input_dict['thin'])
    toFit = int(input_dict['fit'])

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
            chain = readchain_dask(chain_file)
            print("The chain has the {} walkers, {} steps, and {} pars.".format(*chain.shape))
            fchain = flatchain(chain, thin=thin)
        print("Done!")

    chain_bands = [key for key in colKeys if 'wdflux' in key.lower()]
    fluxes = []
    for band in chain_bands:
        print("Doing band {}".format(band))
        index = colKeys.index(band)
        mean, _, std = sigma_clipped_stats(fchain[index])

        flx = Flux(mean, std, band.lower().replace("wdflux_", ""))

        print(flx.absmag(10000, 7.00))
        fluxes.append(flx)
