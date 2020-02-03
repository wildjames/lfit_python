import argparse
import os
from functools import partial

import numpy as np
import seaborn as sns
from astropy import constants as const
from astropy import units
from astropy.table import Table
from astropy.utils.console import ProgressBar as PB
from scipy import interpolate as interp
from scipy.optimize import brentq
from trm import roche
from mcmc_utils import readchain_dask, readflatchain, flatchain


def read_wood_file(filename):
    '''The Wood 1995 thick H layer models (CO)
        returns temp and radius in K and solar units'''
    x, y = np.loadtxt(filename, usecols=(5, 6)).T
    temp = 10**y * units.K
    radius = 10**x * units.R_sun / const.R_sun.to(units.cm).value
    # must reverse so temp increases...
    return temp[::-1], radius[::-1]


def read_panei_file(filename):
    x, y = np.loadtxt(filename, usecols=(1, 2)).T
    temp = 10**y * units.K
    radius = 10**x * units.R_sun / const.R_sun.to(units.cm).value
    temp, mass, radius = np.loadtxt(filename).T
    temp = 1000*temp*units.K
    mass = mass*units.M_sun
    radius = radius*units.R_sun/100
    return temp, mass, radius


def hamada_mr(baseDir):
    '''Hamada-Salpeter 0 Kelvin Mass-radius relation'''
    filename = os.path.join(baseDir, 'Hamada/mr_hamada.dat')
    mass, radius = np.loadtxt(filename, usecols=(0, 5)).T
    # sort out units
    mass, radius = mass*units.M_sun, radius*units.R_sun/100.0
    # return function which interpolates these points linearly
    return interp.interp1d(mass, radius, kind='linear')


def panei_mr(targetTemp, baseDir):
    '''given a target temp, returns a function giving
    radius as a function of mass.
    function is derived from cubic interpolation of Panei models'''
    assert np.all((targetTemp >= 5000*units.K) & (targetTemp < 45000*units.K)), \
        "Model invalid at temps less than 4000 or greater than 45,000 K"

    # read panei model grid in
    teffs, masses, radii = \
        read_panei_file(os.path.join(baseDir, 'Panei/12C-MR-H-He/12C-MR-H-He.dat'))

    # this function interpolates to give radius as function of temp, mass
    func = interp.SmoothBivariateSpline(teffs, masses, radii)

    # this function specifies the target temp to give just radius as fn of mass
    f2 = lambda x: func(targetTemp, x)[0]
    return f2


def wood_mr(targetTemp, baseDir):
    '''given a target temp, returns a function giving
        radius as a function of mass.

        function is derived from cubic interpolation of wood models'''
    files = ['Wood95/co040.2.04dt1', 'Wood95/co050.2.04dt1',
             'Wood95/co060.2.04dt1', 'Wood95/co070.2.04dt1',
             'Wood95/co080.2.04dt1', 'Wood95/co090.2.04dt1',
             'Wood95/co100.2.04dt1']
    m = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * units.M_sun
    r = []
    for file in files:
        temp, rad = read_wood_file(os.path.join(baseDir, file))
        # linear interpolation to find radius at this mass, temp
        r.append(np.interp(targetTemp.value, temp, rad))

    # fix units
    r = np.array(r)*units.R_sun

    # now return function which interpolates m,r arrays cubically
    return interp.interp1d(m, r, kind='cubic')


def geoFunc(mtest, scaled_mass, rw_a):
    '''given a scaled mass thingy and rw/a, returns a radius in solar units'''
    a3 = scaled_mass*mtest*units.M_sun
    a = a3**(1./3.)
    rtest = rw_a*a
    return rtest.to(units.R_sun).value


def find_wdmass(wdtemp, scaled_mass, rw_a, baseDir, model='hamada'):
    '''for a given white dwarf mass we have two estimates of the white dwarf radius:
        one from WD theoretical models, and one from the scaled_mass (which gives a)
        and rw_a - this is the geometric mass.

        this routine finds the white dwarf mass where these estimates agree, if
        one exists.'''
    assert model in ['hamada', 'wood', 'panei'], "Model %s not recognised" % model
    limits = {'hamada': [0.14, 1.44],
              'wood': [0.4, 1.0],
              'panei': [0.4, 1.2]}

    mlo, mhi = limits[model]

    if model == 'hamada':
        rwdFunc = hamada_mr(baseDir)
    elif model == 'wood':
        rwdFunc = wood_mr(wdtemp, baseDir)
    else:
        rwdFunc = panei_mr(wdtemp, baseDir)

    # scipys brentq finds the root of a function - in this case
    # a function which returns the difference in predicted radius at
    # a given mass.
    # geoFunc gives the geometric soln for WD radius at a given mass
    # this function = 0 when two estimates agree
    funcToSolve = lambda x:  geoFunc(x, scaled_mass, rw_a)-rwdFunc(x)

    # first we'll check that it actually changes sign between the two
    # limits of our model. Otherwise it has no root!
    valAtLoLimit = funcToSolve(mlo)
    valAtHiLimit = funcToSolve(mhi)
    if np.sign(valAtLoLimit * valAtHiLimit) > 0:
        # get here only when they are the same sign
        raise Exception('No valid solution for this model')

    # OK, there should be a solution: return it
    return brentq(funcToSolve, mlo, mhi)*units.M_sun

def logg(m,r):
    MSUN = 1.9891e+30
    RSUN = 6.95508e+8
    G    = 6.67384e-11*1000 #cgs units

    try:
        m = m.value * MSUN*1000 #cgs units
        r = r.value * RSUN*100 #cgs units
    except:
        m = m * MSUN*1000 #cgs units
        r = r * RSUN*100 #cgs units
    return np.log10(G*m/(r*r))

def solve(input_data, baseDir):
    q, dphi, rw, twd, p = input_data

    # Kepler's law gives a quantity related to wd mass and separation, a
    # scaled_mass = a**3/M_wd. I call this the scaled mass.
    scaled_mass = const.G * (1+q) * p**2 / 4.0 / np.pi**2

    # convert white dwarf radius to units of separation, rw/a
    xl1_a = roche.xl1(q)
    rw_a = rw*xl1_a

    solved = True
    try:
        # try wood models
        mw = find_wdmass(twd, scaled_mass, rw_a, baseDir, model='wood')
    except FileNotFoundError:
        raise
    except Exception:
        # try panei models (usually for masses above 1 Msun)
        try:
            mw = find_wdmass(twd, scaled_mass, rw_a, baseDir, model='panei')
        except FileNotFoundError:
            raise
        except Exception:
            # try hamada models (for masses less than 0.4 or more than 1.2 Msun)
            try:
                mw = find_wdmass(twd, scaled_mass, rw_a, baseDir, model='hamada')
            except FileNotFoundError:
                raise
            except Exception:
                solved = False

    # do nothing if none of the models yielded a solution
    # otherwise,
    if solved:
        # donor star mass
        mr = mw*q

        # from this wd mass and scaled mass, find a
        a3 = scaled_mass*mw
        a = a3**(1./3.)
        a = a.to(units.R_sun)

        # inc
        inc = roche.findi(q, dphi)
        sini = np.sin(np.radians(inc))

        # radial velocities
        kw = (2.0*np.pi*sini*q*a/(1+q)/p).to(units.km/units.s)
        kr = kw/q

        # secondary radius
        # use from Warner 1995 (eggleton)
        # radius of sphere with same volume as Roche Lobe...
        r2 = 0.49*a*q**(2.0/3.0)
        r2 /= 0.6 * q**(2.0/3.0) + np.log(1.0+q**(1.0/3.0))

        # logg from mass and radius of WD
        logg_wd = logg(mw, rw_a*a)

        # need to be a little careful here for different versions of astropy
        data = (q, mw, rw_a*a, logg_wd, mr, r2, a, kw, kr, inc)
        return data
    else:
        return None


if __name__ == "__main__":
    import pandas as pd
    from mcmc_utils import thumbPlot
    import matplotlib.pyplot as plt

    sns.set()
    parser = argparse.ArgumentParser(description='Calculate physical parameters from MCMC chain of LFIT parameters')
    parser.add_argument('file', action='store', help='chain_prod file from mcmcfit run')
    parser.add_argument('twd', action='store', type=float, help='white dwarf temperature (K)')
    parser.add_argument('e_twd', action='store', type=float, help='error on wd temp')
    parser.add_argument('p', action='store', type=float, help='orbital period (days)')
    parser.add_argument('e_p', action='store', type=float, help='error on period')
    parser.add_argument('--thin', '-t', type=int, help='amount to thin MCMC chain by', default=1)
    parser.add_argument('--nthreads', '-n', type=int, help='number of threads to run', default=6)
    parser.add_argument('--flat', '-f', type=int, help='Factor of thinning if flattened chain used', default=0)
    parser.add_argument('--dir', '-d', help='directory with WD models', default='/home/php18jfw/Github/lfit/params/')
    args = parser.parse_args()
    fname = args.file
    thin = args.thin
    nthreads = args.nthreads
    flat = args.flat
    baseDir = args.dir

    if baseDir is None:
        print("Using the script location as the WD model files location")
        baseDir = os.path.split(__file__)[0]
    print("baseDir (must contain WD model directories): {}".format(baseDir))

    print("Reading chain file...")
    if flat > 0:
        # Input chain already thinned but may require additional thinning
        fchain = utils.readflatchain(fname)
        nobjects = (flat*len(fchain))/thin
        fchain = fchain[:nobjects]
    else:
        chain = readchain_dask(fname)
        nwalkers, nsteps, npars = chain.shape
        fchain = flatchain(chain, npars, thin=thin)
    print("Done!")

    # grab param names from the file
    with open(fname, 'r') as chain_file:
        nameList = chain_file.readline().strip().split(' ')[1:]
    # we need q, dphi, rw from the chain
    qVals = fchain[:, nameList.index('q_core')]
    dphiVals = fchain[:, nameList.index('dphi_core')]
    rwVals = fchain[:, nameList.index('rwd_core')]
    chainLength = len(qVals)
    print("The chain contains {} samples".format(chainLength))

    print("Means; ")
    print("  q: {:.3f}".format(np.mean(qVals)))
    print("  dphi: {:.3f}".format(np.mean(dphiVals)))
    print("  rwd: {:.3f}".format(np.mean(rwVals)))

    # white dwarf temp
    twdVals = np.random.normal(loc=args.twd, scale=args.e_twd, size=chainLength)*units.K

    # period
    pVals = np.random.normal(loc=args.p, scale=args.e_p, size=chainLength)*units.d

    # loop over the MCMC chain, calculating system parameters as we go

    # table for results
    results = Table(names=('q', 'Mw', 'Rw', 'logg', 'Mr', 'Rr', 'a', 'Kw', 'Kr', 'incl'))
    # need to be a little careful about astropy versions here, since only
    # versions >=1.0 allow quantities in tables
    # function below extracts value from quantity and floats alike
    getval = lambda el: getattr(el, 'value', el)

    print("Getting solutions for each step of the MCMC chain...")
    psolve = partial(solve, baseDir=baseDir)
    data = zip(qVals, dphiVals, rwVals, twdVals, pVals)
    solvedParams = PB.map(psolve, data, multiprocess=True)

    print('Writing out results...')
    # loop over these results and put all the solutions in our results table
    bar = PB(solvedParams)
    for thisResult in bar:
        if thisResult is not None:
            results.add_row(thisResult)

    print('Found solutions for %d percent of samples in MCMC chain' % (100*float(len(results))/float(chainLength)))
    results.write('physicalparams.log', format='ascii', overwrite=True)

    data = results.to_pandas()

    with open('physicalparams_summary.log', 'w') as f:
        for p, m, s in zip(data.columns, data.mean(), data.std()):
            f.write("{:>5s}: {:.5f} +/- {:.5f}\n".format(p, m, s))
            print("{:>5s}: {:.5f} +/- {:.5f}".format(p, m, s))

    fig = thumbPlot(data, data.columns)
    plt.savefig("physicalParams_corners.png")