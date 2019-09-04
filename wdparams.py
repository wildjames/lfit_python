from __future__ import division, print_function

import os
import sys
import warnings
from builtins import input, object, range
from collections import MutableSequence
from os.path import realpath

import emcee
import matplotlib.pyplot as plt
import numpy
import scipy.interpolate as interp
import seaborn
from past.utils import old_div

from mcmc_utils import *


class wdModel(MutableSequence):
    '''wd model
    can be passed to MCMC routines for calculating model and chisq, and prior prob

    also behaves like a list, of the current values of all parameters
    this enables it to be seamlessly used with emcee

    Note that parallax should be provided in MILLIarcseconds.'''


    # arguments are Param objects (see mcmc_utils)
    def __init__(self,teff,logg,plax,ebv):
        self.teff = teff
        self.logg = logg
        self.plax = plax
        self.ebv  = ebv

        # initialise list bit of object with parameters
        self.data = [self.teff,self.logg,self.plax,self.ebv]

    # these routines are needed so object will behave like a list
    def __getitem__(self,ind):
        return self.data[ind].currVal
    def __setitem__(self,ind,val):
        self.data[ind].currVal = val
    def __delitem__(self,ind):
        self.data.remove(ind)
    def __len__(self):
        return len(self.data)
    def insert(self,ind,val):
        self.data.insert(ind,val)
    @property
    def npars(self):
        return len(self.data)

    @property
    def dist(self):
        if self.plax.currVal < 0.0:
            return 0.0
        else:
            return 1000./self.plax.currVal

def parseInput(file):
    ''' reads in a file of key = value entries and returns a dictionary'''
    # Reads in input file and splits it into lines
    blob = np.loadtxt(file,dtype='str',delimiter='\n')
    input_dict = {}
    for line in blob:
        # Each line is then split at the equals sign
        k,v = line.split('=')
        input_dict[k.strip()] = v.strip()
    return input_dict

def model(thisModel,mask):
    t, g, p, ebv = thisModel
    d = thisModel.dist

    # load bergeron models
    root, fn = os.path.split(__file__)
    data = np.loadtxt(os.path.join(root, 'Bergeron/da_ugrizkg5.txt'))

    teffs = np.unique(data[:,0])
    loggs = np.unique(data[:,1])

    nteff = len(teffs)
    nlogg = len(loggs)
    assert t <= teffs.max()
    assert t >= teffs.min()
    #assert g >= loggs.min()
    #assert g <= loggs.max()

    abs_mags = []
    # u data in col 4, g in 5, r in 6, i in 7, z in 8, kg5 in 9
    for col_indx in range(4,10):
        z = data[:,col_indx]
        z = z.reshape((nlogg,nteff))
        # cubic bivariate spline interpolation
        func = interp.RectBivariateSpline(loggs,teffs,z,kx=3,ky=3)
        abs_mags.append(func(g,t)[0,0])
    abs_mags = np.array(abs_mags)

    # A_x/E(B-V) extinction from Cardelli (1989)
    # Where are these values from?? (KG5 estimated)

    ext       = ebv*np.array([5.155,3.793,2.751,2.086,1.479,3.5])
    dmod      = 5.0*np.log10(d/10.0)
    app_red_mags = abs_mags + ext + dmod

    #return app_red_mags
    return 3631e3*10**(-0.4*app_red_mags[mask])

def ln_prior(thisModel):
    lnp = 0.0

    #teff, (usually uniform between allowed range - 6 to 90,000)
    param = thisModel.teff
    lnp += param.prior.ln_prob(param.currVal)

    #logg, uniform between allowed range (7.01 to 8.99), or Gaussian from constraints
    param = thisModel.logg
    lnp += param.prior.ln_prob(param.currVal)

    # Parallax, gaussian prior of the gaia value.
    param = thisModel.plax
    lnp += param.prior.ln_prob(param.currVal)

    # reddening, cannot exceed galactic value of 0.121
    param = thisModel.ebv
    lnp += param.prior.ln_prob(param.currVal)
    return lnp

def chisq(thisModel,y,e,mask):
    m = model(thisModel,mask)
    try:
        resids = old_div((y[mask] - m), e[mask])
        return np.sum(resids*resids)
    except:
        return np.inf

def ln_likelihood(thisModel,y,e,mask):
    errs = e[mask]
    return -0.5*(np.sum( np.log( 2.0*np.pi*errs**2 ) ) + chisq(thisModel,y,e,mask))

def ln_prob(pars,thisModel,y,e,mask):

    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(thisModel.npars):
        thisModel[i] = pars[i]

    # now calculate log prob
    lnp = ln_prior(thisModel)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(thisModel,y,e,mask)
    else:
        return lnp

class Flux(object):
    def __init__(self,val,err,band):
        self.val = val
        self.err = err
        self.band = band
        self.mag = 2.5*numpy.log10(old_div(3631000,self.val))
        self.magerr = 2.5*0.434*(old_div(self.err,self.val))

def plotFluxes(fluxes,fluxes_err,mask,model):

    teff, logg, plax, ebv = model
    d = 1000. / plax

    # load bergeron models
    root, fn = os.path.split(__file__)
    data = np.loadtxt(os.path.join(root, 'Bergeron/da_ugrizkg5.txt'))

    teffs = np.unique(data[:,0])
    loggs = np.unique(data[:,1])

    nteff = len(teffs)
    nlogg = len(loggs)

    abs_mags = []
    # u data in col 4, g in 5, r in 6, i in 7, z in 8, kg5 in 9
    for col_indx in range(4,10):
        z = data[:,col_indx]
        z = z.reshape((nlogg,nteff))
        # cubic bivariate spline interpolation
        func = interp.RectBivariateSpline(loggs,teffs,z,kx=3,ky=3)
        abs_mags.append(func(logg,teff)[0,0])
    abs_mags = np.array(abs_mags)

    # A_x/E(B-V) extinction from Cardelli (1989)
    # Where are these values from?? (KG5 estimated)

    ext       = ebv*np.array([5.155,3.793,2.751,2.086,1.479,3.5])
    dmod      = 5.0*np.log10(old_div(d,10.0))
    app_red_mags = abs_mags + ext + dmod

    # calculate fluxes from model magnitudes
    model_fluxes = 3631e3*10**(-0.4*app_red_mags)

    # central wavelengths
    wavelengths = np.array([355.7,482.5,626.1,767.2,909.7,507.5])

    seaborn.set(style='ticks')
    seaborn.set_style({"xtick.direction": "in","ytick.direction": "in"})

    plt.errorbar(wavelengths[mask],model_fluxes[mask],xerr=None,yerr=None,fmt='o',ls='none',color='r',markersize=6,capsize=None)
    plt.errorbar(wavelengths[mask],fluxes[mask],xerr=None,yerr=fluxes_err[mask],fmt='o',ls='none',color='b',markersize=6,linewidth=1,capsize=None)
    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.ylabel('Flux (mJy)', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(top='on',right='on')
    #plt.xlim(300,700)
    #plt.ylim(0.03,0.05)
    #plt.yticks(np.arange(0.03,0.05,0.005))
    plt.subplots_adjust(bottom=0.10, top=0.98, left=0.11, right=0.975)
    plt.savefig('fluxPlot.pdf')
    plt.show()

def plotColors(mags):

    # load bergeron models
    root, fn = os.path.split(__file__)
    data = np.loadtxt(os.path.join(root, 'Bergeron/da_ugrizkg5.txt'))

    # bergeron model magnitudes
    umags = data[:,4]
    gmags = data[:,5]
    rmags = data[:,6]
    imags = data[:,7]
    zmags = data[:,8]
    kg5mags = data[:,9]

    # calculate colours
    ug = umags-gmags
    gr = gmags-rmags

    # make grid of teff, logg and colours
    teff = numpy.unique(data[:,0])
    logg = numpy.unique(data[:,1])
    nteff = len(teff)
    nlogg = len(logg)
    # reshape colours onto 2D grid of (logg, teff)
    ug = ug.reshape((nlogg,nteff))
    gr = gr.reshape((nlogg,nteff))

    # DATA!
    # If u band data available, chances are g and r data available too
    # u-g
    col1  = mags[0].mag - mags[1].mag
    col1e = numpy.sqrt(mags[0].magerr**2 + mags[1].magerr**2)
    col1l = mags[0].band + '-' + mags[1].band

    if rband_used:
        # g-r
        col2  = mags[1].mag - mags[2].mag
        col2e = numpy.sqrt(mags[1].magerr**2 + mags[2].magerr**2)
        col2l = mags[1].band + '-' + mags[2].band

    else:
        # g-i
        col2  = mags[1].mag - mags[3].mag
        col2e = numpy.sqrt(mags[1].magerr**2 + mags[3].magerr**2)
        col2l = mags[1].band + '-' + mags[3].band

    print('%s = %f +/- %f' % (col1l,col1,col1e))
    print('%s = %f +/- %f' % (col2l,col2,col2e))

    # now plot everthing
    for a in range(len(logg)):
        plt.plot(ug[a,:],gr[a,:],'k-')


    for a in range(0,len(teff),4):
        plt.plot(ug[:,a],gr[:,a],'r--')

    # annotate for log g
    #xa = ug[0,nteff/3]+0.03
    #ya = gr[0,nteff/3]-0.02
    #t = plt.annotate('log g = 7.0',xy=(xa,ya),color='k',horizontalalignment='center', verticalalignment='center',size='small')
    #t.set_rotation(30.0)
    #xa = ug[-1,nteff/3]-0.05
    #ya = gr[-1,nteff/3]+0.0
    #t = plt.annotate('log g = 9.0',xy=(xa,ya),color='k',horizontalalignment='center', verticalalignment='center',size='small')
    #t.set_rotation(45.0)

    # annotate for teff
    xa = ug[0,4] + 0.03
    ya = gr[0,4]
    val = teff[4]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='center',size='small')
    t.set_rotation(0.0)
    xa = ug[0,8] + 0.03
    ya = gr[0,8]
    val = teff[8]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='center',size='small')
    t.set_rotation(0.0)
    xa = ug[0,20] + 0.01
    ya = gr[0,20] - 0.01
    val = teff[20]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='top',size='small')
    t.set_rotation(0.0)
    xa = ug[0,24] + 0.01
    ya = gr[0,24] - 0.01
    val = teff[24]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='top',size='small')
    t.set_rotation(0.0)

    # plot data
    plt.errorbar(col1,col2,xerr=col1e,yerr=col2e,fmt='o',ls='none',color='r',capsize=3)
    plt.xlabel(col1l)
    plt.ylabel(col2l)
    plt.xlim([-0.5,1])
    plt.ylim([-0.5,0.5])
    plt.savefig('colorPlot.pdf')
    plt.show()

if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # Allows input file to be passed to code from argument line
    import argparse
    parser = argparse.ArgumentParser(description='Fit WD Fluxes')
    parser.add_argument('file',action='store',help="input file")

    args = parser.parse_args()

    # Use parseInput function to read data from input file
    input_dict = parseInput(args.file)

    # Read information about mcmc, priors, neclipses, sys err
    nburn    = int( input_dict['nburn'] )
    nprod    = int( input_dict['nprod'] )
    nthread  = int( input_dict['nthread'] )
    nwalkers = int( input_dict['nwalkers'] )
    scatter  = float( input_dict['scatter'] )
    thin     = int( input_dict['thin'] )
    toFit    = int( input_dict['fit'] )

    teff = Param.fromString('teff', input_dict['teff'] )
    logg = Param.fromString('logg', input_dict['logg'] )
    plax = Param.fromString('plax', input_dict['plax'] )
    ebv  = Param.fromString('ebv', input_dict['ebv'] )

    syserr = float( input_dict['syserr'] )

    flat = int( input_dict['flat'] )


    # # # # # # # # # # # #
    # Load in chain file  #
    # # # # # # # # # # # #

    chain_file = input_dict['chain']
    print("Reading in the chain file,", chain_file)
    if flat:
        with open(chain_file, 'r') as f:
            colKeys = f.readline().strip().split()[1:]
        fchain = readflatchain(chain_file)
    else:
        #chain = readchain(chain_file)
        with open(chain_file, 'r') as f:
            colKeys = f.readline().strip().split()[1:]
        chain = readchain_dask(chain_file)
        print("The chain has the {} walkers, {} steps, and {} pars.".format(*chain.shape))
        fchain = flatchain(chain, thin=thin)
    print("Done!")

    # Get the filters used from the column headers
    filters = [key.lower() for key in colKeys if key.startswith("wdFlux_")]
    filters = np.array(filters)
    print("I have the following filters:\n", filters)

    # Create arrays to be filled with all wd fluxes and mags
    fluxes = [0,0,0,0,0,0]
    fluxes_err = [0,0,0,0,0,0]
    mags = [0,0,0,0,0,0]


    # # # # # # # # # # # # # # # # # # # # # #
    # Collect the fluxes from the fit result. #
    # # # # # # # # # # # # # # # # # # # # # #

    # In some circumstances, the uband eclipse has to be fit separately
    # e.g. when it is of poor quality
    # For this reason, a uband wd flux and error can be input manually
    if "wdFlux_u" not in colKeys:
        while True:
            mode = input('Add seperate u band wd flux and error? (Y/N): ')
            if mode.upper() == 'Y' or mode.upper() == 'N':
                break
            else:
                print("Please answer Y or N ")

        if mode.upper() == "Y":
            uflux_in,uflux_err_in = input('Enter uband wd flux and error: ').split()
            uflux_in = float(uflux_in); uflux_err_in = float(uflux_err_in)
            uflux = uflux_in
            uflux_err = np.sqrt(uflux_err_in**2 + (uflux*syserr)**2)
            fluxes[0] = uflux
            fluxes_err[0] = uflux_err
            umag = Flux(uflux,uflux_err,'u')
            mags[0] = umag
            print((uflux,uflux_err))
            uband_used = True

    # For each filter, fill lists with wd fluxes from mcmc chain, then append to main array
    if 'wdFlux_u' in colKeys:
        index = colKeys.index('wdFlux_u')
        uband = fchain[:,index]

        uband = np.array([uband])
        # Need to calculate median values and errors
        uflux = np.median(uband)
        uflux_err = np.sqrt((np.std(uband))**2 + (uflux*syserr)**2)
        fluxes[0] = uflux
        fluxes_err[0] = uflux_err

        umag = Flux(uflux,uflux_err,'u')
        mags[0] = umag

        uband_used = True
    else: uband_used = False

    if 'wdFlux_g' in colKeys:
        index = colKeys.index('wdFlux_g')
        gband = fchain[:,index]

        gflux = np.median([gband])
        gflux_err = np.sqrt((np.std(gband))**2 + (gflux*syserr)**2)
        fluxes[1] = gflux
        fluxes_err[1] = gflux_err

        gmag = Flux(gflux,gflux_err,'g')
        mags[1] = gmag

        gband_used = True
    else: gband_used = False

    if 'wdFlux_r' in colKeys:
        index = colKeys.index('wdFlux_r')
        rband = fchain[:,index]

        rband = np.array([rband])
        rflux = np.median(rband)
        rflux_err = np.sqrt((np.std(rband))**2 + (rflux*syserr)**2)
        fluxes[2] = rflux
        fluxes_err[2] = rflux_err

        rmag = Flux(rflux,rflux_err,'r')
        mags[2] = rmag

        rband_used = True
    else: rband_used = False

    if 'wdFlux_i' in colKeys:
        index = colKeys.index('wdFlux_i')
        iband = fchain[:,index]

        iband = np.array([iband])
        iflux = np.median(iband)
        iflux_err = np.sqrt((np.std(iband)**2 + (iflux*syserr)**2))
        fluxes[3] = iflux
        fluxes_err[3] = iflux_err

        imag = Flux(iflux,iflux_err,'i')
        mags[3] = imag

        iband_used = True
    else: iband_used = False

    if 'wdFlux_z' in colKeys:
        index = colKeys.index('wdFlux_z')
        zband = fchain[:,index]

        zband = np.array([zband])
        zflux = np.median(zband)
        zflux_err = np.sqrt((np.std(zband))**2 + (zflux*syserr)**2)
        fluxes[4] = zflux
        fluxes_err[4] = zflux_err

        zmag = Flux(zflux,zflux_err,'z')
        mags[4] = zmag

        zband_used = True
    else: zband_used = False

    if 'wdFlux_kg5' in colKeys:
        index = colKeys.index('wdFlux_kg5')
        kg5band = fchain[:,index]

        kg5band = np.array([kg5band])
        kg5flux = np.median(kg5band)
        kg5flux_err = np.sqrt((np.std(kg5band))**2 + (kg5flux*syserr)**2)
        fluxes[5] = kg5flux
        fluxes_err[5] = kg5flux_err

        kg5mag = Flux(kg5flux,kg5flux_err,'kg5')
        mags[5] = kg5mag

        kg5band_used = True
    else: kg5band_used = False

    # Arrays containing all fluxes and errors
    fluxes = np.array(fluxes)
    fluxes_err = np.array(fluxes_err)

    y = fluxes
    e = fluxes_err

    # Create mask to discard any filters that are not used
    if 'uflux' in locals(): uband_used = True

    mask = np.array([uband_used,gband_used,rband_used,iband_used,zband_used,kg5band_used])

    print("I'm using the filters:")
    temp = ['u', 'g', 'r','i' , 'z', 'kg5']
    for t, m, flux in zip(temp, mask, fluxes):
        report = ''
        if m:
            report = '  -> Mean Flux: {:3f}'.format(flux)
        print("{}: {}{}".format(t, m, report))


    # # # # # # # # #
    # Model Fitting #
    # # # # # # # # #

    myModel = wdModel(teff,logg,plax,ebv)
    npars = myModel.npars

    if toFit:
        guessP = np.array(myModel)
        nameList = ['Teff','log g','Parallax','E(B-V)']

        p0 = emcee.utils.sample_ball(guessP,scatter*guessP,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[myModel,y,e,mask],threads=nthread)

        #burnIn
        pos, prob, state = run_burnin(sampler,p0,nburn)
        #pos, prob, state = sampler.run_mcmc(p0,nburn)

        #production
        sampler.reset()
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain_wd.txt")
        chain = flatchain(sampler.chain,npars,thin=thin)

        bestPars = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            myModel[i] = best

            print("%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim))
            bestPars.append(best)
        fig = thumbPlot(chain,nameList)
        fig.savefig('cornerPlot.pdf')
        fig.show()
        plt.close()
    else:
        bestPars = [par for par in myModel]

    dof = len(mags) - mags.count(0) - npars - 1
    print("Chisq = %.2f (%d D.O.F)" % (chisq(myModel,y,e,mask), dof))

    # Plot color-color plot
    if mask[0]:
        plotColors(mags)

    # Plot measured and model fluxes
    plotFluxes(fluxes,fluxes_err,mask,bestPars)
