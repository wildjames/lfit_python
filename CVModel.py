'''
Subclasses from the `model` module, that actually comprise the tree
structure of the model fed to emcee. The `trunk` is an LCModel or GPLCModel
node, with child Bands, that have XEclipse leaves to evaluate the CV lightcurve
fit to the data. Data is stored in the Lightcurve class.
'''


import os

import configobj
import george
import lfit
import numpy as np
from trm import roche

from model import Node, Param


class Lightcurve:
    '''This object keeps track of the observational data.
    Can be generated from a file, with Lightcurve.from_calib(file).'''
    def __init__(self, name, x, y, ye, w, fname=''):
        '''Here, hold this.'''
        self.name = name

        self.fname = fname

        self.x = x
        self.y = y
        self.ye = ye
        self.w = w

    @property
    def n_data(self):
        return self.x.shape[0]

    @classmethod
    def from_calib(cls, fname, name=None):
        '''Read in a calib file, of the format;
        phase flux error

        and treat lines with # as commented out.
        '''

        data = np.loadtxt(fname, delimiter=' ', comments='#')
        phase, flux, error = data.T

        # Filter out nans.
        mask = np.where(np.isnan(flux) == 0)

        phase = phase[mask]
        flux = flux[mask]
        error = error[mask]

        width = np.mean(np.diff(phase))*np.ones_like(phase)/2.

        # Set the name of this eclipse as the filename of the data file.
        if name is None:
            _, name = os.path.split(fname)

        return cls(name, phase, flux, error, width, fname=fname)

    def trim(self, lo, hi):
        '''Trim the data, so that all points are in the x range lo > xi > hi'''
        xt = self.x

        mask = (xt > lo) & (xt < hi)

        self.x = self.x[mask]
        self.y = self.y[mask]
        self.ye = self.ye[mask]
        self.w = self.w[mask]


# Subclasses.
class SimpleEclipse(Node):
    '''Subclass of Node, specifically for storing a single eclipse.
    Uses the simple BS model.
    Lightcurve data is stored on this level.

    Inputs:
    -------
      lightcurve; Lightcurve:
        A Lightcurve object, containing data
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    '''

    # Define this subclasses parameters
    node_par_names = (
        'dFlux', 'sFlux', 'ulimb', 'rdisc',
        'scale', 'az', 'fis', 'dexp', 'phi0'
    )

    def __init__(self, lightcurve, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If the lightcurve is a Lightcurve object, save it. Otherwise,
        # read it in from the file.
        if isinstance(lightcurve, Lightcurve):
            self.lc = lightcurve
        elif isinstance(lightcurve, str):
            self.lc = Lightcurve.from_calib(lightcurve)
        else:
            msg = "Argument lightcurve is not a string or Lightcurve! "
            msg += "Got {}".format(lightcurve)
            raise TypeError(msg)

        # Create the CV object
        self.cv = lfit.CV(self.cv_parlist)

        self.log('SimpleEclipse.__init__', "Successfully ran the SimpleEclipse initialiser.")

    def calcFlux(self):
        '''Fetch the CV parameter vector, and generate it's model lightcurve'''
        self.log('SimpleEclipse.calcFlux', "Doing calcFlux")

        # Get the model CV lightcurve across our data.
        try:
            flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)
        except Exception as e:
            print(e)
            self.log("SimpleEclipse.calcFlux", str(e))
            flx = np.nan

        self.log('SimpleEclipse.calcFlux', "Computed a lightcurve flux: \n{}\n\n\n".format(flx))
        return flx

    def calcComponents(self):
        '''Return a list of the component fluxes as well as the total
        returns:
          (tot_flx, wdFlux, sFlux, rsFlux, dFlux)
        '''
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)

        return flx, self.cv.ywd, self.cv.ys, self.cv.yrs, self.cv.yd

    def chisq(self, *args, **kwargs):
        '''Return the chisq of this eclipse, given current params.'''
        self.log('SimpleEclipse.chisq', "Doing chisq")
        flx = self.calcFlux()

        # If the model gets any nans, return inf
        if np.any(np.isnan(flx)):
            if self.DEBUG:
                print("Node returned some ({}/{} data) nans.".format(
                    np.sum(np.isnan(flx)),
                    flx.shape[0]
                ))

            self.log('SimpleEclipse.chisq', "I computed a flux that contains nans. Returning an inf chisq.")
            return np.inf

        # Calculate the chisq of this model.
        chisq = ((self.lc.y - flx) / self.lc.ye)**2
        chisq = np.sum(chisq)

        self.log('SimpleEclipse.chisq', "Computed a chisq of {}".format(chisq))
        return chisq

    def ln_like(self, *args, **kwargs):
        '''Calculate the chisq of this eclipse, against the data stored in its
        lightcurve object.

        If plot is True, also plot the data in a figure.'''

        self.log("SimpleEclipse.ln_like", "Returning an ln_like that is (-0.5 * chisq)")

        chisq = self.chisq()

        self.log("SimpleEclipse.ln_like", "Returning a ln_like of {}".format(-0.5*chisq))
        return -0.5 * chisq

    def ln_prior(self, verbose, *args, **kwargs):
        '''At the eclipse level, three constrains must be validated for each
        leaf of the tree.

        - Is the disc large enough to precess? We can't handle superhumping!
        - Is the BS scale unreasonably large, enough to cause the disc model
            to be inaccurate?
        - Is the azimuth of the BS out of range?

        If other constraints on the level of the individual eclipse are
        necessary, they should go in here.
        '''
        self.log("SimpleEclipse.ln_prior", "Checking that the values construct a valid CV!")

        # Before we start, I'm going to collect the necessary parameters. By
        # only calling this once, we save a little effort.
        ancestor_param_dict = self.ancestor_param_dict

        ##############################################
        # ~~ Is the disc large enough to precess? ~~ #
        ##############################################

        # Defined the maximum size of the disc before it starts precessing, as
        # a fraction of Roche Radius
        rdisc_max_a = 0.46

        # get the location of the L1 point from q
        q = ancestor_param_dict['q'].currVal
        xl1 = roche.xl1(q)

        # Get the rdisc, scaled to the Roche Radius
        rdisc = ancestor_param_dict['rdisc'].currVal
        rdisc_a = rdisc * xl1

        if rdisc_a > rdisc_max_a:
            if verbose:
                msg = "The disc radius of {} is large enough to precess! Value: {:.3f}".format(self.name, rdisc)
                print(msg)
            self.log("SimpleEclipse.ln_prior", "The disc radius is too large. Returning ln_prior = -np.inf")
            return -np.inf

        ##############################################
        # ~~~~~ Is the BS scale physically OK? ~~~~~ #
        ##############################################
        # We're gonna check to see if the BS scle is #
        # in the range rwd/3 < (BS scale) < rwd*3.   #
        # If it isn't, then it's either too          #
        # concentrated to make sense, or so large    #
        # that our approximation of a smooth disc is #
        # no longer a good idea.                     #
        #                                            #
        ##############################################

        # Get the WD radius.
        rwd = ancestor_param_dict['rwd'].currVal

        # Enforce the BS scale being within these limits
        rmax = rwd * 3.
        rmin = rwd / 3.

        scale = ancestor_param_dict['scale'].currVal

        if scale > rmax or scale < rmin:
            if verbose:
                print("Leaf {} has a BS scale that lies outside valid range!".format(self.name))
                print("Rwd: {:.3f}".format(rwd))
                print("Scale: {:.3f}".format(scale))
                print("Range: {:.3f} - {:.3f}".format(rmin, rmax))

            self.log("SimpleEclipse.ln_prior", "The BS is too large to be accurately modelled. Returning ln_prior = -np.inf")
            return -np.inf

        ##############################################
        # ~~~~~ Does the stream miss the disc? ~~~~~ #
        ##############################################

        slope = 80.0
        try:
            # q, rdisc_a were previously retrieved
            az = ancestor_param_dict['az'].currVal

            # If the stream does not intersect the disc, this throws an error
            x, y, _, _ = roche.bspot(q, rdisc_a)

            # Find the tangent to the disc
            alpha = np.degrees(np.arctan2(y, x))

            # If alpha is negative, the BS lags the disc.
            # However, the angle has to be less than 90 still!
            if alpha < 0:
                alpha = 90 - alpha

            # Disc tangent
            tangent = alpha + 90

            # Calculate the min and max azimuthes, using the tangent and slope
            minaz = max(0, tangent-slope)
            maxaz = min(178, tangent+slope)

            if az < minaz or az > maxaz:
                self.log("SimpleEclipse.ln_prior", "Azimuth is out of range. Returning ln_prior = -np.inf")
                return -np.inf

        except:
            if verbose:
                print("The mass stream of leaf {} does not intersect the disc!".format(self.name))
            self.log("SimpleEclipse.ln_prior", "The mass stream does not intersect the disc, returning ln_prior = -np.inf")
            return -np.inf

        self.log("SimpleEclipse.ln_prior", "Passed validity checks at {}.".format(self.name))

        # If we pass all that, then calculate the ln_prior normally
        lnp = super().ln_prior(verbose=verbose, *args, **kwargs)

        self.log("SimpleEclipse.ln_prior", "Computed a ln_prior of {}".format(lnp))
        return lnp

    @property
    def cv_parnames(self):
        names = [
            'wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'q', 'dphi',
            'rdisc', 'ulimb', 'rwd', 'scale', 'az', 'fis', 'dexp', 'phi0'
        ]

        return names

    @property
    def cv_parlist(self):
        '''Construct the parameter list needed by the CV'''

        par_name_list = self.cv_parnames

        param_dict = self.ancestor_param_dict

        parlist = [param_dict[key].currVal for key in par_name_list]

        self.log("SimpleEclipse.cv_parlist", "Constructed a cv_parlist of:\n{}".format(parlist))

        return parlist


class ComplexEclipse(SimpleEclipse):
    '''Subclass of Node, specifically for storing a single eclipse.
    Uses the complex BS model.
    Lightcurve data is stored on this level.

    Inputs:
    -------
      lightcurve; Lightcurve:
        A Lightcurve object, containing data
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    '''
    node_par_names = (
                'dFlux', 'sFlux', 'ulimb', 'rdisc',
                'scale', 'az', 'fis', 'dexp', 'phi0',
                'exp1', 'exp2', 'yaw', 'tilt'
            )

    @property
    def cv_parnames(self):
        names = [
            'wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'q', 'dphi',
            'rdisc', 'ulimb', 'rwd', 'scale', 'az', 'fis', 'dexp', 'phi0',
            'exp1', 'exp2', 'tilt', 'yaw'
        ]

        return names


class Band(Node):
    '''Subclass of Node, specific to observation bands. Contains the eclipse
    objects taken in this band.

    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    '''

    # What kind of parameters are we storing here?
    node_par_names = ('wdFlux', 'rsFlux')


class LCModel(Node):
    '''Top layer Node class. Contains Bands, which contain Eclipses.
    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    '''

    # Set the parameter names for this layer
    node_par_names = ('q', 'dphi', 'rwd')

    def ln_prior(self, verbose=False):
        '''Before we calculate the ln_prior of myself or my children, I check
        that my parameters are valid. I check that dphi is not too large for
        the current value of q.

        If other constraints on the core parameters become necessary, they
        should go here. If these tests fail, -np.inf is immediately returned.
        '''
        self.log('LCModel.ln_prior', "Checking global parameters for validity.")
        lnp = 0.0

        # Check that dphi is within limits
        tol = 1e-6

        dphi = getattr(self, 'dphi').currVal
        q = getattr(self, 'q').currVal

        try:
            # Get the value of dphi that we WOULD have at an
            # inclination of 90 degrees
            maxphi = roche.findphi(q, 90.0)

            # If dphi is out of range, return negative inf.
            if dphi > (maxphi - tol):
                if verbose:
                    msg = "{} has a dphi out of tolerance!\nq: {:.3f}"
                    msg += "\ndphi: {:.3f}, max: {:.3f} - {:.3g}"
                    msg += "\nReturning inf.\n\n"

                    msg.format(self.name, q, dphi, maxphi, tol)

                    print(msg)
                self.log('LCModel.ln_prior', "dphi is out of range. Returning ln_prior = -np.inf")
                return -np.inf

        except:
            # If we get here, then roche couldn't find a dphi for this q.
            # That's bad!
            if verbose:
                msg = "Failed to calculate a value of dphi at node {}"
                print(msg.format(self.name))
            self.log('LCModel.ln_prior', "Failed to calculate a value of dphi. Returning ln_prior = -np.inf")
            return -np.inf

        self.log('LCModel.ln_prior', "Passed parameter value validity checks.")

        # Then, if we pass this, move on to the 'normal' ln_prior calculation.
        lnp += super().ln_prior(verbose=verbose)

        self.log('LCModel.ln_prior', "Returning a ln_prior of {}".format(lnp))
        return lnp


class GPLCModel(LCModel):
    '''This is a subclass of the LCModel class. It uses the Gaussian Process.

    This version will, rather than evaluating via chisq, evaluate the
    likelihood of the model by calculating the residuals between the model
    and the data and computing the likelihood of those data given certain
    Gaussian Process hyper-parameters.

    These parameters require some explaination.
    ln_tau_gp:
      the ln(timescale) of the covariance matrix
    ln_ampin_gp:
      The base amplitude of the covariance matrix
    ln_ampout_gp:
      The additional amplitude of the covariance matrix, when the WD
      is visible
    '''

    # Add the GP params
    node_par_names = LCModel.node_par_names
    node_par_names += ('ln_ampin_gp', 'ln_ampout_gp', 'ln_tau_gp')


class SimpleGPEclipse(SimpleEclipse):
    # Set the initial values of q, rwd, and dphi. These will be used to
    # caclulate the location of the GP changepoints. Setting to initially
    # unrealistically high values will ensure that the first time
    # calcChangepoints is called, the changepoints are calculated.
    _olddphi = 9e99
    _oldq = 9e99
    _oldrwd = 9e99

    # _dist_cp is initially set to whatever, it will be overwritten anyway.
    _dist_cp = 9e99

    def calcChangepoints(self):
        '''Caclulate the WD ingress and egresses, i.e. where we want to switch
        on or off the extra GP amplitude.

        Requires an eclipse object, since this is specific to a given phase
        range.
        '''

        self.log('SimpleGPEclipse.calcChangepoints', "Calculating GP changepoints")

        # Also get object for dphi, q and rwd as this is required to determine
        # changepoints
        pardict = self.ancestor_param_dict

        dphi = pardict['dphi']
        q    = pardict['q']
        rwd  = pardict['rwd']
        phi0 = pardict['phi0']

        # Have they changed significantly?
        # If not, dont bother recalculating dist_cp
        dphi_change = np.fabs(self._olddphi - dphi.currVal) / dphi.currVal
        q_change = np.fabs(self._oldq - q.currVal) / q.currVal
        rwd_change = np.fabs(self._oldrwd - rwd.currVal) / rwd.currVal

        # Check to see if our model parameters have changed enough to
        # significantly change the location of the changepoints.
        if (dphi_change > 1.2) or (q_change > 1.2) or (rwd_change > 1.2):
            self.log('SimpleGPEclipse.calcChangepoints', "The GP changepoint locations have chnged significantly enough to warrant a recalculation...")

            # Calculate inclination
            inc = roche.findi(q.currVal, dphi.currVal)

            # Calculate wd contact phases 3 and 4
            phi3, phi4 = roche.wdphases(q.currVal, inc, rwd.currVal, ntheta=10)

            # Calculate length of wd egress
            dpwd = phi4 - phi3

            # Distance from changepoints to mideclipse
            dist_cp = (dphi.currVal+dpwd)/2.

            # save these values for speed
            self._dist_cp = dist_cp
            self._oldq = q.currVal
            self._olddphi = dphi.currVal
            self._oldrwd = rwd.currVal
        else:
            self.log('SimpleGPEclipse.calcChangepoints', "Using old values of dist_cp")
            # Use the old values
            dist_cp = self._dist_cp

        # Find location of all changepoints
        min_ecl = int(np.floor(self.lc.x.min()))
        max_ecl = int(np.ceil(self.lc.x.max()))

        eclipses = [e for e in range(min_ecl, max_ecl+1)
                    if np.logical_and(e > self.lc.x.min(),
                                      e < 1+self.lc.x.max()
                                      )
                    ]

        changepoints = []
        for e in eclipses:
            # When did the last eclipse end?
            egress = (e-1) + dist_cp + phi0.currVal
            # When does this eclipse start?
            ingress = e - dist_cp + phi0.currVal
            changepoints.append([egress, ingress])

        self.log('SimpleGPEclipse.calcChangepoints', "Computed GP changepoints as:\n{}".format(changepoints))
        return changepoints

    def create_GP(self):
        """Constructs a kernel, which is used to create Gaussian processes.

        Creates kernels for both inside and out of eclipse,
        works out the location of any changepoints present, constructs a single
        (mixed) kernel and uses this kernel to create GPs

        Requires an Eclipse object to create the GP for. """

        self.log('SimpleGPEclipse.create_GP', "Creating a new GP")

        # Get objects for ln_ampin_gp, ln_ampout_gp, ln_tau_gp and find the exponential
        # of their current values
        pardict = self.ancestor_param_dict

        ln_ampin   = pardict['ln_ampin_gp']
        ln_ampout  = pardict['ln_ampout_gp']
        ln_tau     = pardict['ln_tau_gp']

        ampin_gp   = np.exp(ln_ampin.currVal)
        ampout_gp  = np.exp(ln_ampout.currVal)
        tau_gp     = np.exp(ln_tau.currVal)

        # Calculate kernels for both out of and in eclipse WD eclipse
        # Kernel inside of WD has smaller amplitude than that of outside
        # eclipse.

        # First, get the changepoints
        changepoints = self.calcChangepoints()

        # We need to make a fairly complex kernel.
        # Global flicker
        self.log('SimpleGPEclipse.create_GP', "Constructing a new kernel")
        kernel = ampin_gp * george.kernels.Matern32Kernel(tau_gp)
        # inter-eclipse flicker
        for gap in changepoints:
            kernel += ampout_gp * george.kernels.Matern32Kernel(
                tau_gp,
                block=gap
            )

        # Use that kernel to make a GP object
        georgeGP = george.GP(kernel, solver=george.HODLRSolver)

        self.log('SimpleGPEclipse.create_GP', "Successfully created a new GP!")
        return georgeGP

    def ln_like(self):
        '''The GP sits at the top of the tree. It replaces the LCModel
        class. When the evaluate function is called, this class should
        hijack it, calculate the residuals of all the eclipses in the tree,
        and find the likelihood of each of those residuals given the current GP
        hyper-parameters.

        Inputs:
        -------
        label; str:
            A label to apply to the node. Mostly used when searching trees.
        parameter_objects; list(Param), or Param:
            The parameter objects that correspond to this node. Single Param is
            also accepted.
        parent; Node, optional:
            The parent of this node.
        children; list(Node), or Node:
            The children of this node. Single Node is also accepted
        '''

        self.log('SimpleGPEclipse.ln_like', "Computing ln_like for a GP")

        # For each eclipse, I want to know the log likelihood of its residuals
        gp_ln_like = 0.0

        # Get the residuals of the model
        residuals = self.lc.y - self.calcFlux()
        # Did the model turn out ok?
        if np.any(np.isinf(residuals)) or np.any(np.isnan(residuals)):
            if self.DEBUG:
                self.log('SimpleGPEclipse.ln_like', "GP ln_like computed inf or nan residuals for the model. Returning -np.inf for the likelihood.")
            return -np.inf

        # Create the GP of this eclipse
        gp = self.create_GP()
        # Compute the GP
        gp.compute(self.lc.x, self.lc.ye)

        # The 'quiet' argument tells the GP to return -inf when you get
        # an invalid kernel, rather than throwing an exception.
        gp_ln_like = gp.log_likelihood(residuals, quiet=True)

        if self.DEBUG:
            self.log('SimpleGPEclipse.ln_like', "GP computed a ln_like of {}".format(gp_ln_like))

        return gp_ln_like


class ComplexGPEclipse(SimpleGPEclipse):
    # Exactly as the simple GP Eclipse, but this time with the extra 4 params.
    node_par_names = ComplexEclipse.node_par_names

    @property
    def cv_parnames(self):
        names = [
            'wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'q', 'dphi',
            'rdisc', 'ulimb', 'rwd', 'scale', 'az', 'fis', 'dexp', 'phi0',
            'exp1', 'exp2', 'tilt', 'yaw'
        ]

        return names


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


def construct_model(input_file, debug=False):
    '''Takes an input filename, and parses it into a model tree.

    Inputs:
    -------
      input_file, str:
        The input.dat file to be parsed
      debug, bool:
        Enable the debugging flag for the Nodes. Debugging will be written to
        a file.

    Output:
    -------
      model root node
    '''

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
        neclipses = 9999

    # # # # # # # # # # # # # # # # #
    # Get the initial model setup # #
    # # # # # # # # # # # # # # # # #

    # Start by creating the overall Node. Gather the parameters:
    if use_gp:
        core_par_names = GPLCModel.node_par_names
        core_pars = [Param.fromString(name, input_dict[name])
                     for name in core_par_names]

        # and make the model object with no children
        model = GPLCModel('core', core_pars, DEBUG=debug)
    else:
        core_par_names = LCModel.node_par_names
        core_pars = [Param.fromString(name, input_dict[name])
                     for name in core_par_names]

        # and make the model object with no children
        model = LCModel('core', core_pars, DEBUG=debug)

    # # # # # # # # # # # # # # # # #
    # # # Now do the band names # # #
    # # # # # # # # # # # # # # # # #

    # Collect the bands and their params. Add them total model.
    band_par_names = Band.node_par_names

    # Use the Eclipse class to find the parameters we're interested in
    if is_complex:
        ecl_pars = ComplexEclipse.node_par_names
    else:
        ecl_pars = SimpleEclipse.node_par_names

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

        if use_gp:
            if is_complex:
                ComplexGPEclipse(lc, label, params, parent=my_band)
            else:
                SimpleGPEclipse(lc, label, params, parent=my_band)
        else:
            if is_complex:
                ComplexEclipse(lc, label, params, parent=my_band)
            else:
                SimpleEclipse(lc, label, params, parent=my_band)

    # Make sure that all the model's Band have eclipses. Otherwise, prune them
    model.children = [band for band in model.children if len(band.children)]

    return model
