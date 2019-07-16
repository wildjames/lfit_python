from model import *


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

    def plot(self, flx=None, save=False, show=True):
        '''Plot the data for this data. If we're also passed a model flx, plot
        that and its residuals.'''

        if flx is None:
            fig, ax = plt.subplots()

            # Plot the data
            ax.errorbar(
                self.x, self.y,
                yerr=self.ye,
                linestyle='none', ecolor='grey', zorder=1
                )
            ax.step(self.x, self.y, where='mid', color='black')

            # Labels. Phase is unitless
            ax.set_title(self.name)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Flux, mJy')

        else:
            fig, axs = plt.subplots(2, sharex=True)

            # Plot the data first. Also do errors
            axs[0].errorbar(
                self.x, self.y,
                yerr=self.ye,
                linestyle='none', ecolor='grey', zorder=1
                )
            axs[0].step(self.x, self.y, where='mid', color='black')

            # Plot the model over the data
            axs[0].plot(self.x, flx, color='red')

            # Plot the errorbars
            axs[1].errorbar(
                self.x, self.y-flx,
                yerr=self.ye,
                linestyle='none', ecolor='grey', zorder=1
                )
            axs[1].step(self.x, self.y-flx, where='mid', color='darkred')

            # 0 residuals line, to guide the eye
            axs[1].axhline(0.0, linestyle='--', color='black', alpha=0.7,
                           zorder=0)

            # Labelling. Top one gets title, bottom one gets x label
            axs[0].set_title(self.name)
            axs[0].set_ylabel('Flux, mJy')

            axs[1].set_xlabel('Phase')
            axs[1].set_ylabel('Residual Flux, mJy')

        # Arrange the figure on the page
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        # Do we want to save the figure?
        if save:
            plt.savefig(self.fname.replace('.calib', '')+'.png')

        if show:
            plt.show()
            return


# Subclasses.
class SimpleEclipse(Model):
    '''Subclass of Model, specifically for storing a single eclipse.
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
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
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
        if "Lightcurve" in str(lightcurve.__class__):
            self.lc = lightcurve
        elif isinstance(lightcurve, str):
            self.lc = Lightcurve.from_calib(lightcurve)
        else:
            msg = "Argument lightcurve is not a string or Lightcurve! "
            msg += "Got {}".format(lightcurve)
            raise TypeError(msg)

        self.initCV()

    def initCV(self):
        '''Try to create a CV.
        If we cant construct the input list of params, set cv to None.'''

        self.cv = CV(self.cv_parlist)
        # print("Created a CV object for eclipse {}...".format(self.label))

    def calcFlux(self):
        '''Fetch the CV parameter vector, and generate it's model lightcurve'''

        if self.cv is None:
            self.initCV()

        # Get the model CV lightcurve across our data.
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)


        return flx

    def chisq(self):
        '''Return the chisq of this eclipse, given current params.'''
        flx = self.calcFlux()

        # If the model gets any nans, return inf
        if np.any(np.isnan(flx)):
            if self.DEBUG:
                print("Model returned some ({}/{} data) nans.".format(
                    np.sum(np.isnan(flx)),
                    flx.shape[0]
                ))

            return np.inf

        # Calculate the chisq of this model.
        chisq = ((self.lc.y - flx) / self.lc.ye)**2
        chisq = np.sum(chisq)

        return chisq

    def ln_like(self, plot=False):
        '''Calculate the chisq of this eclipse, against the data stored in its
        lightcurve object.

        If plot is True, also plot the data in a figure.'''

        if plot:
            self.plotter()

        chisq = self.chisq()

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

            return -np.inf

        ##############################################
        # ~~~~~ Does the stream miss the disc? ~~~~~ #
        ##############################################

        slope = 80.0
        try:
            # q, rdisc_a were previously retrieved
            az = ancestor_param_dict['az'].currVal

            # If the stream does not intersect the disc, this throws an error
            x, y, vx, vy = roche.bspot(q, rdisc_a)

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
                return -np.inf

        except:
            if verbose:
                print("The mass stream of leaf {} does not intersect the disc!".format(self.name))
            return -np.inf

        # If we pass all that, then calculate the ln_prior normally
        lnp = super().ln_prior(verbose=verbose, *args, **kwargs)

        return lnp

    def plotter(self, save=False, figsize=(11., 8.), fname=None, save_dir='.'):
        '''Create a plot of the eclipse's data.

        If save is True, a copy of the figure is saved.

        If fname is defined, save the figure with that filename. Otherwise,
        infer one from the data filename
        '''

        # Re-init my CV object with the current params.
        self.initCV()

        # Generate the lightcurve of the total, and the components.
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)
        wd_flx = self.cv.ywd
        sec_flx = self.cv.yrs
        BS_flx = self.cv.ys
        disc_flx = self.cv.yd

        # print("This model has a chisq of {:.3f}".format(self.chisq()))

        # Start the plotting area
        fig, axs = plt.subplots(2, sharex=True, figsize=figsize)

        # Plot the data first. Also do errors
        axs[0].errorbar(
            self.lc.x, self.lc.y,
            yerr=self.lc.ye,
            linestyle='none', ecolor='grey', zorder=1
            )
        axs[0].step(self.lc.x, self.lc.y, where='mid', color='black')

        # Plot the model over the data
        axs[0].plot(self.lc.x, wd_flx, color='lightblue', label='WD')
        axs[0].plot(self.lc.x, sec_flx, color='magenta', label='Sec')
        axs[0].plot(self.lc.x, BS_flx, color='darkblue', label='BS')
        axs[0].plot(self.lc.x, disc_flx, color='brown', label='Disc')
        axs[0].plot(self.lc.x, flx, color='red')
        axs[0].legend()

        # Plot the errorbars
        axs[1].errorbar(
            self.lc.x, self.lc.y-flx,
            yerr=self.lc.ye,
            linestyle='none', ecolor='grey', zorder=1
            )
        axs[1].step(self.lc.x, self.lc.y-flx, where='mid', color='black')

        # 0 residuals line, to guide the eye
        axs[1].axhline(0.0, linestyle='--', color='black', alpha=0.7,
                       zorder=0)

        # Labelling. Top one gets title, bottom one gets x label
        axs[0].set_title(self.lc.name)
        axs[0].set_ylabel('Flux, mJy')

        axs[1].set_xlabel('Phase')
        axs[1].set_ylabel('Residual Flux, mJy')

        # Arrange the figure on the page, and show it
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        if save:
            # Check that save_dir exists
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            # If we didnt get told to use a certain fname, use this node's name
            if fname is None:
                fname = self.lc.name.replace('.calib', '.png')

            # Make the filename
            fname = '/'.join([save_dir, fname])

            # If the user specified a path like './figs/', then the above could
            # return './figs//Eclipse_N.pdf'; I want to be robust against that.
            while '//' in fname:
                fname = fname.replace('//', '/')

            plt.savefig(fname)

        return fig, axs

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

        return parlist


class ComplexEclipse(SimpleEclipse):
    '''Subclass of Model, specifically for storing a single eclipse.
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
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
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


class Band(Model):
    '''Subclass of Model, specific to observation bands. Contains the eclipse
    objects taken in this band.

    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
    '''

    # What kind of parameters are we storing here?
    node_par_names = ('wdFlux', 'rsFlux')


class LCModel(Model):
    '''Top layer Model class. Contains Bands, which contain Eclipses.
    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
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
                return -np.inf

        except:
            # If we get here, then roche couldn't find a dphi for this q.
            # That's bad!
            if verbose:
                msg = "Failed to calculate a value of dphi at node {}"
                print(msg.format(self.name))
            return -np.inf

        # Then, if we pass this, move on to the 'normal' ln_prior calculation.
        lnp += super().ln_prior(verbose=verbose)
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
    #TODO: Impliment this properly.

    def calcChangepoints(self):
        '''Caclulate the WD ingress and egresses, i.e. where we want to switch
        on or off the extra GP amplitude.

        Requires an eclipse object, since this is specific to a given phase
        range.
        '''

        # Also get object for dphi, q and rwd as this is required to determine
        # changepoints
        dphi = self.ancestor_param_dict['dphi']
        q = self.ancestor_param_dict['q']
        rwd = self.ancestor_param_dict['rwd']

        phi0 = getattr(self, 'phi0')

        dphi_change = np.fabs(self._olddphi - dphi.currVal) / dphi.currVal
        q_change = np.fabs(self._oldq - q.currVal) / q.currVal
        rwd_change = np.fabs(self._oldrwd - rwd.currVal) / rwd.currVal

        # Check to see if our model parameters have changed enough to
        # significantly change the location of the changepoints.
        if (dphi_change > 1.2) or (q_change > 1.2) or (rwd_change > 1.2):
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

        return changepoints

    def create_GP(self):
        """Constructs a kernel, which is used to create Gaussian processes.

        Creates kernels for both inside and out of eclipse,
        works out the location of any changepoints present, constructs a single
        (mixed) kernel and uses this kernel to create GPs

        Requires an Eclipse object to create the GP for. """

        # Get objects for ln_ampin_gp, ln_ampout_gp, ln_tau_gp and find the exponential
        # of their current values
        ln_ampin = self.ancestor_param_dict['ln_ampin_gp']
        ln_ampout = self.ancestor_param_dict['ln_ampout_gp']
        ln_tau = self.ancestor_param_dict['ln_tau_gp']

        ampin_gp = np.exp(ln_ampin.currVal)
        ampout_gp = np.exp(ln_ampout.currVal)
        tau_gp = np.exp(ln_tau.currVal)

        # Calculate kernels for both out of and in eclipse WD eclipse
        # Kernel inside of WD has smaller amplitude than that of outside
        # eclipse.

        # First, get the changepoints
        changepoints = self.calcChangepoints()

        # We need to make a fairly complex kernel.
        # Global flicker
        kernel = ampin_gp * george.kernels.Matern32Kernel(tau_gp)
        # inter-eclipse flicker
        for gap in changepoints:
            kernel += ampout_gp * george.kernels.Matern32Kernel(
                tau_gp,
                block=gap
            )

        # Use that kernel to make a GP object
        georgeGP = george.GP(kernel, solver=george.HODLRSolver)

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
        parent; Model, optional:
            The parent of this node.
        children; list(Model), or Model:
            The children of this node. Single Model is also accepted
        '''
        # For each eclipse, I want to know the log likelihood of its residuals
        gp_ln_like = 0.0

        # Get the residuals of the model
        residuals = self.lc.y - self.calcFlux()
        # Did the model turn out ok?
        if np.any(np.isinf(residuals)) or np.any(np.isnan(residuals)):
            return -np.inf

        # Create the GP of this eclipse
        gp = self.create_GP()
        # Compute the GP
        gp.compute(self.lc.x, self.lc.ye)

        # The 'quiet' argument tells the GP to return -inf when you get
        # an invalid kernel, rather than throwing an exception.
        gp_lnl = gp.log_likelihood(residuals, quiet=True)
        gp_ln_like += gp_lnl

        return gp_ln_like

    def plotter(self, save=False, *args, **kwargs):
        '''For each eclipse descended from me, plot their data.

        If save is True, save the figures.
        Figsize is passed to matplotlib.
        '''

        # Get the figure and axes from the eclipse
        fig, ax = super().plotter(save=False, *args, **kwargs)

        # Get the residuals of the model
        residuals = self.lc.y - self.calcFlux()
        # Did the model turn out ok?
        if np.any(np.isinf(residuals)) or np.any(np.isnan(residuals)):
            return -np.inf

        # Create the GP of this eclipse
        gp = self.create_GP()
        # Compute the GP
        gp.compute(self.lc.x, self.lc.ye)

        # Draw samples from the GP
        samples = gp.sample_conditional(residuals, self.lc.x, size=300)

        # Get the mean, mu, standard deviation, and
        mu = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        ax[1].fill_between(
            self.lc.x,
            mu + (1.0*std),
            mu - (1.0*std),
            color='r',
            alpha=0.4,
            zorder=20
        )

        return fig, ax


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