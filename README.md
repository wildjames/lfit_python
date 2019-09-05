# LFIT_PYTHON
This project is the successor to `LFIT` - a `C++` code for fitting cataclysmic variable (CV) star's eclipse lightcurves. `LFIT` is fast but cannot be easily adapted to fit ligthcurves with lots of flickering or to fit individual eclipses whilst sharing certain parameters (e.g eclipse width) between eclipses. This software uses a tree structure, to group eclipses into branches that can share some parameters but not others. 


## LFIT; functional knowledge you're gonna need
[lfit](https://github.com/StuartLittlefair/lfit) is a pretty robust piece of code. It's core functionality is easy to use, though perhaps a bit finnicky to pass parameters to. More on this in a few paragraphs.

`lfit` operates not in time space, but rather in pahse space. This necessitates a but of pre-processing to get a lightcurve from the raw format, into flux as a function of eclipse phase. The eclipse minimum is assumed to occur at phase 0.0, though a variable in the model is a phase offset to tweak this... as a last resort! Good ephemeral data is much better than putting a plaster on the problem. To reiterate - `lfit` will output a lightcurve in `phase`, `flux`. 

The module has, essentially, one main useful object with one main function. The `lfit.CV` class handles the four components that are considered (namely, the disc, donor star, white dwarf, and bright spot), and puts them all together for you. These components *can* be called individually, but this **usually** is more useful for diagnostics than for modelling. The `lfit.CV` has a `calcFlux()` method, which will make up your lightcurve.

### Calc-ing Flux
The `calcFlux()` method requires the CV parameters in a specific order. Some of these are optional, and are only supplied when calculating the (more computationally expensive) complex bright spot (BS) model.
```python
pars = [
    'wdFlux -  white dwarf flux at maximum light'
    'dFlux  -  disc flux at maximum light'
    'sFlux  -  bright spot flux at maximum light'
    'rsFlux -  donor flux at maximum light'
    'q      -  mass ratio'
    'dphi   -  full width of white dwarf at mid ingress/egress'
    'rdisc  -  radius of accretion disc (scaled by distance to inner lagrangian point XL1)'
    'ulimb  -  linear limb darkening parameter for white dwarf'
    'rwd    -  white dwarf radius (scaled to XL1)'
    'scale  -  bright spot scale (scaled to XL1)'
    'az     -  the azimuth of the bright spot strip (w.r.t to line of centres between stars)'
    'fis    -  the fraction of the bright spot's flux which radiates isotropically'
    'dexp   -  the exponent which governs how the brightness of the disc falls off with radius'
    'phi0   -  a phase offset'
    'exp1   -  [OPTIONAL] BS strip first exponent'
    'exp2   -  [OPTIONAL] BS strip second exponent'
    'tilt   -  [OPTIONAL] BS strip emission tilt, away from normal'
    'yaw    -  [OPTIONAL] BS strip emission yaw, away from normal'
]
# initialise the CV
cv_object = lfit.CV(pars)

phase = np.linspace(-0.3, 0.3, 300)
flux  = cv_object(pars, phase)
```
Note that the CV must be initialised with the initial parameters. This is because tweaking the parameters a bit is much less expensive than restarting the model from scratch - when running an MCMC as fast as possible, this saving adds up!

### Limb Darkening
The model is usually only very dimly sensitive to the limb darkening coefficient (LDC), as this parameter only comes into play *during* the WD ingress and egress. As such, it's recommended to first make a fit with a naiive LDC, then use the resulting WD model to calculate the value of the LDC that *should* have been used, and conduct another fit with the new LDC.

### Quickly diagnosing a fittable eclipse
Something that is important to understand is that the location of the BS ingress and egress, in conjuntion with the width of the eclipse and the size of the disc, constrains the mass ratio, q. It is crucial to be able to resolve these BS features in order to lift degeneracy between the other three.

### Real world data considerations
One part of LFIT that I'd be remiss not to mention is the bin width. When comparing models to data, we must account for the duration of a single exposure - if we don't, then periods where the flux is changing rapidly (i.e. the highly important ingresses and egresses), would be incorrect. `calcFlux()` can also take a `width` argument, which is the exposure width of the data. This will often not be an issue if your observations are sucessive, i.e. negligible dead time between frames, but if this is *not* the case (as it often is), you must define this! If it's not defined, the software will infer the bin width from the data, perhaps incorrectly.

These are, obviously, proxy parameters for the actual physical parameters of the system. The resulting chain from the MCMC simulation can be converted into physical parameters after the fact, using `wdparams.py`


## Markov Chain Monte-Carlo (MCMC), emcee, and the Affine-Invariant Ensemble Sampler (AIES)
[`emcee`](https://github.com/dfm/emcee) is the workhorse of this fitting process. MCMC essentially works by taking an initial position in parameter space, tweaking it a bit, and seeing if the fit to data improves. If it does, great, move there. If it doesn't, the 'walker' still has a finite chance to move to the new position anyway. This helps (in theory) to stop the walker from getting trapped in local minima; a core property of an MCMC is that it will explore 100% of the parameter space, given an infinite amount of time. Over time, the areas(or volumes) of parameter space that have a better fit to the data will accumulate more 'footprints' from the walkers, and the sampling frequencies will eventually converge towards the posterior distribution of the model.

It's hopefully obvious that we generally have less than infinite computer time. There are various algorithms that tweak the outline above, and the one currently used is the AIES. The `emcee` [paper](https://arxiv.org/abs/1202.3665) has a better explaination than I can give, so I recommend reading that rather than this, but here's a digested version. The chain uses many walkers rather than just one. When proposing new steps, a walker takes a random peer, and leapfrogs over its position by some fraction. This gives the proposed 'stretch move' new position of the walker. This is repeated (in parallel) for each walker's position, and then proceeds as normal - if it's a better fit, accept, if it isn't, roll some dice to see if it's accepted anyway. Note that this means that the walkers are ***not*** independant of each other, so don't use Gelman-Rubin convergence testing on these chains!


## wdparams
`wdparams.py` is essentially a thin version of `mcmcfit.py`, only this time it takes the parameters found in a `chain_prod.txt` file as the 'observation', and fits white dwarf model atmosphere colours (the DA models found [here](http://www.astro.umontreal.ca/~bergeron/CoolingModels/)) to it. Using it is pretty much the same as using `mcmcfit.py`


## The model
This package uses a tree structure to group observational data. It's easier to explain with a diagram;
```

                   Trunk
                     |
                    / \
        Branch1 ---     --- Branch2
           |                  |
          / \                / \
     leaf1   leaf2      leaf3   leaf4
```
Parameters are set from the top with an ordered list (as per `emcee` docs), and sorted down into the branches and leaves by the model structure. The eclipses are stored and modelled on the leaves, with the branches separating different observational filters and the `Trunk` storing the global parameters, `q`, `dhpi`, and `rwd`. When we want to evaluate an eclipse fit, say that of `leaf1`, we call `leaf1.calcFlux`. This retrieves the parameters it inherits from its parent, `Branch1`, and grandparent, `Trunk`, passes them off to `lfit.CV`, and returns the result. 

By segregating the eclipses like this, rather than fitting each individually, much better constraints on parameters can be achieved, even with less than stellar data. CV flickering can cause the BS ingress and egress to be ambiguous, as the amplitude of this stochastic process is often similar to the size of the BS features. It is unlikely that two eclipses will be unfortunate enough to be masked in this way *in the same place*, so by fitting them together, the MCMC should find it preferable to settle on the feature common to them both. Similarly, the white dwarf and donor fluxes (which should not change on human timescales) should be constant between eclipses - hence are shared at the branch level.

### If you want to use this...
The core functionality is ambiguous to the model being fitted, and sub-classed to work with `lfit` & `emcee`. This means that this structure should be *relatively* easy to apply to other models in similar use-cases. The tree is constructed of `Node`s, that can have at most one parent, and any number of children. It can also be arbitrarily deep, for example the above illustration could have two levels of branches, one separating branches, and another layer separating observations that are distant in time, for a 4-layer model.

## Fitting the Proxy Parameters
Actually using the software is fairly easy. In essence,

1. Write a configuration file defining the initial conditions, and parameters of the MCMC
2. Run `mcmcfit.py` with the input file, e.g. `python3 /PATH/TO/LFIT_PYTHON/mcmcift.py mcmc_input.dat`
3. Wait. 
4. Run `wdparams.py` with its relevant input file, e.g. `python3 /PATH/TO/LFIT_PYTHON/wdparams.py wdinput.dat`
5. Use `ldparams.py` to calculate the limb darkening coefficient of this WD model
6. Repeat steps 1-4 with the new value of limb darkening
5. Analyse results!

In reality, this is often iterative, and the result of one chain leads into the start position of another, until convergence is reached. Then, the resulting converged chain is fed into `wdparams.py` for conversion into physical parameters.

This branch also has a notifier, which will email the resulting lightcurve figures, and the likelihood history and summary of the chain_prod file. Corner plots are not sent, as these are often several MB each, so must be retrieved manually. To use this, first a gmail bot needs to be created, and its credentials supplied in a file called `PATH/TO/LFIT_PYTHON/email_details.json`, with the following format:
```json
{
  "user": "ADDRESS",
  "pass": "PASSWORD"
}
```
***DO NOT USE YOUR PERSONAL EMAIL*** as the credentials are stored here in plaintext. Just make a new (gmail!) account fresh for this.

### input.dat
The input file needs a few parameters at a minimum. It's almost certainly easier to understand the structure by looking at the [example](test_data/mcmc_input.dat) file, so just look in there for the documentation. Similarly, the `wdparams` script needs an input configuration, and this [example](./wdinput.dat) is also given.

## Installation
1. Install `lfit`, as per the instructions.
2. pip install the requirements of `lfit_python`; `pip3 install -r requirements.txt`
3. Thats it


## TODO 
- AIES samplers are likely not suitable for parameter spaces with N > ~5. Should we move to a different algorithm?
- The `emcee` implimentation of parallel tempering is deprecated. [This](https://github.com/willvousden/ptemcee) branch is now the preferred one to use, and needs to be integrated into `lfit_python`.
