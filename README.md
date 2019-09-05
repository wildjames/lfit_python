# LFIT_PYTHON
-----

This project is the successor to `LFIT` - a `C++` code for fitting cataclysmic variable (CV) star's eclipse lightcurves. `LFIT` is fast but cannot be easily adapted to fit ligthcurves with lots of flickering or to fit individual eclipses whilst sharing certain parameters (e.g eclipse width) between eclipses. This software uses a tree structure, to group eclipses into branches that can share some parameters but not others. 

## LFIT; functional knowledge you're gonna need
-----

[lfit](https://github.com/StuartLittlefair/lfit) is a pretty robust piece of code. It's core functionality is easy to use, though perhaps a bit finnicky to pass parameters to. More on this in a few paragraphs.

`lfit` operates not in time space, but rather in pahse space. This necessitates a but of pre-processing to get a lightcurve from the raw format, into flux as a function of eclipse phase. The eclipse minimum is assumed to occur at phase 0.0, though a variable in the model is a phase offset to tweak this... as a last resort! Good ephemeral data is much better than putting a plaster on the problem. To reiterate - `lfit` will output a lightcurve in `phase`, `flux`. 

The module has, essentially, one main useful object with one main function. The `lfit.CV` class handles the four components that are considered (namely, the disc, donor star, white dwarf, and bright spot), and puts them all together for you. These components *can* be called individually, but this **usually** is more useful for diagnostics than for modelling. The `lfit.CV` has a `calcFlux()` method, which will make up your lightcurve.

The `calcFlux()` method requires the CV parameters in a specific order. Some of these are optional, and are only supplied when calculating the (more computationally expensive) complex model.
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

One final part of LFIT that I'd be remiss not to mention is the bin width. When comparing models to data, we must account for the duration of a single exposure - if we don't, then periods where the flux is changing rapidly (i.e. the highly important ingresses and egresses), would be incorrect. `calcFlux()` can also take a `width` argument, which is the exposure width of the data. This will often not be an issue if your observations are sucessive, i.e. negligible dead time between frames, but if this is *not* the case (as is often can be!!), you must define this! If it's not defined, the software will infer the bin width from the data.

## emcee, and the Affine-Invariant Ensemble Sampler
-----

[`emcee`](https://github.com/dfm/emcee)



## TODO 
-----
- AIES samplers are likely not suitable for parameter spaces with N > ~5. Should we move to a different algorithm?
- The `emcee` implimentation of parallel tempering is deprecated. [This](https://github.com/willvousden/ptemcee) branch is now the preferred one to use, and needs to be integrated into `lfit_python`.
