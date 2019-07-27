# LFIT_PYTHON
-----

This project is the successor to `LFIT` - a `C++` code for fitting CV lightcurves. `LFIT` is fast but cannot be easily adapted to fit ligthcurves with lots of flickering or to fit individual eclipses whilst sharing certain parameters (e.g eclipse width) between eclipses.

The main change to the software is the creation of a 'middle-man' class, which acts as a mediator between the MCMC sampling software (in this case `emcee`) and the `LFIT` `C++` code. `LFIT` uses `CV` objects, capable of quickly simulating a lightcurve across a phase array, however this only handles <i>one</i> eclipse at a time. The CV system itself may change over time, and between bands, but some parameters will not vary much, if at all. For example, the mass ratio of the system will be constant over time, so ideally <i>all</i> eclipses will share this parameter. Conversely, the accretion disc radius can be highly variable over time, so need not necessarly be shared between datasets; somewhere in the middle are parameters like the white dwarf flux, which will be shared between eclipses in the same filter, but will be different between filters - some may share parameters, while others don't.
  
 ## Model Structure
 
  The 'middle-man' software written here is organised as a hierarchical tree. This model structure has been written as generally as possible, so may be adapted to other projects. The model has `Nodes`, which can have at most one parent, and any number (including zero) children. As a simple example, a tree may look like this:
  
  ```
                   Trunk
                     |
                    / \
        Branch1 ---     --- Branch2
           |                  |
          / \                / \
     leaf1   leaf2      leaf3   leaf4

Trunk stores: q
Branch stores: WD Flux, disc flux
Leaf stores: disc radius, WD parameters
```

Here, the leaves are capable of evaluating a model that has some parameters stored on the leaf in question, its parent Branch, and it's grandparent Trunk. `leaf1` and `leaf2` will have independant disc radii and WD parameters, but will share WD and disc fluxes, and q. `leaf2` and `leaf3` will <i>only</i> share q. 

Parameters are stored as Param objects.  Nodes may hold no parameters, in which case they will act as simple dividers between datasets that still, for whatever reason, are desired to be fit together. 

Parameters can be set and retrieved from any `Node` very easily. Calling `Node.dynasty_par_vals` will return a list of the parameter values below that node, for example, calling `Branch1.dynasty_par_vals` will return `[WD Flux, disc Flux, leaf1 disc radius, leaf1 WD parameters, leaf2 disc radius, leaf2 WD parameters]`. Similarly, setting the vector is as easy as `Branch1.dynasty_par_vals = new_vector`. As long as the list is constructed in the same order as the `Node` would output, the parameters are sorted into the correct child `Nodes` by the tree. Typically, users will likely want to set and get parameter vectors from the Trunk of their tree. 

When a leaf wants to evaluate its model, it will need to retrieve its parameters. This is done with `leaf.ancestor_param_dict`. This returieves a python `dict` of the parameters inherited by that leaf. The `dict` keys are simply the names of the `Param` objects stored at or above the node called. Note that only Params <i>above</i> the node are collected; i.e. `Branch1.ancestor_param_dict` will not contain either `leaf1` or `leaf2` parameters.

## LFIT - A brief user's guide

[LFIT](https://github.com/StuartLittlefair/lfit) is most useful in providing a CV class object. This object has a method, `calcFLux`, that takes a list of (14 or 18) CV parameters, a phase array, and some optional paramters on the resolution of the donor star and disc. The function then returns the CV flux along that phase array. Components can also be individually retrieved if necessary, though the sum is more often the important part. The model holds best to Cataclysmic Variables with a roughtly homogenous disc, and a bright spot that is not excessively extended.

## Model Fitting

The model is fit with an affine-invariant MCMC method, implimented by emcee. For an in-depth run-down of what exactly this means, see [the emcee paper](https://arxiv.org/abs/1202.3665), but in brief, the goal of an MCMC sampler is to draw samples from the posterior distribution of the free parameters, based on their priors and the likelihood of the model they produce to be the 'correct' fit to data. 

In a classic Metropolis-Hasting chain, steps are proposed, and either accepted or rejected based on the likelihood relative to the previous step. If the likelihood is worse, the step still has a finite chance of being accepted - this allows the chain to (in theory, given an infinite amount of time) acess the whole parameter space regardless of the starting position. An affine-invariant approach introduces so-called 'stretch moves', which uses many parallel chains to inform proposed moves. 

The following is a summary of the algorithm from the mentioned paper:


'This method involves simultaneously evolving an ensemble of K walkers, where the proposal distribution for one walker, k, is based on the current positions of the K−1 walkers in the [rest of the ensemble]. Here, “position” refers to a vector in the N-dimensional, real-valued parameter space. To update the position of a walker at position X<sub>k</sub>, a walker X<sub>j</sub> is drawn randomly from the remaining walkers and a new position is proposed: 

(Equation 7)
X<sub>k</sub>(t) → Y=X<sub>j</sub> + Z[X<sub>k</sub>(t)−X<sub>j</sub>]

where Z is a random variable drawn from a distribution g(Z=z).  It is clear that if g satisfies 

(Equation 8)
g(z<sup>−1</sup>) = z g(z)

the proposal of Equation (7) is symmetric. In this case, the chain will satisfy detailed balance if the proposal is accepted with probability

(Equation 9)
q = min(1, Z<sup>N−1</sup> [p(Y) / p(X<sub>k</sub>(t))] )

where N is the dimension of the parameter space. This procedure is then repeated for each walker in the ensemble in series.'

