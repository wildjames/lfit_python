
import numpy as np
import matplotlib.pyplot as plt
from pymultinest.solve import Solver
from pymultinest.analyse import Analyzer

import sys
sys.path.append("./lfit_TESTING/")
from model import CubeConverter, Prior
from mcmc_utils import thumbPlot

class WaveSolver(Solver):
    DEBUG = False
    def __init__(self, func, data, prior_list, *args, **kwargs):
        print("args passed to me:")
        print(args)
        print("Kwargs passed to me:")
        for k, v in kwargs.items():
            print("{}: {}".format(k, v))

        self.func = func
        self.data = data
        self.priors = prior_list
        self.convert = CubeConverter().convert

        super().__init__(*args, **kwargs)

    def chisq(self, vect, cube=False):
        '''Calculate the chisq of my function for a parameter vector'''
        if cube:
            thetas = []
            for u, prior in zip(vect, self.priors):
                theta = self.convert(u, prior)
                thetas.append(theta)
            vect = thetas

        obs_x, obs_y, obs_yerr = self.data
        mod_y = self.func(obs_x, vect)

        chisq = ((obs_y - mod_y)/obs_yerr)**2
        chisq = np.sum(chisq)
        return chisq

    def plot_model(self, vect, cube=False):
        x, y, yerr = self.data

        if cube:
            thetas = []
            for u, prior in zip(vect, self.priors):
                theta = self.convert(u, prior)
                thetas.append(theta)
            vect = thetas

        fig, ax = plt.subplots()
        ax.set_title("Toy data: $y = {:.1f} * sin({:.1f}*x)$".format(*vect))
        ax.errorbar(x, y, yerr, fmt='x ', color='black')
        ax.plot(x, model(x, vect), color='red')
        plt.show()

    def Prior(self, cube):
        '''Take a cube vector, and return the correct thetas'''
        vect = []

        if self.DEBUG:
            print("Entered prior with this cube:")
            print(cube)
        for u, prior in zip(cube, self.priors):
            if self.DEBUG:
                print("--------")
                print("  u = {:.2f}".format(u))
            theta = self.convert(u, prior)

            vect.append(theta)

            if self.DEBUG:
                print("  theta = {:.2f}".format(theta))
                print("  lnp = {:.2f}".format(prior.ln_prob(theta)))

        if self.DEBUG:
            print("")

        return vect

    def LogLikelihood(self, vect):
        '''Take a parameter vector, convert to desired parameters, and calculate the ln(like) of that vector

        This will be maximized
        '''
        if self.DEBUG:
            print("~~~~~~~~~~~")
            print("Entered the LogLikelihood with this vector:")
            print(vect)

        chisq = self.chisq(vect)

        if self.DEBUG:
            print("Got chisq = {:.3f}".format(chisq))
            print("~~~~~~~~~~~")

        return -0.5*chisq


# Dumb toy model. Literally just a sine wave
def model(x, vect):
    a,b,c,d = vect

    value = a * np.sin(b*x) + c
    return value

def chisq(vect, data, func):
    x, y, err = data

    model = func(x, vect)
    chisq = ((y-model)/err)**2
    chisq = np.sum(chisq)

    return chisq


# # # # # # # # # # # # # # # # # #
# # Generate some toy data here # #
# # # # # # # # # # # # # # # # # #

actual_vect = (5, 10, 3, 5)
err = 0.5
N_data = 10
# How many dimensions have we got?
dims = len(actual_vect)


x = np.random.uniform(0, 2, N_data)
y = model(x, actual_vect) + np.random.normal(0.0, err, N_data)
yerr = np.ones_like(y) * err

# This is my observational data '''lightcurve'''
observations = (x, y, yerr)

# uniform priors
p1 = Prior('uniform', 0, 10)
p2 = Prior('uniform', 0, 20)
p3 = Prior('uniform', 0, 10)
dummy_param = Prior('uniform', 0, 10)

plist = [p1, p2, p3, dummy_param]


# Run the solver
solution = WaveSolver(
    func=model, data=observations, prior_list=plist,
    n_dims=dims, verbose=True, outputfiles_basename='./out/'
)


## Analysis
# create analyzer object
a = Analyzer(dims, outputfiles_basename="./out/")

# get a dictionary containing information about
#   the logZ and its errors
#   the individual modes and their parameters
#   quantiles of the parameter posteriors
stats = a.get_stats()

# get the best fit (highest likelihood) point
bestfit_params = a.get_best_fit()

pos = bestfit_params['parameters']


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(solution)
print("Multinest best fit parameters:")
print(pos)
print("A.K.A, ")
print("y = {:.3f} * sin({:.3f}*x) + {:.3f}".format(*pos))
print("A model at this vector has chisq = {:.3f}".format(chisq(pos, observations, model)))
print("Actual solution: y = {:.3f} * sin({:.3f}*x) + {:.3f}".format(*actual_vect))
print("i.e., discrepancies of:")
for found, actual in zip(pos, actual_vect):
    print("  -> {:.1f}%".format(100*(found - actual)/actual))


## Plot results
MN_x = np.linspace(x.min(), x.max(), 1000)
MN_y = model(MN_x, pos)
ideal_y = model(MN_x, actual_vect)

fig, ax = plt.subplots(figsize=(10,5))

ax.set_title(
    "MultiNest solution: $y = {:.3f} * sin({:.3f}*x) + {:.3f}$".format(*pos))
ax.errorbar(x, y, yerr, fmt='x ', color='black', label='Observations')
ax.plot(MN_x, MN_y, color='red', linestyle='--', zorder=10, label='Multinest fit')
ax.plot(MN_x, ideal_y, color='blue', linestyle='--', zorder=0, label='Actual values')
ax.legend()


# "posterior chain" kinda but not really, this is actually the
chain = a.get_equal_weighted_posterior()[:, :dims]
thumb_fig = thumbPlot(chain, ['a', 'b', 'c', 'd'])


plt.tight_layout()
plt.show()



