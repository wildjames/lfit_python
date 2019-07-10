import numpy as np
import matplotlib.pyplot as plt

import mcmc_utils as u
import configobj
from mcmcfit import construct_model, extract_par_and_key

import argparse


argparse.ArgumentParser(description="Takes a chain file and the input file that created it, and summarises the results. Plots the before and after lightcurves, and the corner plot of each eclipse. Plots the likelihood evolution.")

chain_fname = 'chain_prod.txt'
input_fname = 'mcmc_input.dat'

nskip = 0
thin = 1

input_dict = configobj.ConfigObj(input_fname)
nsteps = int(input_dict['nprod'])
nwalkers = int(input_dict['nwalkers'])

# Grab the column names from the top.
print("Reading in the file...")
with open(chain_fname, 'r') as chain_file:
    colKeys = chain_file.readline().strip().split(',')

    data = []
    for i in range(nsteps):
        step = []
        for j in range(nwalkers):
            line = chain_file.readline().strip().split(' ')
            line = [float(val) for val in line]
            step.append(line[1:])
        data.append(list(step))
data = np.array(data)

print("Done!\nData shape: {}".format(data.shape))
nwalkers, nsteps, npars = data.shape

# Create the flattened version of the chain
chain = data[nskip::thin, :, :].reshape((-1, npars))
print("The flattened chain has the shape: {}".format(chain.shape))

# Analyse the chain. Take the mean as the result, and 2 sigma as the error
result = np.mean(chain, axis=0)
lolim, result, uplim = np.percentile(chain, [16, 50, 84], axis=0)

print("Result has the shape: {}".format(result.shape))

# report. While we're doing that, make a dict of the values we want to assign
# to the new version of the model.
resultDict = {}
print("Result of the chain:")
for n, r, lo, up in zip(colKeys, result, lolim, uplim):
    print("{:>20s} = {:.3f}   +{:.3f}   -{:.3f}".format(n, r, r-lo, up-r))
    resultDict[n] = r


# # # # # # # # # # # # # # # # # # # # # # # #
# Use the input file to reconstruct the model #
# # # # # # # # # # # # # # # # # # # # # # # #
model = construct_model(input_fname)

# We want to know where we started, so we can evaluate improvements.
# Wok out how many degrees of freedom we have in the model
eclipses = model.search_node_type('Eclipse')
# How many data points do we have?
dof = np.sum([ecl.lc.x.size for ecl in eclipses])
# Subtract a DoF for each variable
dof -= len(model.dynasty_par_vals)
# Subtract one DoF for the fit
dof -= 1

print("\n\nInitial guess has a chisq of {:.3f} ({:d} D.o.F.).".format(model.chisq(), dof))
print("\nEvaluating the model, we get;")
print("a ln_prior of {:.3f}".format(model.ln_prior()))
print("a ln_like of {:.3f}".format(model.ln_like()))
print("a ln_prob of {:.3f}".format(model.ln_prob()))
print()
model.plot_data(save=True, figsize=(11, 8), save_dir='./Initial_figs')

# Set the parameters of the model to the results of the chain
for key, value in resultDict.items():
    if key in model.dynasty_par_names:
        # msg = "Setting parameter {} to the result value of {:.3f}".format(key, value)
        # print(msg)

        name, label = extract_par_and_key(key)

        model.search_par(label, name).currVal = value


# # # # # # # # # # # # # # # # #
# The model is now fully built. #
# # # # # # # # # # # # # # # # #

print("\n\nMCMC result has a chisq of {:.3f} ({:d} D.o.F.).".format(model.chisq(), dof))
print("\nEvaluating the model, we get;")
print("a ln_prior of {:.3f}".format(model.ln_prior()))
print("a ln_like of {:.3f}".format(model.ln_like()))
print("a ln_prob of {:.3f}".format(model.ln_prob()))
print()
model.plot_data(save=True, figsize=(11, 8), save_dir='./Final_figs/')

# Plot an image of the walker likelihoods over time.
# data is shape (nwalkers, nsteps, ndim+1)
print("Reading in the chain file for likelihoods...")
likes = data[:, :, -1].T

ax = plt.imshow(likes)
plt.show()


# Plot the mean likelihood evolution
likes = np.mean(likes, axis=0)
steps = np.arange(0, len(likes))
std = np.std(likes)

# Make the likelihood plot
fig, ax = plt.subplots(figsize=(11, 8))
ax.fill_between(steps, likes-std, likes+std, color='red', alpha=0.4)
ax.plot(likes, color="green")

ax.set_title('SDSS J0748 g_binned only')
ax.set_xlabel("Step")
ax.set_ylabel("-ln_like")

plt.tight_layout()
plt.savefig('Final_figs/likelihood.pdf')
plt.show()

# Corner plots. Collect the eclipses.
eclipses = model.search_node_type("Eclipse")
for eclipse in eclipses:
    # Get the par names from the eclipse
    par_labels = eclipse.node_par_names
    par_labels = ["{}_{}".format(par, eclipse.label) for par in par_labels]

    # Get the par names from the band
    band = eclipse.parent
    par_labels += ["{}_{}".format(par, band.label) for par in band.node_par_names]

    # get the par names from the core part of the model
    my_model = band.parent
    par_labels += ["{}_{}".format(par, my_model.label) for par in my_model.node_par_names]

    print("\nMy par_labels is:")
    print(par_labels)

    # Get the indexes in the chain file, and gather those columns
    keys = [colKeys.index(par) for par in par_labels]
    chain_slice = chain[:, keys]
    print("chain_slice has the shape:", chain_slice.shape)

    fig = u.thumbPlot(chain_slice, par_labels)
    plt.savefig("Final_figs/"+eclipse.name+'_corners.png')
    plt.close()
