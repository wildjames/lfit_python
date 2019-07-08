import numpy as np
import matplotlib.pyplot as plt

import mcmc_utils as u
import model as toy
import configobj
from mcmcfit import construct_model


chain_fname = 'chain_prod.txt'
input_fname = 'mcmc_input.dat'

input_dict = configobj.ConfigObj(input_fname)
nsteps = int(input_dict['nprod'])
nwalkers = int(input_dict['nwalkers'])

# Grab the column names from the top.
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

print("data shape: {}".format(data.shape))
print("Data slice")
print(data[0, 59, :])
nwalkers, nsteps, npars = data.shape
chain = u.flatchain(data)

result = np.mean(chain, axis=0)
lolim, result, uplim = np.percentile(chain, [16, 50, 84], axis=0)

print("Result has the shape: {}".format(result.shape))

resultDict = {}
print("Result of the chain:")
for n, r in zip(colKeys, result):
    print("{:>20s} = {:.3f}".format(n, r))
    resultDict[n] = r


# # # # # # # # # # # # # # # # # # # # # # # #
# Use the input file to reconstruct the model #
# # # # # # # # # # # # # # # # # # # # # # # #
model = construct_model(input_fname)

# Set the parameters of the model to the results of the chain
for key, value in resultDict.items():
    if key in model.dynasty_par_names:
        # msg = "Setting the model parameter {} to the result value of {:.3f}".format(key, value)
        # print(msg)

        key = key.split('_')
        label = '_'.join(key[1:])
        name = key[0]

        model.search_par(label, name).currVal = value

# # # # # # # # # # # # # # # # #
# The model is now fully built. #
# # # # # # # # # # # # # # # # #

# Get the model's graph
# model.draw()
print("Model evaluated to ln_prob of {:.3f}".format(model.ln_prob()))
# model.plot_data(save=False)

# Plot an image of the walker likelihoods over time
# data is shape (nwalkers, nsteps, ndim+1)
print("Reading in the chain file for likelihoods...")
likes = data[:, :, -1].T

ax = plt.imshow(likes)
plt.show()