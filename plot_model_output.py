import numpy as np
import matplotlib.pyplot as plt

import mcmc_utils as u
import model as toy
import configobj
from mcmcfit import construct_model


chain_fname = 'chain_prod.txt'
input_fname = 'mcmc_input.dat'

# Grab the column names from the top.
with open(chain_fname, 'r') as chain_file:
    colKeys = chain_file.readline().split(',')

print("Reading in the chain file...")
data = u.readchain_dask(chain_fname, skiprows=1)
chain = u.flatchain(data)

result = np.mean(chain, axis=0)
lolim, result, uplim = np.percentile(chain, [16, 50, 84], axis=0)

print("Result has the shape: {}".format(result.shape))

resultDict = {}
print("Result of the chain:")
for n, r in zip(colKeys, result):
    print("{:>10s} = {:.3f}".format(n, r))
    resultDict[n] = r


###############################################
# Use the input file to reconstruct the model #
###############################################
model = construct_model(input_fname)

# Set the parameters of the model to the results of the chain
for key, value in resultDict.items():
    if key in model.dynasty_par_names:
        msg = "\nSetting the model parameter {} to the result value of {:.3f}"
        msg.format(key, value)
        print(msg)

        key = key.split('_')
        label = key[1]
        name = key[0]

        model.search_par(label, name).currVal = value
model.report()

# model.search_par('0', 'sFlux').currVal = 0.080

#################################
# The model is now fully built. #
#################################

# Get the model's graph
# model.draw()
print("Model evaluated to ln_prob of {:.3f}".format(model.ln_prob()))
model.plot_data(save=True)
