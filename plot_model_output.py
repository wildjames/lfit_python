import numpy as np
import matplotlib.pyplot as plt

import mcmc_utils as u
import model as toy
import configobj
from mcmcfit import construct_model

colnames = 'chain_colnames.txt'
chain_fname = 'chain_prod.txt'
input_fname = 'input.dat'


# Gather the results of the file
with open(colnames, 'r') as f:
    colKeys = f.read().strip().split(',')

print("Reading in the chain file...")
data = u.readchain_dask(chain_fname)
chain = u.flatchain(data)

result = np.mean(chain, axis=0)

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
    if key in model.parNames:
        msg = "\n\n"
        msg += "Setting the model parameter {} to the result value of {:.3f}"
        msg.format(key, value)
        print(msg)

        key = key.split('_')
        label = key[1]
        name = key[0]

        msg = "Searching for model.search_Par({}, {}) "
        mag += "gave a Param object with the name {}"
        msg.format(label, name, model.search_Par(label, name).name)
        print(msg)

        model.search_Par(label, name).currVal = value
model.report()

#################################
# The model is now fully built. #
#################################

# Get the model's graph
# model.draw()
model.plot_data(save=True)
