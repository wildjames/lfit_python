import numpy as np
import matplotlib.pyplot as plt

import mcmc_utils as u
import toy_model as toy
import configobj


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

input_dict = configobj.ConfigObj(input_fname)

# Read in information about mcmc, neclipses, use of complex/GP etc.
nburn = int(input_dict['nburn'])
nprod = int(input_dict['nprod'])
nthreads = int(input_dict['nthread'])
nwalkers = int(input_dict['nwalkers'])
ntemps = int(input_dict['ntemps'])
scatter_1 = float(input_dict['first_scatter'])
scatter_2 = float(input_dict['second_scatter'])
toFit = int(input_dict['fit'])

# neclipses no longer needed, but can be used to limit the maximum number of
# fitted eclipses
try:
    neclipses = int(input_dict['neclipses'])
except:
    neclipses = -1

iscomplex = bool(int(input_dict['complex']))
useGP = bool(int(input_dict['useGP']))
usePT = bool(int(input_dict['usePT']))
corner = bool(int(input_dict['corner']))
double_burnin = bool(int(input_dict['double_burnin']))
comp_scat = bool(int(input_dict['comp_scat']))

if useGP:
    # Read in GP params using fromString function from mcmc_utils.py
    ampin_gp = u.Param.fromString('ampin_gp', input_dict['ampin_gp'])
    ampout_gp = u.Param.fromString('ampout_gp', input_dict['ampout_gp'])
    tau_gp = u.Param.fromString('tau_gp', input_dict['tau_gp'])


# Start by creating the overall Model. Gather the parameters:
core_pars = [u.Param.fromString(name, s) for name, s in input_dict.items()
             if name in ['rwd', 'dphi', 'q']]
# and make the model object with no children
model = toy.LCModel('core', core_pars)

# Collect the bands and their params. Add them total model.
band_pars = ['wdFlux', 'rsFlux']

# Get a sub-dict of only band parameters
bandDict = {}
for key, string in input_dict.items():
    if np.any([key.startswith(par) for par in band_pars]):
        bandDict[key] = string

# Get a set of the bands we have in the input file
defined_bands = [key.split('_')[-1] for key in bandDict.keys()]
defined_bands = set(defined_bands)
print("I found definitions of the following bands: {}".format(defined_bands))

for band in defined_bands:
    band_pars = []
    for key, string in bandDict.items():
        if key.endswith("_{}".format(band)):
            name = key.split("_")[0]
            band_pars.append(u.Param.fromString(name, string))

    band = toy.Band(band, band_pars, parent=model)

model.report()

# These are the entries to ignore. I'll create a special subclass for complex
# eclipses later, but ignore them for now.
descriptors = ['file', 'plot', 'band']
complex_desc = ['exp1', 'exp2', 'yaw', 'tilt']
if not iscomplex:
    print("Using the complex BS model. ")
    descriptors.extend(complex_desc)

iecl = -1
while True:
    iecl += 1

    # The user can limit the number if eclipses to fit.
    if iecl == neclipses:
        break

    # Collect this eclipses' parameters.
    if np.any([key.endswith("_{}".format(iecl)) for key in input_dict]):
        # Initialise this eclipses's stuff.
        eclPars = []

        # What band are we going to be looking at?
        band = input_dict['band_{}'.format(iecl)]
        # Retrieve the band object, so we can request it as a parent later
        band = model.search_Node('Band', band)

        print("Eclipse {} belongs to the {}".format(iecl, band.name))

        # Loop through the input dict, searching for keys that have a tail
        # matching this eclipse
        for key, string in input_dict.items():
            if key.endswith("_{}".format(iecl)):
                # Make sure we don't create a parameter from any of the
                # descriptors.
                test = [d in key for d in descriptors] one
                if np.any(test):
                    continue

                # Construct the name of the parameter,
                # i.e. strip off the tail code
                name = key.replace("_{}".format(iecl), '')
                print("{} has the parameter {}, calling it {}".format(
                    iecl, key, name))

                # Make the Param object from the string, and
                # add it to our list of pars.
                param = u.Param.fromString(name, string)
                eclPars.append(param)

        # Read in the datafile associated with this eclipse
        fname = input_dict['file_{}'.format(iecl)]
        lc = toy.Lightcurve.from_calib(fname)

        # Trim the eclipse down to our desired range.
        start = float(input_dict['phi_start'])
        end = float(input_dict['phi_end'])
        lc.trim(start, end)

        # Construct the eclipse object
        eclipse = toy.Eclipse(lc, iscomplex, str(iecl), eclPars, parent=band)
        print("\n\n")
    else:
        break

neclipses = iecl
print("the input file has {} eclipses".format(neclipses))

model.report()
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