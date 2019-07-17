import argparse
import getpass
import glob
import json
import os

import configobj
import matplotlib.pyplot as plt
import numpy as np
import yagmail as yag

import mcmc_utils as u
from mcmcfit import construct_model, extract_par_and_key


def notipy(send_to, fnames, body):
    print("fnames: ")
    for name in fnames:
        print("-> '{}'".format(name))


    # Who do we send the email to?
    # Also contains email bot login
    location = __file__.split('/')[:-1] + ["email_details.json"]
    details_loc = '/'.join(location)
    with open(details_loc, 'r') as f:
        details = json.load(f)

    send_from = details['user']
    pword = details['pass']

    if send_from == "Bot email address":
        print("Please set up the bot email in {} to enable email reporting!".format(details_loc))
        return

    subject = "Model plotting output."

    # Construct the email contents
    contents = [body]
    contents.extend(fnames)

    # Send with yagmail
    client = yag.SMTP(send_from, pword)
    client.send(send_to, subject, contents)
    print("Email sent!")

    return


def fit_summary(chain_fname, input_fname, nskip=0, thin=1, destination='', automated=False):

    # Retrieve some info from the input dict.
    input_dict = configobj.ConfigObj(input_fname)

    nsteps = int(input_dict['nprod'])
    nwalkers = int(input_dict['nwalkers'])

    # Do I need to send an email?
    emailme = destination != ''

    # Grab the column names from the top.
    with open(chain_fname, 'r') as chain_file:
        colKeys = chain_file.readline().strip().split(' ')[1:]
    print("Reading in the file...")
    data = u.readchain_dask(chain_fname)

    print("Done!\nData shape: {}".format(data.shape))
    print("Expected a shape (nwalkers, nprod, npars): ({}, {}, {})".format(nwalkers, nsteps, len(colKeys)))

    # Create the Final_figs and Initial_figs directories.
    if not os.path.isdir("Final_figs"):
        os.mkdir("Final_figs")
    if not os.path.isdir("Initial_figs"):
        os.mkdir("Initial_figs")

    # Plot an image of the walker likelihoods over time.
    # data is shape (nwalkers, nsteps, ndim+1)

    print("Reading in the chain file for likelihoods...")
    likes = data[:, :, -1]

    # # Image representation
    # ax = plt.imshow(likes)
    # plt.show()

    # Plot the mean likelihood evolution
    likes = np.mean(likes, axis=0)
    steps = np.arange(nskip, nskip+len(likes))
    std = np.std(likes)

    # Make the likelihood plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.fill_between(steps, likes-std, likes+std, color='red', alpha=0.4)
    ax.plot(steps, likes, color="green")

    ax.set_xlabel("Step")
    ax.set_ylabel("-ln_like")

    plt.tight_layout()
    plt.savefig('Final_figs/likelihood.pdf')
    if automated:
        plt.close()
    else:
        plt.show()

    # Check that the user is sure about not thinning or skipping, in light of the chain size.
    if not automated:
        if nskip == 0:
            nskip = input("You opted not to skip any data, are you still sure? (default 0)\n-> nskip: ")
            try:
                nskip = int(nskip)
            except:
                nskip = 0
        if thin == 1:
            thin = input("You opted not to thin the data, are you sure? (default 1)\n-> thin: ")
            try:
                thin = int(thin)
            except:
                thin = 1

    data = data[nskip::thin, :, :]
    nwalkers, nsteps, npars = data.shape

    print("After thinning and skipping: {}".format(data.shape))


    # Create the flattened version of the chain
    chain = data.reshape((-1, npars))
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

    model_report = ''
    model_report += "\n\nInitial guess has a chisq of {:.3f} ({:d} D.o.F.).\n".format(model.chisq(), dof)
    model_report += "\nEvaluating the model, we get;\n"
    model_report += "a ln_prior of {:.3f}\n".format(model.ln_prior())
    model_report += "a ln_like of {:.3f}\n".format(model.ln_like())
    model_report += "a ln_prob of {:.3f}\n".format(model.ln_prob())

    model.plot_data(show=automated, save=True, figsize=(11, 8), save_dir='Initial_figs')

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

    model_report += "\n\nMCMC result has a chisq of {:.3f} ({:d} D.o.F.).\n".format(model.chisq(), dof)
    model_report += "\nEvaluating the model, we get;\n"
    model_report += "a ln_prior of {:.3f}\n".format(model.ln_prior())
    model_report += "a ln_like of {:.3f}\n".format(model.ln_like())
    model_report += "a ln_prob of {:.3f}\n".format(model.ln_prob())
    print(model_report)
    model.plot_data(show=automated, save=True, figsize=(11, 8), save_dir='Final_figs/')

    if emailme:
        # Gather the files
        fnames = list(glob.iglob('Final_figs/*.pdf', recursive=True))
        fnames = list(glob.iglob('Final_figs/*.png', recursive=True))
        fnames = [name for name in fnames if not "corner" in name.lower()]

        notipy(destination, fnames, model_report)


    # Corner plots. Collect the eclipses.
    eclipses = model.search_node_type("SimpleEclipse")
    eclipses = eclipses.union(model.search_node_type("ComplexEclipse"))
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

        oname = "Final_figs/" + eclipse.name + '_corners.png'
        print("Saving to {}...".format(oname))
        plt.savefig(oname)
        plt.close()

        del chain_slice
        try:
            del fig
        except:
            pass


if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser(
        description='''Takes a chain file and the input file that created it, and
        summarises the results.
        Plots the before and after lightcurves, and the corner plot of each eclipse.
        Plots the likelihood evolution.'''
    )

    parser.add_argument(
        "chain_fname",
        help="The filename for the MCMC chain",
        type=str,
    )
    parser.add_argument(
        "input_fname",
        help="The input command file for the MCMC chain",
        type=str,
    )
    parser.add_argument(
        "--nskip",
        help="How many steps to cut off the front of the MCMC chain",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--thin',
        help='I will only take every Nth step of the chain',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--notify',
        help='I will email this address with the results of the chain.',
        type=str,
        default='',
    )


    args = parser.parse_args()

    chain_fname = args.chain_fname
    input_fname = args.input_fname

    nskip = args.nskip
    thin = args.thin

    destination = args.notify


    fit_summary(chain_fname, input_fname, nskip, thin, destination)
