import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def nxdraw(model):
    '''Draw a hierarchical node map of the model.'''

    # Build the network
    G = model.create_tree()
    pos = model.hierarchy_pos(G)

    # Figure has two inches of width per node
    figsize = (2*float(G.number_of_nodes()), 8.0)
    print("Figure will be {}".format(figsize))

    _, ax = plt.subplots(figsize=figsize)
    nx.draw(
        G,
        ax=ax,
        pos=pos, with_labels=True,
        node_color='grey', font_weight='heavy')

    plt.show()

def plot_eclipse(ecl_node, save=False, figsize=(11., 8.), fname=None,
                 save_dir='.', ext='.png'):
    '''Create a plot of the eclipse's data.

    If save is True, a copy of the figure is saved.

    If fname is defined, save the figure with that filename. Otherwise,
    infer one from the data filename
    '''

    # Re-init my CV object with the current params.
    ecl_node.initCV()

    # Generate the lightcurve of the total, and the components.
    flx = ecl_node.cv.calcFlux(ecl_node.cv_parlist, ecl_node.lc.x, ecl_node.lc.w)
    wd_flx = ecl_node.cv.ywd
    sec_flx = ecl_node.cv.yrs
    BS_flx = ecl_node.cv.ys
    disc_flx = ecl_node.cv.yd

    # print("This model has a chisq of {:.3f}".format(ecl_node.chisq()))

    # Start the plotting area
    fig, axs = plt.subplots(2, sharex=True, figsize=figsize)

    # Plot the data first. Also do errors
    axs[0].errorbar(
        ecl_node.lc.x, ecl_node.lc.y,
        yerr=ecl_node.lc.ye,
        linestyle='none', ecolor='grey', zorder=1
        )
    axs[0].step(ecl_node.lc.x, ecl_node.lc.y, where='mid', color='black')

    # Plot the model over the data
    axs[0].plot(ecl_node.lc.x, wd_flx, color='lightblue', label='WD')
    axs[0].plot(ecl_node.lc.x, sec_flx, color='magenta', label='Sec')
    axs[0].plot(ecl_node.lc.x, BS_flx, color='darkblue', label='BS')
    axs[0].plot(ecl_node.lc.x, disc_flx, color='brown', label='Disc')
    axs[0].plot(ecl_node.lc.x, flx, color='red')
    axs[0].legend()

    # Plot the errorbars
    axs[1].errorbar(
        ecl_node.lc.x, ecl_node.lc.y-flx,
        yerr=ecl_node.lc.ye,
        linestyle='none', ecolor='grey', zorder=1
        )
    axs[1].step(ecl_node.lc.x, ecl_node.lc.y-flx, where='mid', color='black')

    # 0 residuals line, to guide the eye
    axs[1].axhline(0.0, linestyle='--', color='black', alpha=0.7,
                    zorder=0)

    # Labelling. Top one gets title, bottom one gets x label
    axs[0].set_title(ecl_node.lc.name)
    axs[0].set_ylabel('Flux, mJy')

    axs[1].set_xlabel('Phase')
    axs[1].set_ylabel('Residual Flux, mJy')

    # Arrange the figure on the page, and show it
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    if save:
        # Check that save_dir exists
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # If we didnt get told to use a certain fname, use this node's name
        if fname is None:
            fname = ecl_node.lc.name.replace('.calib', ext)

        # Make the filename
        fname = '/'.join([save_dir, fname])

        # If the user specified a path like './figs/', then the above could
        # return './figs//Eclipse_N.pdf'; I want to be robust against that.
        while '//' in fname:
            fname = fname.replace('//', '/')

        plt.savefig(fname)

    return fig, axs


def plot_GP_eclipse(ecl_node, save=False, figsize=(11., 8.), fname=None,
                    save_dir='.', ext='.png'):
    '''Plot my data. Returns fig, ax

    If save is True, save the figures.
    Figsize is passed to matplotlib.
    '''

    # Get the figure and axes from the eclipse
    fig, ax = plot_eclipse(ecl_node, False, figsize, fname, save_dir,
                ext)

    # Get the residuals of the model
    residuals = ecl_node.lc.y - ecl_node.calcFlux()

    # Create the GP of this eclipse
    gp = ecl_node.create_GP()
    # Compute the GP
    gp.compute(ecl_node.lc.x, ecl_node.lc.ye)

    # Draw samples from the GP
    samples = gp.sample_conditional(residuals, ecl_node.lc.x, size=300)

    # Get the mean, mu, standard deviation, and
    mu = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    ax[1].fill_between(
        ecl_node.lc.x,
        mu + (1.0*std),
        mu - (1.0*std),
        color='r',
        alpha=0.4,
        zorder=20
    )

    if save:
        # Check that save_dir exists
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # If we didnt get told to use a certain fname, use this node's name
        if fname is None:
            fname = ecl_node.lc.name.replace('.calib', ext)

        # Make the filename
        fname = '/'.join([save_dir, fname])

        # If the user specified a path like './figs/', then the above could
        # return './figs//Eclipse_N.pdf'; I want to be robust against that.
        while '//' in fname:
            fname = fname.replace('//', '/')

        plt.savefig(fname)

    return fig, ax

def plot_model(model, show, *args, **kwargs):
    '''Calls the relevant plotter for each eclipse contained in the model.
    Passes *args and **kwargs to it.
    '''

    eclipses = model.search_node_type("Eclipse")
    for eclipse in eclipses:
        if str(eclipse.__class__.__name__) in ["SimpleEclipse", "ComplexEclipse"]:
            fig, ax = plot_eclipse(eclipse, *args, **kwargs)
        elif str(eclipse.__class__.__name__) in ["SimpleGPEclipse", "ComplexGPEclipse"]:
            fig, ax = plot_GP_eclipse(eclipse, *args, **kwargs)

        if show:
            plt.show()

        plt.close()
        del fig
        del ax