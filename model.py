import numpy as np
import networkx as nx
import george
import matplotlib.pyplot as plt

from lfit import CV
from trm import roche


class Lightcurve:
    '''This object keeps track of the observational data.
    Can be generated from a file, with Lightcurve.from_calib(file).'''
    def __init__(self, name, x, y, ye, w, fname=''):
        '''Here, hold this.'''
        self.name = name

        self.fname = fname

        self.x = x
        self.y = y
        self.ye = ye
        self.w = w

    @classmethod
    def from_calib(cls, fname, name=None):
        '''Read in a calib file, of the format;
        phase flux error

        and treat lines with # as commented out.
        '''

        data = np.loadtxt(fname, delimiter=' ', comments='#')
        phase, flux, error = data[:, 0], data[:, 1], data[:, 2]
        width = np.mean(np.diff(phase))*np.ones_like(phase)/2.

        # Set the name of this eclipse as the filename of the data file.
        if name is None:
            name = fname.split('/')[-1]
        return cls(fname, phase, flux, error, width, fname=fname)

    def trim(self, lo, hi):
        xt = self.x

        mask = (xt > lo) & (xt < hi)
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.ye = self.ye[mask]
        self.w = self.w[mask]

    def plot(self, flx=None, save=False, show=True):
        '''Plot the data for this data. If we're also passed a model flx, plot
        that and its residuals.'''

        if flx is None:
            fig, ax = plt.subplots()

            # Plot the data
            ax.errorbar(
                self.x, self.y,
                yerr=self.ye,
                linestyle='none', ecolor='grey', zorder=1
                )
            ax.step(self.x, self.y, where='mid', color='black')

            # Labels. Phase is unitless
            ax.set_title(self.name)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Flux, mJy')

        else:
            fig, axs = plt.subplots(2, sharex=True)

            # Plot the data first. Also do errors
            axs[0].errorbar(
                self.x, self.y,
                yerr=self.ye,
                linestyle='none', ecolor='grey', zorder=1
                )
            axs[0].step(self.x, self.y, where='mid', color='black')

            # Plot the model over the data
            axs[0].plot(self.x, flx, color='red')

            # Plot the errorbars
            axs[1].errorbar(
                self.x, self.y-flx,
                yerr=self.ye,
                linestyle='none', ecolor='grey', zorder=1
                )
            axs[1].step(self.x, self.y-flx, where='mid', color='darkred')

            # 0 residuals line, to guide the eye
            axs[1].axhline(0.0, linestyle='--', color='black', alpha=0.7,
                           zorder=0)

            # Labelling. Top one gets title, bottom one gets x label
            axs[0].set_title(self.name)
            axs[0].set_ylabel('Flux, mJy')

            axs[1].set_xlabel('Phase')
            axs[1].set_ylabel('Residual Flux, mJy')

        # Arrange the figure on the page
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        # Do we want to save the figure?
        if save:
            plt.savefig(self.fname.replace('.calib', '')+'.png')

        if show:
            plt.show()
            return


class Model:
    r'''
    Inputs:
    -------
      label, str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted

    Description:
    ------------
    Base class for the hierarchical model interface. This functions as a node,
    and can have children and a parent.

    Can have any number of children, but at most one parent.

    Parameters can be retrieved from the bottom-up, and can be set from the
    top-down. With some fiddling, the opposite is also possible, but not
    recommended!

    This is a block used to build mode structures, something like:
                   Trunk
                     |
                    / \
        Branch1 ---     --- Branch2
           |                  |
          / \                / \
     leaf1   leaf2      leaf3   leaf4

    Leaves inherit Params from their branches, which inherit from their Trunk.
    This can be arbitrarily deep, and leaves need to have a leaf.calc()
    function defined in order to terminate the recursive chisq function.

    Parameter lists for the tree can be retrieved and set from any level.
    e.g., Trunk.par_val_list contains the parameters for all nodes, but
    Branch2.par_val_list contains only those of Branch2, leaf3, and leaf4.
    Setting X.par_val_list sorts the parameters to the correct nodes, provided
    it's in the correct order (which can be retrieved with X.par_names)

    Conversely, leaf4.par_dict moves the other way. It will contain ONLY the
    parameters of leaf4, Branch2, and Trunk.

    Should be subclassed with the self.node_par_names variable defined, as this
    is a blank slate. Without that variable, this model cannot store
    parameters. There may be usecases where this can be exploited to group
    branches or leaves without them sharing any parameters.
    '''

    # Init the node_par_names to be empty and a list
    node_par_names = ()

    def __init__(self, label, parameter_objects, parent=None, children=None):
        '''Store parameter values to the parameter names, and track the parent
        and children, if necessary.'''

        # I expect my parameter values to be fed in as a list. If they're not
        # a list, assume I have a single Param object, and wrap it in a list.
        parameter_objects = list(parameter_objects)

        # Make sure our label is valid
        assert isinstance(label, str), "Label must be a string!"
        self.label = label

        # print("Creating a new {}, labelled {}".format(
        #     self.__class__.__name__, self.label))

        # Check that the user defined their parameter names!
        if len(self.node_par_names) != len(parameter_objects):
            fail_msg = 'I recieved the wrong number of parameters!'
            fail_msg += ' Expect: \n{}\nGot:\n{}'.format(
                self.node_par_names,
                [obj.name for obj in parameter_objects]
            )
            raise TypeError(fail_msg)

        # Add the parameters to the self.XXX.
        for par in parameter_objects:
            setattr(self, par.name, par)

        # Save the variable param names as a list
        self.node_varpars = [par.name for par in parameter_objects
                             if par.isVar]

        # Handle the family
        if children is None:
            children = []
        self.children = children
        self.parent = parent

        # Verify my parameters get put in the right places
        self.__check_par_assignments__()

    # Tree handling methods
    def search_par(self, label, name):
        '''Check if I have <name> in my attributes. If I don't, check my children.
        If they don't, return None'''
        # If I'm the desired node, get my parameter
        if self.label == label:
            return getattr(self, name)
        # Otherwise, check my children.
        else:
            for child in self.children:
                val = child.search_par(label, name)
                if val is not None:
                    return val
            return None

    def search_Node(self, class_type, label):
        '''Search for a node below me of class_type, with the label requested.
        '''
        if self.name == "{}_{}".format(class_type, label):
            return self
        else:
            for child in self.children:
                val = child.search_Node(class_type, label)
                if val is not None:
                    return val
                else:
                    pass
            return None

    def add_child(self, children):
        # This check allows XXX.add_child(Param) to be valid
        if not isinstance(children, list):
            children = [children]

        new_children = self.children + children

        self.children = new_children

    # Tree evaluation methods
    def chisq(self, plot=False):
        '''Call the calc on each of my children. Overwrite this for the
        bottom layer Model!'''

        chisq = 0.0

        if self.is_leaf:
            if not hasattr(self, 'calc'):
                print("{} has no calc function! Skipping...".format(self.name))
            else:
                calc = getattr(self, 'calc')
                calc(plot)

        for child in self.children:
            try:
                chisq += child.calc(plot)
            except:
                chisq += child.chisq(plot)

        return chisq

    def ln_like(self):
        '''Calculate the log likelihood, via chi squared.'''
        return -0.5 * self.chisq()

    def ln_prior(self, verbose=False):
        """Return the natural log of the prior probability of the Param objects
        below this node.

        If model has more prior information not captured in the priors of the
        parameters, the details of such additional prior information must be
        codified in subclass methods!."""

        # Start at a log prior probablity of 0. We'll add to this for each node
        lnp = 0.0

        # Get the sum of this node's prior probs
        for param in [getattr(self, name) for name in self.node_par_names]:
            if param.isValid:
                lnp += param.prior.ln_prob(param.currVal)
            else:
                if verbose:
                    print("Param {} in {} is invalid!".format(
                        param.name, self.name))
                return -np.inf

        if verbose:
            print("The sum of parameter ln_priors of {} is {:.3f}!".format(
                self.name, lnp))

        # Then recursively fetch my decendants
        for child in self.children:
            lnp += child.ln_prior(verbose=verbose)

            # If my child returns negative infinite prob, terminate here.
            if np.isinf(lnp):
                return lnp

        # Pass it up the chain, or back to the main program
        return lnp

    def ln_prob(self, verbose=False):
        """Calculates the natural log of the posterior probability
        (ln_prior + ln_like)"""

        # First calculate ln_prior
        lnp = self.ln_prior(verbose=verbose)

        # Then add ln_prior to ln_like
        if np.isfinite(lnp):
            try:
                return lnp + self.ln_like()
            except:
                if verbose:
                    print("Failed to evaluate ln_like at {}".format(self.name))
                return -np.inf
        else:
            if verbose:
                print("{} ln_prior returned infinite!".format(self.name))
            return lnp

    # Dunder methods that are generally hidden from the user.
    def __get_inherited_parameter_names__(self):
        '''Construct a list of the variable parameters that I have, and append
        it with the variables stored in my parents.

        This is a list of ONLY the names of the parameters.
        '''
        names = []

        # First, get my own parameter names
        names += self.node_par_names

        # Then, fetch the names of my parent's parameters - in order!
        if self.parent is not None:
            names += self.parent.__get_inherited_parameter_names__()

        return names

    def __get_inherited_parameter_vector__(self):
        '''Query all my parents for their parameter vectors. When they've all
        given me them, return the full list.

        Parameter vectors are the
        '''

        # This is where I'll build my list of parameters
        vector = []

        # What are my parameters?
        vector += [getattr(self, name) for name in self.node_par_names]

        # Get my parent's vectors...
        if self.parent is not None:
            vector += self.parent.__get_inherited_parameter_vector__()

        return vector

    def __get_descendant_params__(self):
        '''Get all the Param objects at or below this node'''
        params = []
        node_names = []

        params += [getattr(self, par) for par in self.node_par_names]
        node_names += [self.label for par in self.node_par_names]

        for child in self.children:
            child_params, child_node_names = child.__get_descendant_params__()
            params += child_params
            node_names += child_node_names

        return params, node_names

    def __get_descendant_parameter_vector__(self):
        '''Get a list of the values of the Param objects at or below this node

        The (V)ector contains the (V)alues
        '''
        params, _ = self.__get_descendant_params__()

        # Filter out the entries that are non-variable
        vector = [v.currVal for v in params if v.isVar]

        return vector

    def __get_descendant_parameter_names__(self):
        '''Get the keys for the lower parameter vector'''

        params, names = self.__get_descendant_params__()

        # Filter out the entries that are non-variable
        vector = [v.name+"_"+n for v, n in zip(params, names) if v.isVar]

        return vector

    def __set_parameter_vector__(self, vector_values):
        '''Take a parameter vector, and pop values off the back until all this
        models' variables are set. Then pass the remainder to the children of
        this model, in order.'''

        vector_values = list(vector_values)

        # I need to read off the children backwards
        for child in self.children[::-1]:
            vector_values = child.__set_parameter_vector__(vector_values)

        # Now, add my own.
        # Remember, backwards!
        for name in self.node_varpars[::-1]:
            val = vector_values.pop()
            par = getattr(self, name)
            par.currVal = val

        return vector_values

    def __check_par_assignments__(self):
        '''Loop through my variables, and make sure that the Param.name is the
        same as what I've got listed in self.node_par_names. This is probably
        paranoid, but it makes me feel safer'''

        param_dict = {key: getattr(self, key) for key in self.node_par_names}

        for key, value in param_dict.items():
            if key != value.name:
                fail_msg = "Incorrect parameter name, {} assigned to {}. "
                fail_msg += "Parameters are taken in the order {}".format(
                    value.name, key, self.node_par_names
                )
                raise NameError(fail_msg)

    # Properties to make everything cleaner
    @property
    def name(self):
        '''The name of this object, of the form <class name>_<label>'''
        return "{}_{}".format(self.__class__.__name__, self.label)

    @property
    def parent(self):
        '''My parent <3'''
        return self.__parent

    @parent.setter
    def parent(self, parent):
        '''When setting the parent, I also need to add myself to their list of
        children'''
        self.__parent = parent
        if self.__parent is None:
            pass
        else:
            self.__parent.add_child(self)

    @property
    def children(self):
        return self.__children

    @children.setter
    def children(self, children):
        # I need to preserve the order of the children, so keep as a list.
        if not isinstance(children, list):
            children = list(children)

        # Set the internal variable
        self.__children = children

        # Make sure my children know who's in charge
        for child in self.__children:
            child.__parent = self

    @property
    def par_names(self):
        '''A list of the keys to self.par_val_list'''
        return self.__get_descendant_parameter_names__()

    @property
    def par_val_list(self):
        '''A list of the variable parameter values below this node'''
        return self.__get_descendant_parameter_vector__()

    @par_val_list.setter
    def par_val_list(self, par_val_list):
        if not len(par_val_list) >= len(self.par_val_list):
            raise ValueError('Fail on {}'.format(self.name))
        self.__set_parameter_vector__(par_val_list)

    @property
    def par_dict(self):
        '''A dict of the Param objects ABOVE! this node'''
        return {key: val for key, val in
                zip(self.__get_inherited_parameter_names__(),
                    self.__get_inherited_parameter_vector__())}

    @property
    def is_root(self):
        '''True if I have no parents'''
        return self.parent is None

    @property
    def is_leaf(self):
        '''True if I have no children'''
        return len(self.children) == 0

    # Diagnostic methods
    @property
    def structure(self):
        '''Return the tree structure below me as a str, generated from nx.'''
        self.create_tree()

        return nx.readwrite.tree_data(self.nx_graph, self.name)

    def report_relatives(self):
        '''This is a pretty crappy, inflexible way of doing this. Can I
        come up with a nicer, perhaps recursive way of it?'''
        print("Reporting family tree of {}:".format(self.name))
        try:
            parent = self.parent.name
        except:
            parent = 'None'
        print("    Parent: {}".format(parent))
        print("    Children:")
        for child in self.children:
            print("      {}".format(child.name))
            for grandchild in child.children:
                print("       - {}".format(grandchild.name))

    def report(self, also_relatives=True):
        if also_relatives:
            self.report_relatives()
        print("  Parameter vector, and labels:")
        for par, val in zip(self.par_names, self.par_val_list):
            print("  {:>10}: {}".format(par, val))
        print("\n")

    def create_tree(self, G=None, called=True):
        '''Construct a tree node graph of the model structure.
        Start from the called tier, and work down from there.'''
        if called:
            G = nx.DiGraph()
            G.add_node(self.name)

            for child in self.children:
                # Add the child's children to the graph
                G = child.create_tree(G, called=False)
                # Connect myself to the child
                G.add_edge(self.name, child.name)

        else:
            # Add myself to the graph
            G.add_node(self.name)
            for child in self.children:
                # Add my child as a node, and connect it to me
                G.add_node(child.name)
                G.add_edge(self.name, child.name)
                G = child.create_tree(G, called=False)

            return G

        self.nx_graph = G
        return G

    def plot_data(self, save=False):
        '''Cycles down the tree until we find a leaf. Then, call it's data
        plotting method, X.__plot_data.
        If save is True, save the figure'''
        if self.is_leaf:
            if hasattr(self, '_plot_data'):
                plot = getattr(self, '_plot_data')
                plot(save=save)
        else:
            for child in self.children:
                child.plot_data(save=save)

    def draw(self):
        '''Draw a hierarchical node map of the model.'''

        G = self.create_tree()
        pos = self.hierarchy_pos(G)

        # Figure has two inches of width per node
        figsize = (2*float(G.number_of_nodes()), 8.0)
        print("Figure will be {}".format(figsize))

        fig, ax = plt.subplots(figsize=figsize)

        nx.draw(
            G,
            ax=ax,
            pos=pos, with_labels=True,
            node_color='grey', font_weight='heavy')
        plt.show()

    def hierarchy_pos(self, G,
                      root=None, width=1.,
                      vert_gap=0.2, vert_loc=0, xcenter=0.5):
        '''
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
        Licensed under Creative Commons Attribution-Share Alike

        If the graph is a tree this will return the positions to plot this in a
        hierarchical layout.

        G: the graph (must be a tree)

        root: the root node of current branch
        - if the tree is directed and this is not given,
        the root will be found and used
        - if the tree is directed and this is given, then
        the positions will be just for the descendants of this node.
        - if the tree is undirected and not given,
        then a random choice will be used.

        width: horizontal space allocated for this branch - avoids overlap with
        other branches

        vert_gap: gap between levels of hierarchy

        vert_loc: vertical location of root

        xcenter: horizontal location of root
        '''
        if not nx.is_tree(G):
            fail_msg = 'cannot use hierarchy_pos on a graph that is not a tree'
            raise TypeError(fail_msg)

        if root is None:
            if isinstance(G, nx.DiGraph):
                # Allows back compatibility with nx version 1.11
                root = next(iter(nx.topological_sort(G)))
            else:
                import random
                root = random.choice(list(G.nodes))

        return self._hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def _hierarchy_pos(self, G, root,
                       width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                       pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        children = list(G.neighbors(root))

        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)

        if len(children) != 0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = self._hierarchy_pos(
                    G, child,
                    width=dx, vert_gap=vert_gap,
                    vert_loc=vert_loc-vert_gap, xcenter=nextx,
                    pos=pos, parent=root
                )

        return pos


# Subclasses.
class Eclipse(Model):
    '''Subclass of Model, specifically for storing a single eclipse. Lightcurve
    data is stored on this level.

    Inputs:
    -------
      lightcurve; Lightcurve:
        A Lightcurve object, containing data
      iscomplex; bool:
        Do we use the complex model?
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
    '''

    # Define this subclasses parameters
    node_par_names = (
        'dFlux', 'sFlux', 'ulimb', 'rdisc',
        'scale', 'az', 'fis', 'dexp', 'phi0'
    )

    def __init__(self, lightcurve, iscomplex, *args, **kwargs):
        # If we're a complex eclipse, add the complex parameter names
        if iscomplex:
            complex_parNames = (
                'exp1', 'exp2', 'yaw', 'tilt'
            )
            self.node_par_names += complex_parNames

        self.iscomplex = iscomplex

        super().__init__(*args, **kwargs)

        # If the lightcurve is a Lightcurve object, save it. Otherwise,
        # read it in from the file.
        if isinstance(lightcurve, Lightcurve):
            self.lc = lightcurve
        elif isinstance(lightcurve, str):
            self.lc = Lightcurve.from_calib(lightcurve)
        else:
            msg = "Argument lightcurve is not a string or Lightcurve! "
            msg += "Got {}".format(lightcurve)
            raise TypeError(msg)

        self.initCV()

    def initCV(self):
        '''Try to create a CV.
        If we cant construct the input list of params, set cv to None.'''

        self.cv = CV(self.cv_parlist)
        # print("Created a CV object for eclipse {}...".format(self.label))

    def calc(self, plot=False):
        '''Calculate the chisq of this eclipse, against the data stored in its
        lightcurve object.'''

        if self.cv is None:
            self.initCV()

        # Get the model CV lightcurve across our data.
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)

        # Calculate the chisq of this model.
        chisq = (self.lc.y - flx)**2
        chisq = np.sum(chisq)

        if plot:
            self.lc.plot(flx)

        return chisq

    def ln_prior(self, verbose):
        '''At the eclipse level, three constrains must be validated for each
        leaf of the tree.

        - Is the disc large enough to precess? We can't handle superhumping!
        - Is the BS scale unreasonably large, enough to cause the disc model
            to be inaccurate?
        - Is the azimuth of the BS out of range?

        If other constraints on the level of the individual eclipse are
        necessary, they should go in here.
        '''

        # Before we start, I'm going to collect the necessary parameters. By
        # only calling this once, we save a little effort.
        par_dict = self.par_dict

        ##############################################
        # ~~ Is the disc large enough to precess? ~~ #
        ##############################################

        # Defined the maximum size of the disc before it starts precessing, as
        # a fraction of Roche Radius
        rdisc_max_a = 0.46

        # get the location of the L1 point from q
        q = par_dict['q'].currVal
        xl1 = roche.xl1(q)

        # Get the rdisc, scaled to the Roche Radius
        rdisc = par_dict['rdisc'].currVal
        rdisc_a = rdisc * xl1

        if rdisc_a > rdisc_max_a:
            if verbose:
                msg = "The disc radius of {} is large enough to precess!"
                msg += "Value: {:.3f}"
                msg.format(self.name, rdisc)
                print(msg)
            return -np.inf

        ##############################################
        # ~~~~~ Is the BS scale physically OK? ~~~~~ #
        ##############################################
        # We're gonna check to see if the BS scle is #
        # in the range rwd/3 < (BS scale) < rwd*3.   #
        # If it isn't, then it's either too          #
        # concentrated to make sense, or so large    #
        # that our approximation of a smooth disc is #
        # no longer a good idea.                     #
        #                                            #
        ##############################################

        # Get the WD radius.
        rwd = par_dict['rwd'].currVal

        # Enforce the BS scale being within these limits
        rmax = rwd * 3.
        rmin = rwd / 3.

        scale = par_dict['scale'].currVal

        if scale > rmax or scale < rmin:
            if verbose:
                msg = "Leaf {} has a BS scale that lies outside valid range!"
                print(msg.format(self.name))
                print("Rwd: {:.3f}".format(rwd))
                print("Scale: {:.3f}".format(scale))
                print("Range: {:.3f} - {:.3f}".format(rmin, rmax))

            return -np.inf

        ##############################################
        # ~~~~~ Does the stream miss the disc? ~~~~~ #
        ##############################################

        slope = 80.0
        try:
            # q, rdisc_a were previously retrieved
            az = par_dict['az'].currVal

            # If the stream does not intersect the disc, this throws an error
            x, y, vx, vy = roche.bspot(q, rdisc_a)

            # Find the tangent to the disc
            alpha = np.degrees(np.arctan2(y, x))

            # If alpha is negative, the BS lags the disc.
            # However, the angle has to be less than 90 still!
            if alpha < 0:
                alpha = 90 - alpha

            # Disc tangent
            tangent = alpha + 90

            # Calculate the min and max azimuthes, using the tangent and slope
            minaz = max(0, tangent-slope)
            maxaz = min(178, tangent+slope)

            if az < minaz or az > maxaz:
                return -np.inf

        except:
            if verbose:
                msg = "The mass stream of leaf {} does not intersect the disc!"
                print(msg.format(self.name))
            return -np.inf

        # If we pass all that, then calculate the ln_prior normally
        return super().ln_prior(verbose=verbose)

    def _plot_data(self, save=False, figsize=(11., 8.)):
        '''When a (grand)parent is asked to plot its data, it will recursively
        ask its children for the X._plot_data method. This is that method,
        terminating the recursion.

        If save is True, a copy of the figure is saved.
        '''

        cv_par_name_list = [
            'wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'q', 'dphi',
            'rdisc', 'ulimb', 'rwd', 'scale', 'az', 'fis', 'dexp', 'phi0'
        ]
        if self.iscomplex:
            cv_par_name_list.extend(['exp1', 'exp2', 'yaw', 'tilt'])

        # Re-init my CV object with the current params.
        self.initCV()

        # Generate the lightcurve of the total, and the components.
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)
        wd_flx = self.cv.ywd
        sec_flx = self.cv.yrs
        BS_flx = self.cv.ys
        disc_flx = self.cv.yd

        # print("This model has a chisq of {:.3f}".format(self.calc()))

        # Start the plotting area
        fig, axs = plt.subplots(2, sharex=True, figsize=figsize)

        # Plot the data first. Also do errors
        axs[0].errorbar(
            self.lc.x, self.lc.y,
            yerr=self.lc.ye,
            linestyle='none', ecolor='grey', zorder=1
            )
        axs[0].step(self.lc.x, self.lc.y, where='mid', color='black')

        # Plot the model over the data
        axs[0].plot(self.lc.x, wd_flx, color='lightblue', label='WD')
        axs[0].plot(self.lc.x, sec_flx, color='magenta', label='Sec')
        axs[0].plot(self.lc.x, BS_flx, color='darkblue', label='BS')
        axs[0].plot(self.lc.x, disc_flx, color='brown', label='Disc')
        axs[0].plot(self.lc.x, flx, color='red')
        axs[0].legend()

        # Plot the errorbars
        axs[1].errorbar(
            self.lc.x, self.lc.y-flx,
            yerr=self.lc.ye,
            linestyle='none', ecolor='grey', zorder=1
            )
        axs[1].step(self.lc.x, self.lc.y-flx, where='mid', color='darkred')

        # 0 residuals line, to guide the eye
        axs[1].axhline(0.0, linestyle='--', color='black', alpha=0.7,
                       zorder=0)

        # Labelling. Top one gets title, bottom one gets x label
        axs[0].set_title(self.lc.name)
        axs[0].set_ylabel('Flux, mJy')

        axs[1].set_xlabel('Phase')
        axs[1].set_ylabel('Residual Flux, mJy')

        # Arrange the figure on the page, and show it
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        if save:
            fig_name = self.lc.name.replace('.calib', '.pdf')
            plt.savefig(fig_name)
        plt.show()

        return

    @property
    def cv_parlist(self):
        '''Construct the parameter list needed by the CV'''
        cv_par_name_list = [
            'wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'q', 'dphi',
            'rdisc', 'ulimb', 'rwd', 'scale', 'az', 'fis', 'dexp', 'phi0'
        ]
        if self.iscomplex:
            cv_par_name_list.extend(['exp1', 'exp2', 'tilt', 'yaw'])

        par_dict = self.par_dict

        try:
            cv_parlist = [par_dict[key].currVal for key in cv_par_name_list]
            return cv_parlist
        except:
            msg = 'Could not construct the CV parameter list from {}'
            msg.format(self.name)
            raise KeyError(msg)


class Band(Model):
    '''Subclass of Model, specific to observation bands. Contains the eclipse
    objects taken in this band.

    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
    '''

    # What kind of parameters are we storing here?
    node_par_names = ('wdFlux', 'rsFlux')


class LCModel(Model):
    '''Top layer Model class. Contains Bands, which contain Eclipses.
        Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Model, optional:
        The parent of this node.
      children; list(Model), or Model:
        The children of this node. Single Model is also accepted
    '''

    # Set the parameter names for this layer
    node_par_names = ('q', 'dphi', 'rwd')

    def ln_prior(self, verbose=False):
        '''Before we calculate the ln_prior of myself or my children, I check
        that my parameters are valid. I check that dphi is not too large for
        the current value of q.

        If other constraints on the core parameters become necessary, they
        should go here. If these tests fail, -np.inf is immediately returned.
        '''
        lnp = 0.0

        # Check that dphi is within limits
        tol = 1e-6

        dphi = getattr(self, 'dphi').currVal
        q = getattr(self, 'q').currVal

        try:
            # Get the value of dphi that we WOULD have at an
            # inclination of 90 degrees
            maxphi = roche.findphi(q, 90.0)

            # If dphi is out of range, return negative inf.
            if dphi > (maxphi - tol):
                if verbose:
                    msg = "{} has a dphi out of tolerance!\nq: {:.3f}"
                    msg += "\ndphi: {:.3f}, max: {:.3f} - {:.3g}"
                    msg += "\nReturning inf.\n\n"

                    msg.format(self.name, q, dphi, maxphi, tol)

                    print(msg)
                return -np.inf

        except:
            # If we get here, then roche couldn't find a dphi for this q.
            # That's bad!
            if verbose:
                msg = "Failed to calculate a value of dphi at node {}"
                print(msg.format(self.name))
            return -np.inf

        # Then, if we pass this, move on to the 'normal' ln_prior calculation.
        lnp += super().ln_prior(verbose=verbose)
        return lnp


class GPEclipse(Eclipse):
    '''This is a subclass of the Eclipse class. It uses the Gaussian Process.

    This version will, rather than evaluating via chisq, evaluate the
    likelihood of the model by calculating the residuals between the model
    and the data and computing the likelihood of those data given certain
    Gaussian Process hyper-parameters. '''

    # Set the initial values of q, rwd, and dphi. These will be used to
    # caclulate the location of the GP changepoints. Setting to initially
    # unrealistically high values will ensure that the first time
    # calcChangepoints is called, the changepoints are calculated.
    _olddphi = 9e99
    _oldq = 9e99
    _oldrwd = 9e99

    # _dist_cp is initially set to whatever, it will be overwritten anyway.
    _dist_cp = 9e99

    def __init__(self, ampin_GP, ampout_GP, tau_GP, *args, **kwargs):

        # GP hyperparameters
        self.ampin = ampin_GP
        self.ampout = ampout_GP
        self.tau = tau_GP

        super().__init__(*args, **kwargs)

    def calcChangepoints(self):
        '''Caclulate the WD ingress and egresses, i.e. where we want to switch
        on or off the extra GP amplitude'''

        # Also get object for dphi, q and rwd as this is required to determine
        # changepoints
        dphi = self.par_dict['dphi']
        q = self.par_dict['q']
        rwd = self.par_dict['rwd']
        phi0 = self.par_dict['phi0']

        dphi_change = np.fabs(self._olddphi - dphi.currVal) / dphi.currVal
        q_change = np.fabs(self._oldq - q.currVal) / q.currVal
        rwd_change = np.fabs(self._oldrwd - rwd.currVal) / rwd.currVal

        # Check to see if our model parameters have changed enough to
        # significantly change the location of the changepoints.
        if (dphi_change > 1.2) or (q_change > 1.2) or (rwd_change > 1.2):
            # Calculate inclination
            inc = roche.findi(q.currVal, dphi.currVal)

            # Calculate wd contact phases 3 and 4
            phi3, phi4 = roche.wdphases(q.currVal, inc, rwd.currVal, ntheta=10)

            # Calculate length of wd egress
            dpwd = phi4 - phi3

            # Distance from changepoints to mideclipse
            dist_cp = (dphi.currVal+dpwd)/2.
        else:
            # Use the old values
            dist_cp = self._dist_cp

        # Find location of all changepoints
        min_ecl = int(np.floor(self.lc.x.min()))
        max_ecl = int(np.ceil(self.lc.x.max()))

        eclipses = [e for e in range(min_ecl, max_ecl+1)
                    if np.logical_and(e > self.lc.x.min(),
                                      e < 1+self.lc.x.max()
                                      )
                    ]

        changepoints = []
        for e in eclipses:
            # When did the last eclipse end?
            egress = (e-1) + dist_cp
            # When does this eclipse start?
            ingress = e - dist_cp
            changepoints.append([egress, ingress])

        # save these values for speed
        if (dphi_change > 1.2) or (q_change > 1.2) or (rwd_change > 1.2):
            self._dist_cp = dist_cp
            self._oldq = q.currVal
            self._olddphi = dphi.currVal
            self._oldrwd = rwd.currVal

        return changepoints

    def createGP(self):
        """Constructs a kernel, which is used to create Gaussian processes.

        Using values for the two hyperparameters (amp,tau), amp_ratio and dphi,
        this function: creates kernels for both inside and out of eclipse,
        works out the location of any changepoints present, constructs a single
         (mixed) kernel and uses this kernel to create GPs"""

        # Get objects for ampin_gp, ampout_gp, tau_gp and find the exponential
        # of their current values
        ln_ampin = self.ampin
        ln_ampout = self.ampout
        ln_tau = self.tau

        ampin = np.exp(ln_ampin.currVal)
        ampout = np.exp(ln_ampout.currVal)
        tau = np.exp(ln_tau.currVal)

        # Calculate kernels for both out of and in eclipse WD eclipse
        # Kernel inside of WD has smaller amplitude than that of outside
        # eclipse.

        # First, get the changepoints
        changepoints = self.calcChangepoints()

        # We need to make a fairly complex kernel.
        # Global flicker
        kernel = ampin * george.kernels.Matern32Kernel(tau)
        # inter-eclipse flicker
        for gap in changepoints:
            kernel += ampout * george.kernels.Matern32Kernel(tau, block=gap)

        # Use that kernel to make a GP object
        georgeGP = george.GP(kernel, solver=george.HODLRSolver)

        return georgeGP

    def calc(self, plot=False):
        '''Overwrites the vanilla Eclipse calc function. We no longer want to
        calculate chisq, rather return the flux of the model.

        Calculate the residuals of the model of this eclipse, against the data
        stored in its lightcurve object.'''

        if self.cv is None:
            self.initCV()

        # TODO: When the model moves far enough, re-initialise the CV?

        # Get the model CV lightcurve across our data.
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)

        # Calculate the residuals of this model.
        resids = self.lc.y - flx

        # Check for bugs in model
        if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
            warnings.warn('GP gave nan or inf answers')
            return -np.inf

        # For This eclipse, create (and compute) Gaussian process and calculate
        #  the model
        gp = self.createGP()
        gp.compute(self.lc.x, self.lc.ye)

        # The 'quiet' argument tells the GP to return -inf when you get an
        # invalid kernel, rather than throwing an exception.
        ln_like = gp.log_likelihood(resids, quiet=True)

        if plot:
            self.lc.plot(flx)

        return ln_like

    def ln_like(self):
        '''No longer statistically accurate to return half the chisq,
        just get the likelihood from the GP and return that. '''
        return self.chisq()
