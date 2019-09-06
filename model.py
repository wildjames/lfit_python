'''
Base classes for the MCMC fitting routine. Allows the creation of a
hierarchical model structure, that can also track the prior knowledge of the
parameters of that model.
'''

import sys
import os
import warnings

import george
import networkx as nx
import numpy as np
import scipy.integrate as intg
import scipy.stats as stats
from matplotlib import pyplot as plt

import inspect

TINY = -np.inf


class Prior(object):
    '''a class to represent a prior on a parameter, which makes calculating
    prior log-probability easier.

    Priors can be of five types:
        gauss, gaussPos, uniform, log_uniform and mod_jeff

    gauss is a Gaussian distribution, and is useful for parameters with
    existing constraints in the literature
    gaussPos is like gauss but enforces positivity
    Gaussian priors are initialised as Prior('gauss',mean,stdDev)

    uniform is a uniform prior, initialised like:
        Prior('uniform',low_limit,high_limit)
    uniform priors are useful because they are 'uninformative'

    log_uniform priors have constant probability in log-space. They are the
    uninformative prior for 'scale-factors', such as error bars (look up
    Jeffreys prior for more info)

    mod_jeff is a modified jeffries prior - see Gregory et al 2007
    they are useful when you have a large uncertainty in the parameter value,
    so a jeffreys prior is appropriate, but the range of allowed values
    starts at 0

    they have two parameters, p0 and pmax.
    they act as a jeffrey's prior about p0, and uniform below p0. typically
    set p0=noise level
    '''
    def __init__(self, type, p1, p2):
        assert type in ['gauss', 'gaussPos', 'uniform', 'log_uniform', 'mod_jeff']
        self.type = type
        self.p1 = p1
        self.p2 = p2
        if type == 'log_uniform' and self.p1 < 1.0e-30:
            warnings.warn('lower limit on log_uniform prior rescaled from %f to 1.0e-30' % self.p1)
            self.p1 = 1.0e-30
        if type == 'log_uniform':
            self.normalise = 1.0
            self.normalise = np.fabs(intg.quad(self.ln_prob, self.p1, self.p2)[0])
        if type == 'mod_jeff':
            self.normalise = np.log((self.p1+self.p2)/self.p1)

    def ln_prob(self, val):
        if self.type == 'gauss':
            prob = stats.norm(scale=self.p2, loc=self.p1).pdf(val)
            if prob > 0:
                return np.log(prob)
            else:
                return TINY
        elif self.type == 'gaussPos':
            if val <= 0.0:
                return TINY
            else:
                prob = stats.norm(scale=self.p2, loc=self.p1).pdf(val)
                if prob > 0:
                    return np.log(prob)
                else:
                    return TINY
        elif self.type == 'uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0/np.abs(self.p1-self.p2))
            else:
                return TINY
        elif self.type == 'log_uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0 / self.normalise / val)
            else:
                return TINY
        elif self.type == 'mod_jeff':
            if (val > 0) and (val < self.p2):
                return np.log(1.0 / self.normalise / (val+self.p1))
            else:
                return TINY


class Param(object):
    '''A Param needs a starting value, a current value, and a prior
    and a flag to state whether is should vary'''
    def __init__(self, name, startVal, prior, isVar=True):
        self.name = name
        self.startVal = startVal
        self.prior = prior
        self.currVal = startVal
        self.isVar = isVar

    @classmethod
    def fromString(cls, name, parString):
        fields = parString.split()
        val = float(fields[0])
        priorType = fields[1].strip()
        priorP1 = float(fields[2])
        priorP2 = float(fields[3])
        if len(fields) == 5:
            isVar = bool(int(fields[4]))
        else:
            isVar = True
        return cls(name, val, Prior(priorType, priorP1, priorP2), isVar)

    @property
    def isValid(self):
        return np.isfinite(self.prior.ln_prob(self.currVal))


class Node:
    r'''
    Inputs:
    -------
      label, str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted

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
    e.g., Trunk.dynasty_par_vals contains the parameters for all nodes, but
    Branch2.dynasty_par_vals contains only those of Branch2, leaf3, and leaf4.
    Setting X.dynasty_par_vals sorts the parameters to the correct nodes,
    provided it's in the correct order (which can be retrieved with
    X.dynasty_par_names)

    Conversely, leaf4.ancestor_param_dict moves the other way. It will contain
    ONLY the parameters of leaf4, Branch2, and Trunk.

    Should be subclassed with the self.node_par_names variable defined, as this
    is a blank slate. Without that variable, this model cannot store
    parameters. There may be usecases where this can be exploited to group
    branches or leaves without them sharing any parameters.
    '''

    # Init the node_par_names to be empty and a tuple.
    # Change this when you subclass Node!
    node_par_names = ()

    def __init__(self, label, parameter_objects, parent=None, children=None,
                 DEBUG=None):
        '''Initialse the node. Does the following:

        - Store parameter values to attributes named after the parameter names
        - Store the object defined as my parent
        - Store a list of objects defined as my children
        - Checks that my Param objects were stored correctly.

        Inputs:
        -------
          label: str
            A label for this node. The name attribute will be the node class
            name joined with this label.
          parameter_objects: list of Param
            The parameters to be stored on this node.
          parent: Node object, Optional
            The node that this node is a child of.
          children: Node, list of Node, Optional
            The children of this node.
          DEBUG: bool
            A useful debugging flag for you to use.
        '''
        # Handle the family
        if children is None:
            children = []
        self.children = children
        self.parent = parent

        # If the user tells us a debugging flag, use it.
        if isinstance(DEBUG, bool):
            self.DEBUG = DEBUG
        # Otherwise, inherit my parent's debugging flag
        elif self.parent is not None:
            self.DEBUG = self.parent.DEBUG
        # unless I don't have one. Then default to False
        else:
            self.DEBUG = False

        # I expect my parameter values to be fed in as a list. If they're not
        # a list, assume I have a single Param object, and wrap it in a list.
        parameter_objects = list(parameter_objects)

        # Make sure our label is valid
        assert isinstance(label, str), "Label must be a string!"
        self.label = label

        # Check that the user defined their parameter names!
        if len(self.node_par_names) != len(parameter_objects):
            fail_msg = 'I recieved the wrong number of parameters!'
            fail_msg += ' Expect: \n{}\nGot:\n{}'.format(
                self.node_par_names,
                [getattr(param, 'name') for param in parameter_objects]
            )
            raise TypeError(fail_msg)

        # Add the parameters to the self.XXX.
        for par in parameter_objects:
            setattr(self, par.name, par)

        self.log('base.__init__', "Successfully did the base Node init")

    # Tree handling methods
    def search_par(self, label, name):
        '''Search the tree recursively downwards, and return the Param.
        Returns None if the Param is not found.

        Inputs:
        -------
          label: str
            The Param I'm searching for will be associated with a node
            having this label.
          name: str
            The name of the Param object. I'm looking for

        Returns:
        --------
          Param, None if the search fails
            The Param object to be searched.
        '''

        self.log('base.search_par', "Searching for a Param called {}, on a Node labelled {}".format(name, label))

        # If I'm the desired node, get my parameter
        if self.label == label:
            self.log('base.search_par', "I am that Node!")
            return getattr(self, name)
        # Otherwise, check my children.
        else:
            self.log('base.search_par', "Searching my children for that Node.")
            for child in self.children:
                val = child.search_par(label, name)
                if val is not None:
                    return val
            self.log('base.search_par', "Could not find that node.")
            return None

    def search_Node(self, class_type, label):
        '''Search for a node below me of class_type, with the label requested.
        Returns None if this is not found.

        Inputs:
        -------
          class_type: str
            The nodes will be checked that their class name is this string
          label: str
            The nodes will be checked that their label is this string

        Outputs:
        --------
          Node, None is the search fails
            The node that was requested.
        '''
        self.log('base.search_Node', "Searching for a Node of class type {}, with a label {}".format(class_type, label))
        if self.name == "{}:{}".format(class_type, label):
            self.log('base.search_Node', "I am that node. Returning self")
            return self
        else:
            self.log('base.search_Node', "Checking my children")
            for child in self.children:
                val = child.search_Node(class_type, label)
                if val is not None:
                    return val
                else:
                    pass
            self.log('base.search_Node', "Could not find that node.")
            return None

    def search_node_type(self, class_type, nodes=None):
        '''Construct a set of all the nodes of a given type below me

        Inputs:
        -------
          class_type: str
            If the node class contains this string, it will be added.
          nodes: set of Node, Optional
            The existing list of nodes that will be extended with my result.

        Outputs:
        --------
          nodes: set of Node
            The search result.
        '''
        self.log('base.search_node_type', "Constructing a set of Nodes of type {}".format(class_type))

        if nodes is None:
            nodes = set()

        for child in self.children:
            child_nodes = child.search_node_type(class_type, nodes)
            nodes = nodes.union(child_nodes)

        if class_type in str(self.__class__.__name__):
            nodes.add(self)

        self.log('base.search_node_type', "Returning: \n{}".format(nodes))
        return nodes

    def add_child(self, children):
        '''Add children to my list of children

        Inputs:
        -------
          children: Node, or list of Node
            Add this to my list of children. They will be altered to
            have this node as a parent.
        '''
        self.log('base.add_child', "Adding: \n{}\nto my existing list of children, which was \n{}".format(children, self.children))
        if not isinstance(children, list):
            children = [children]

        # Set the children.
        self.children.extend(children)


    # # # # # # # # # # # # # #
    # Tree evaluation methods #
    # # # # # # # # # # # # # #

    def __call_recursive_func__(self, name, *args, **kwargs):
        '''
        Descend the model heirarchy, calling a function at each leaf.

        This is used, for example, to evaluate chisq or ln_like for a given model,
        where we want to sum this quantity for all fully defined models.

        Inputs:
        -------
          name: str
            The name of the function to be called
          *args, **kwargs
            Arguments to be passed to the function.

        Outputs:
        --------
          float:
            The sum of the function called at each relevant node.
        '''
        # self.log("base.__call_recursive_func", "Calling the function {} recursively, passing it the args:\n{}\nkwargs:\n{}".format(name, args, kwargs))
        val = 0.0
        if self.is_leaf:
            self.log("base.__call_recursive_func", "Reached the bottom of the Tree with no function by that name.")
            raise NotImplementedError('must overwrite {} on leaf nodes of model'.format(
                name
            ))
        for child in self.children:
            func = getattr(child, name)
            val += func(*args, **kwargs)
            if np.any(np.isinf(val)):
                # we've got an invalid model, no need to evaluate other leaves
                self.log("base.__call_recursive_func", "The function {} called on {}, but returned an inf.".format(name, child.name))
                return val
        return val

    def chisq(self, *args, **kwargs):
        '''Returns the sum of my children's  chisqs. Must be overwritten on
        leaf nodes, or nodes capable of evaluating a model.'''
        return self.__call_recursive_func__('chisq', *args, **kwargs)

    def ln_like(self, *args, **kwargs):
        '''Calculate the log likelihood'''
        return self.__call_recursive_func__('ln_like', *args, **kwargs)

    def ln_prior(self, verbose=False):
        """Return the natural log of the prior probability of the Param objects
        below this node.

        If model has more prior information not captured in the priors of the
        parameters, the details of such additional prior information must be
        codified in subclass methods!."""
        self.log('base.ln_prior', "Summing the ln_prior of all my Params ({} Params)".format(len(self.node_par_names)))

        # Start at a log prior probablity of 0. We'll add to this for each node
        lnp = 0.0

        # Get the sum of this node's variable's prior probs
        for param in [getattr(self, name) for name in self.node_par_names]:
            if param.isValid and param.isVar:
                lnp += param.prior.ln_prob(param.currVal)

            elif not param.isValid:
                if verbose:
                    print("Param {} in {} is invalid!".format(
                        param.name, self.name))
                self.log('base.ln_prior', "Param {} in {} is invalid!".format(
                        param.name, self.name))
                return -np.inf

        # Reporting, if necessary
        if verbose:
            print("{} has the following Params:".format(self.name))
            for i, _ in enumerate(self.node_par_names[::4]):
                j = 4*i
                k = j+4
                print(self.node_par_names[j:k])
            print("The sum of parameter ln_priors of {} is {:.3f}\n".format(
                self.name, lnp))

        # self.log('base.ln_prior', "My ln_prior is {}. Gathering my descendant ln_priors".format(lnp))

        # Then recursively fetch my decendants
        for child in self.children:
            lnp += child.ln_prior(verbose=verbose)

            # If my child returns negative infinite prob, terminate here.
            if np.isinf(lnp):
                self.log('base.ln_prior', "My child, {}, yielded an inf ln_prior".format(child.name))
                return lnp

        # Pass it up the chain, or back to the main program
        self.log('base.ln_prior', "I computed a total ln_prior at and below me of {}".format(lnp))
        return lnp

    def ln_prob(self, verbose=False):
        """Calculates the natural log of the posterior probability
        (ln_prior + ln_like)"""

        # First calculate ln_prior
        lnp = self.ln_prior(verbose=verbose)

        # Then add ln_prior to ln_like
        if np.isfinite(lnp):
            try:
                lnp = lnp + self.ln_like()
                self.log('base.ln_prob', "Calculated ln_prob = ln_prior + ln_like = {}".format(lnp))
                return lnp
            except:
                if verbose:
                    print("Failed to evaluate ln_like at {}".format(self.name))
                self.log('base.ln_prob', "Failed to evaluate ln_prob!")
                return -np.inf
        else:
            if verbose:
                print("{} ln_prior returned infinite!".format(self.name))
            self.log('base.ln_prob', "{} ln_prior returned infinite!".format(self.name))
            return lnp

    # Dunder methods that are generally hidden from the user.
    def __get_inherited_parameter_names__(self):
        '''Construct a list of the variable parameters that I have, and append
        it with the names of those stored in my parents.

        This is a list of ONLY the names of the parameters, regardless of if
        they're variable.
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

        Outputs:
        --------
          list of Param objects
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
        '''Get all the Param objects at or below this node

        Outputs:
        --------
          list of Params,
            All the Param objects of the nodes descended from this node.
          list of node labels,
            The node label corresponding to the Param at the corresponding
            index. Has the same shape as the list of Params.
        '''
        params = []
        node_names = []

        params += [getattr(self, par) for par in self.node_par_names]
        node_names += [self.label for par in self.node_par_names]

        for child in self.children:
            child_params, child_node_names = child.__get_descendant_params__()
            params.extend(child_params)
            node_names.extend(child_node_names)

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
        vector = list(vector_values)

        # I need to read off the children backwards
        for child in self.children[::-1]:
            vector = child.__set_parameter_vector__(vector)

        # Now, add my own.
        # Remember, backwards!
        for name, val in zip(self.node_varpars[::-1], vector[::-1]):
            par = getattr(self, name)
            par.currVal = val

        n_used = len(vector) - len(self.node_varpars)
        return vector[:n_used]

    def __check_par_assignments__(self):
        '''Loop through my variables, and make sure that the Param.name is the
        same as what I've got listed in self.node_par_names. This is probably
        paranoid, but it makes me feel safer'''

        param_dict = {key: getattr(self, key) for key in self.node_par_names}

        for key, value in param_dict.items():
            if key != value.name:
                fail_msg = "Incorrect parameter name, {} assigned to {}. \nParameters are taken in the order {}".format(
                    value.name, key, self.node_par_names
                )
                raise NameError(fail_msg)

    # Properties to make everything cleaner
    @property
    def name(self):
        '''The name of this object, of the form <class name>_<label>'''
        return "{}:{}".format(self.__class__.__name__, self.label)

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
        '''Set the children list to children.

        If the child already has a parent, remove the child from the
        ex-parent's children list.

        Set the childs parent to this node.
        '''

        # I need to preserve the order of the children, so keep as a list.
        if not isinstance(children, list):
            children = list(children)

        # Set the internal variable
        self.__children = children

        # Make sure my children know who's in charge
        for child in self.__children:
            child.__parent = self

    @property
    def dynasty_par_names(self):
        '''A list of the keys to self.dynasty_par_vals'''
        return self.__get_descendant_parameter_names__()

    @property
    def dynasty_par_vals(self):
        '''A list of the variable parameter values below this node'''
        return self.__get_descendant_parameter_vector__()

    @dynasty_par_vals.setter
    def dynasty_par_vals(self, dynasty_par_vals):
        if not len(dynasty_par_vals) >= len(self.dynasty_par_vals):
            raise ValueError('Wrong vector length on {} - Expected {}, got {}'.format(self.name, len(self.dynasty_par_vals), len(dynasty_par_vals)))
        self.__set_parameter_vector__(dynasty_par_vals)

    @property
    def ancestor_param_dict(self):
        '''A dict of the Param objects ABOVE! this node'''
        return {key: val for key, val in
                zip(self.__get_inherited_parameter_names__(),
                    self.__get_inherited_parameter_vector__())}

    @property
    def ancestor_par_names(self):
        '''Construct a list of the variable parameters that I have, and append
        it with the names of those stored in my parents.

        This is a list of ONLY the names of the parameters, regardless of if
        they're variable.
        '''
        return self.__get_inherited_parameter_names__()

    @property
    def node_varpars(self):
        '''Returns the list of THIS node's variable parameter names.'''
        varpars = []
        for name in self.node_par_names:
            par = getattr(self, name)
            if par.isVar:
                varpars.append(par.name)

        return varpars

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

    @property
    def DEBUG(self):
        return self.__DEBUG

    @DEBUG.setter
    def DEBUG(self, flag):
        self.__DEBUG = flag

    def log(self, called_by, message='\n', log_stack=False):
        '''
        Logging function. Writes the node name, and the current function stack
        so the dev can trace what functions are calling what. Writes a message
        if the user asks it to.
        '''
        if not self.DEBUG:
            return

        # the call to inspect.stack() takes a looooong time (~ms)
        if log_stack:
            stack = ["File {}, line {}, function {}".format(x.filename, x.lineno, x.function) for x in inspect.stack()][::-1]
            stack = "\n     ".join(stack)

        # Construct an output filename
        my_fname = "{}.txt".format(os.getpid())
        oname = os.path.join('DEBUGGING', my_fname)

        if not os.path.isdir("DEBUGGING"):
            os.mkdir("DEBUGGING")

        if not message.endswith('\n'):
            message += "\n"

        with open(oname, 'a+') as f:
            f.write('*'*150 + "\n")
            f.write("--> Logger called by function {} in node {}\n".format(called_by, self.name))
            if log_stack:
                f.write("--> The function stack is \n     {}\n".format(stack))
            f.write(message)
            f.write('~'*150 + "\n\n\n")


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
        for par, val in zip(self.dynasty_par_names, self.dynasty_par_vals):
            print("  {:>10s} = {:<.3f}".format(par, val))
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

    # def draw(self, figsize=None):
    #     '''Draw a hierarchical node map of the model.'''

    #     G = self.create_tree()
    #     pos = self.hierarchy_pos(G)

    #     if figsize is None:
    #         # Figure has two inches of width per node
    #         figsize = (2*float(G.number_of_nodes()), 8.0)
    #         print("Figure will be {}".format(figsize))

    #     _, ax = plt.subplots(figsize=figsize)

    #     nx.draw(
    #         G,
    #         ax=ax,
    #         pos=pos, with_labels=True,
    #         node_color='grey', font_weight='heavy')

    #     return ax

    # def hierarchy_pos(self, G,
    #                   root=None, width=1.,
    #                   vert_gap=0.2, vert_loc=0, xcenter=0.5):
    #     '''
    #     From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    #     Licensed under Creative Commons Attribution-Share Alike

    #     If the graph is a tree this will return the positions to plot this in a
    #     hierarchical layout.

    #     G: the graph (must be a tree)

    #     root: the root node of current branch
    #     - if the tree is directed and this is not given,
    #     the root will be found and used
    #     - if the tree is directed and this is given, then
    #     the positions will be just for the descendants of this node.
    #     - if the tree is undirected and not given,
    #     then a random choice will be used.

    #     width: horizontal space allocated for this branch - avoids overlap with
    #     other branches

    #     vert_gap: gap between levels of hierarchy

    #     vert_loc: vertical location of root

    #     xcenter: horizontal location of root
    #     '''
    #     if not nx.is_tree(G):
    #         fail_msg = 'cannot use hierarchy_pos on a graph that is not a tree'
    #         raise TypeError(fail_msg)

    #     if root is None:
    #         if isinstance(G, nx.DiGraph):
    #             # Allows back compatibility with nx version 1.11
    #             root = next(iter(nx.topological_sort(G)))
    #         else:
    #             import random
    #             root = random.choice(list(G.nodes))

    #     return self._hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    # def _hierarchy_pos(self, G, root,
    #                    width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
    #                    pos=None, parent=None):
    #     '''
    #     see hierarchy_pos docstring for most arguments

    #     pos: a dict saying where all nodes go if they have been assigned
    #     parent: parent of this branch. - only affects it if non-directed

    #     '''

    #     if pos is None:
    #         pos = {root: (xcenter, vert_loc)}
    #     else:
    #         pos[root] = (xcenter, vert_loc)

    #     children = list(G.neighbors(root))

    #     if not isinstance(G, nx.DiGraph) and parent is not None:
    #         children.remove(parent)

    #     if len(children) != 0:
    #         dx = width/len(children)
    #         nextx = xcenter - width/2 - dx/2
    #         for child in children:
    #             nextx += dx
    #             pos = self._hierarchy_pos(
    #                 G, child,
    #                 width=dx, vert_gap=vert_gap,
    #                 vert_loc=vert_loc-vert_gap, xcenter=nextx,
    #                 pos=pos, parent=root
    #             )

    #     return pos
