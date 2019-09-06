import json
import sys
import time
from os import getcwd, path

import bokeh as bk
import configobj
import numpy as np
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import Band, ColumnDataSource, Span, Whisker
from bokeh.models.annotations import Title
from bokeh.models.widgets import (DataTable, Dropdown, Panel, Slider,
                                  TableColumn, Tabs, TextInput, inputs,
                                  markups, tables)
from bokeh.models.widgets.buttons import Button, Toggle
from bokeh.plotting import curdoc, figure
from bokeh.server.callbacks import NextTickCallback
from pandas import DataFrame, read_csv

import george as g

from CVModel import construct_model



try:
    from lfit import CV
    print("Successfully imported CV class from lfit!")
    import mcmc_utils as u
    print("Successfully imported mcmc_utils")
    from trm import roche
    print("Successfully imported trm.roche!")
except ImportError:
    raise ImportError("Failed to import model modules!")

def parseInput(file):
    """Splits input file up making it easier to read"""
    input_dict = configobj.ConfigObj(file)
    return input_dict

class Watcher():
    '''This class will initialise a bokeh page, with some useful lfit MCMC chain supervision tools.
        - Ability to plot a live chain file's evolution over time
        - Interactive lightcurve model, with input sliders or the ability to grab the last step's mean
    '''
    def __init__(self, mcmc_input, tail=5000, thin=0):
        '''
        In the following order:
            - Save the tail and thin parameters to the self.XX
            - Read in the mcmc_input file to self.parDict
            - Set up self.parNames to have all parameter names
            - Set the other misc. trackers in handling files
            - Initialise the data storage object
            - Set up the first tab, with the live chain tracking page
            - Now, regardless of if we're complex or not, generate all 18 of the parameter sliders
                - If we're not complex BS model, the last 4 will just not do anything.
            - Set up the second tab, with the parameter tweaking tool.
            - Start watching for the creation of the chain file
        '''

        # TODO:
        # - improve startup speed
        # - Add a 'plot all parameters' button to the parameter tracker, per eclipse
        # - Make the corner plot function threaded?
        # - Time how frequent steps are, and predict how long the chain will take to complete.

        #####################################################
        ############### Information Gathering ###############
        #####################################################

        print("Gathering information about my initial conditions...")
        # Save these, just in case I need to use them again later.
        self.mcmc_fname  = mcmc_input

        print("Looking for the mcmc input '{}'".format(self.mcmc_fname))

        # Parse the mcmc_input file
        self.parse_mcmc_input()

        ######################################################
        ############### Tab 2: Model inspector ###############
        ######################################################

        print("Creating the model tweaker tab...")
        # I need a myriad of parameter sliders. The ranges on these should be set by the priors.
        self.par_sliders = []
        for par, title in zip(self.parNames[:14], self.parDesc[:14]):
            param = self.parDict[par]
            slider = Slider(
                title = title,
                start = param[1],
                end   = param[2],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                format='0.0000'
            )

            self.par_sliders.append(slider)

        # Default values for the complex BS, as simple mcmc_input files may not have them:
        defaults = {
            'exp1_0':    [ 1.00, 0.001,  5.0],
            'exp2_0':    [ 2.00, 0.001,  5.0],
            'tilt_0':    [45.00, 0.001,  180],
            'yaw_0':     [ 0.00, -90.0, 90.0],
            'ampin_gp':  [-9.99, -25.0, -1.0],
            'ampout_gp': [-9.99, -25.0, -1.0],
            'tau_gp':    [-5.00, -20.0, -1.0]
        }
        self.par_sliders_complex = []
        for par, title in zip(self.parNames[14:18], self.parDesc[14:18]):
            try:
                param = self.parDict[par]
            except:
                param = defaults[par]
            slider = Slider(
                title = title,
                start = param[1],
                end   = param[2],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                format='0.0000'
            )

            self.par_sliders_complex.append(slider)

        self.par_sliders_GP = []
        for par, title in zip(['ampin_gp', 'ampout_gp', 'tau_gp'], self.parDesc[-3:]):
            try:
                param = self.parDict[par]
            except:
                param = defaults[par]
            slider = Slider(
                title = title,
                start = param[1],
                end   = param[2],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                format='0.0000'
            )
            self.par_sliders_GP.append(slider)


        # Add the callbacks:
        for slider in self.par_sliders:
            slider.on_change('value', self.update_lc_model)
        if self.complex:
            for slider in self.par_sliders_complex:
                slider.on_change('value', self.update_lc_model)
        print("Made the sliders...")

        # Data file picker
        self.lc_change_fname_button = Dropdown(label="Choose Data", button_type="success", menu=menu, width=500)
        self.lc_obs_fname = menu[0][0]
        self.lc_change_fname_button.on_change('value', self.update_lc_obs)
        print("Made the data picker...")

        # Button to switch from the complex to simple BS model, and vice versa
        if self.complex:
            col = 'success'
        else:
            col = 'danger'
        self.complex_button = Toggle(label='Complex BS?', width=120, button_type=col, active=self.complex)
        self.complex_button.on_click(self.update_complex)
        print("Made the complex button...")

        # Button to force GP update
        self.GP_button = Button(label='Update GP', width=120)
        self.GP_button.on_click(self.recalc_GP_model)

        print("Grabbing the observations...")
        # Grab the data from the file, to start with just use the first in the list
        self.lc_obs = read_csv(menu[0][1],
                sep=' ', comment='#',
                header=None,
                names=['phase', 'flux', 'err'],
                skipinitialspace=True)
        self.lc_obs.dropna(inplace=True, axis='index', how='any')

        # Total model lightcurve
        # TODO: This is slow, make the page with this empty at first, then populate the data in a callback afterwards
        self.lc_obs['calc']  = np.zeros_like(self.lc_obs['phase'])
        self.lc_obs['res']   = np.zeros_like(self.lc_obs['phase'])
        # Components
        self.lc_obs['sec']   = np.zeros_like(self.lc_obs['phase'])
        self.lc_obs['bspot'] = np.zeros_like(self.lc_obs['phase'])
        self.lc_obs['wd']    = np.zeros_like(self.lc_obs['phase'])
        self.lc_obs['disc']  = np.zeros_like(self.lc_obs['phase'])
        # GP
        self.lc_obs['GP_up'] = np.zeros_like(self.lc_obs['phase'])
        self.lc_obs['GP_lo'] = np.zeros_like(self.lc_obs['phase'])

        print("Read in the observation, with the shape {}".format(self.lc_obs.shape))

        # Whisker can only take the ColumnDataSource, not the pandas array
        self.lc_obs = ColumnDataSource(self.lc_obs)


        print("Creating the LC plot...", end='')
        # Initialise the figure
        title = menu[0][0]
        self.lc_plot = bk.plotting.figure(title=title, plot_height=500, plot_width=1200,
            toolbar_location='above', y_axis_location="left", x_axis_location=None)
        # Plot the lightcurve data
        self.lc_plot.scatter(x='phase', y='flux', source=self.lc_obs, size=5, color='black')

        # also plot residuals
        self.lc_res_plot = bk.plotting.figure(plot_height=250, plot_width=1200,
            toolbar_location=None, y_axis_location="left",
            x_range=self.lc_plot.x_range)#, y_range=self.lc_plot.y_range)
        # Plot the lightcurve data
        self.lc_res_plot.scatter(x='phase', y='res', source=self.lc_obs, size=5, color='red')
        self.lc_res_plot.renderers += [Span(location=0, dimension='width', line_color='green', line_width=1)]
        # Plot the GP over the residuals
        band = Band(base='phase', lower='GP_lo', upper='GP_up', source=self.lc_obs,
                    level='underlay', fill_alpha=0.3, line_width=0, line_color='black', fill_color='red')
        self.lc_res_plot.add_layout(band)


        # # Plot the error bars - Bokeh doesnt have a built in errorbar!?!
        # # The following function does NOT remove old errorbars when new ones are supplied!
        # # This is because they are plotted as annotations, NOT something readliy modifiable!
        # self.lc_plot.add_layout(
        #     Whisker(base='phase', upper='upper', lower='lower', source=self.lc_obs,
        #     upper_head=None, lower_head=None, line_color='black', )
        # )

        # Plot the model
        self.lc_plot.line(x='phase', y='calc',  source=self.lc_obs,            line_color='red')
        self.lc_plot.line(x='phase', y='sec',   source=self.lc_obs, alpha=0.5, line_color='brown')
        self.lc_plot.line(x='phase', y='wd',    source=self.lc_obs, alpha=0.5, line_color='blue')
        self.lc_plot.line(x='phase', y='bspot', source=self.lc_obs, alpha=0.5, line_color='green')
        self.lc_plot.line(x='phase', y='disc',  source=self.lc_obs, alpha=0.5, line_color='magenta')
        print(" Done")

        # I want a button that'll turn red when the parameters are invalid. When clicked, it will either return the
        # model back to the initial values, or, if a chain has been read in, set the model to the last step read by the
        # watcher.
        self.lc_isvalid = Button(label='Initial Parameters', width=150)
        self.lc_isvalid.on_click(self.reset_sliders)
        print("Made the valid parameters button")

        # Write the current slider values to mcmc_input.dat
        self.write2input_button = Button(label='Write current values', width=150)
        self.write2input_button.on_click(self.write2input)
        print("Made the write2input button")

        self.lnlike = None
        self.like_label = markups.Div(width=1000)

        # Arrange the tab layout
        tab2_layout = row([
            column([
                row([self.lc_change_fname_button, self.complex_button, self.GP_button, self.lc_isvalid, self.write2input_button]),
            self.lc_plot, self.lc_res_plot,
            ]),
            column([
                self.like_label,
                gridplot(self.par_sliders, ncols=2, toolbar_options={'logo': None}),
                gridplot(self.par_sliders_complex, ncols=2, toolbar_options={'logo': None}),
                gridplot(self.par_sliders_GP, ncols=2, toolbar_options={'logo': None})
            ])
        ])

        tab2 = Panel(child=tab2_layout, title="Lightcurve Inspector")
        print("Constructed the Lightcurve Inspector tab!")

        # ######################################################
        # ################# Tab 3: Param Table #################
        # ######################################################

        # self.tableColumns = ['wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'rdisc',
        #     'ulimb', 'scale', 'az', 'fis', 'dexp', 'phi0']

        # # Extra parameters for the complex model
        # if self.complex:
        #     self.tableColumns.extend(['exp1', 'exp2', 'tilt', 'yaw'])

        # tableDict = {}
        # for p in self.tableColumns:
        #     get = p + "_{}"
        #     tableDict[p] = [self.parDict[get.format(i)][0] for i in range(self.necl)]
        # # This is opaque, but just trust me. Couldn't be bothered to be clearer.
        # tableDict['file'] = [self.mcmc_input_dict['file_{}'.format(i)].split('/')[-1] for i in range(self.necl)]

        # self.lastStep_CDS = ColumnDataSource(tableDict)
        # columns = [
        #     TableColumn(field=par, title=par, formatter=tables.NumberFormatter(format='0.0000')) for par in self.tableColumns
        # ]
        # columns.insert(0,
        #     TableColumn(field='file', title='File', width=1200)
        #     )
        # self.parameter_table = DataTable(source=self.lastStep_CDS, columns=columns, width=1600)

        # tab3_layout = column([self.parameter_table])
        # tab3 = Panel(child=tab3_layout, title="Parameter table")


        ######################################################
        ############# Add the tabs to the figure #############
        ######################################################

        # Make a tabs object
        self.tabs = Tabs(tabs=[self.tab1, tab2])
        # Add it
        self.doc = curdoc()
        self.doc.add_root(self.tabs)
        print("Added the tabs to the document!")
        self.doc.title = 'MCMC Chain Supervisor'

        ######################################################
        ## Setup for, and begin watching for the chain file ##
        ######################################################

        print("Setting up the chain file watcher...")
        # Keep track of how many steps we've skipped so far
        self.thinstep = 0

        # Initial values
        self.s    = 0                                       # Number of steps read in so far
        self.f    = False                                   # File object, initially false so we can wait for it to be created

        # Initialise data storage
        paramFollowSource = ColumnDataSource(dict(step=[]))
        self.paramFollowSource = paramFollowSource

        # Lists of what parameters we want to plot
        self.pars   = []     # List of params
        self.labels = []     # The labels, in the same order as pars

        # Is the file open? Check once a second until it is, then once we find it remove this callback.
        self.check_file = self.doc.add_next_tick_callback(self.open_file)
        self.doc.add_next_tick_callback(self.recalc_lc_model)

        print("Finished initialising the dashboard!")

    def make_header(self):
        '''Update the text at the top of the first tab to reflect mcmc_input, and the user defined stuff.'''

        header  = "I'm working from the directory: <b>{}</b></br>".format(getcwd())
        header +=  'This chain has <b>{:,d}</b> burn steps, and <b>{:,d}</b> product steps.</br>'.format(
            self.nBurn, self.nProd)
        header += " We're using <b>{:,d}</b> walkers,".format(self.nWalkers)

        if bool(int(self.mcmc_input_dict['usePT'])):
            header += ' with parallel tempering sampling <b>{:,d}</b> temperatures,'.format(
                int(self.mcmc_input_dict['ntemps']))

        header += ' and running on <b>{:,d}</b> cores.</br>'.format(int(self.mcmc_input_dict['nthread']))

        # self.reportChain_label.text = header

    def write2input(self):
        '''Get the slider values, and modify mcmc_input.dat to match them.'''

        # Figure out which eclipse we're looking at
        fname = self.lc_obs_fname
        template = 'file_{}'
        for i in range(self.necl):
            this = self.mcmc_input_dict[template.format(i)]
            if fname in this:
                break
            else: i = None
        if i is None:
            print("Couldn't find the index of that file!")
            return
        fileNumber = int(i)
        print('This is file {}'.format(fileNumber))

        # Get a list of the parameters we're going to change
        values = [slider.value for slider in self.par_sliders]

        labels = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        if self.complex:
            labels.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            values.extend([slider.value for slider in self.par_sliders_complex])

        labels = [l.replace('_0', '_{}'.format(fileNumber)) for l in labels]

        newvalues = {}
        for par, value in zip(labels, values):
            newvalues[par] = value

        # Make a copy of the mcmc_input file and edit that.
        mcmc_file = []
        with open(self.mcmc_fname, 'r') as f:
            for line in f:
                line_components = line.strip().split()
                if len(line_components) > 0:
                    par = line_components[0]
                    if par in labels:
                        value = newvalues[par]

                        newline = line_components.copy()
                        newline[2] = value
                        newline = "{:>10s} = {:>12.4f} {:>12} {:>12.4f} {:>12.4f} {:>12}\n".format(
                            str(newline[0]),
                            float(newline[2]),
                            str(newline[3]),
                            float(newline[4]),
                            float(newline[5]),
                            int(newline[6])
                            )

                        line = newline
                mcmc_file.append(line)

        # Overwrite the old mcmc_input file.
        print('Writing new file, {}'.format(self.mcmc_fname))
        with open(self.mcmc_fname, 'w') as f:
            for line in mcmc_file:
                f.write(line)
        self.parse_mcmc_input()

    def parse_mcmc_input(self):
        '''Parse the mcmc input dict, and store the following:
            - self.complex: bool
                Is the model using the simple or complex BS
            - self.GP: bool
                Is the model using the gaussian process?
            - self.nWalkers: int
                How many walkers are expected to be in the chain?
            - self.necl: int
                How many eclipses are we using?
            - self.parDict: dict
                Storage for the variables, including priors and initial guesses.
            - self.nBurn: int
                The number of burn-in steps.
            - self.nProd: int
                The number of product steps.
        '''
        print("Parsing the mcmc_input file, '{}'...".format(self.mcmc_fname))
        self.mcmc_input_dict = parseInput(self.mcmc_fname)

        # Gather the parameters we can use
        self.complex  = bool(int(self.mcmc_input_dict['complex']))
        self.nWalkers = int(self.mcmc_input_dict['nwalkers'])
        self.necl     = int(self.mcmc_input_dict['neclipses'])
        self.GP       = bool(int(self.mcmc_input_dict['useGP']))
        self.nBurn    = int(self.mcmc_input_dict['nburn'])
        self.nProd    = int(self.mcmc_input_dict['nprod'])

        # Parameter keys
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        if self.complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])
            print("Using the complex BS model!")
        else:
            print("Using the simple BS model!")

        if self.GP:
            print("Using the GP!")
            parNames.extend(['ampin_gp', 'ampout_gp', 'tau_gp'])

        for i in range(1, self.necl):
            parNames.extend([template.format(i) for template in parNameTemplate])

        ##### Here, read the parameters all into a dict of {key: [value, lowerlim, upperlim]} #####
        self.parDict = {}
        for param in parNames:
            line = self.mcmc_input_dict[param].strip().split(' ')
            line = [x for x in line if x != '']

            if 'gauss' in line:
                prior_min = float(line[2]) - (5*float(line[3]))
                prior_max = float(line[2]) + (5*float(line[3]))
                line[2] = prior_min
                line[3] = prior_max

            line = [float(line[0]), float(line[2]), float(line[3]), bool(int(line[-1]))]
            print("Par: {:>15s}: Val: {:>6.3f}, prior range: {:>6.3f} - {:<6.3f}, fit?: {}".format(
                param, line[0], line[1], line[2], line[3]
            ))

            parameter = [float(x) for x in line]
            self.parDict[param] = list(parameter)

        self.parNames = parNames
        self.model = construct_model(self.mcmc_fname)

    def reset_sliders(self):
        '''Set the parameters to the initial guesses.'''
        print("Resetting the sliders!")

        # Figure out which eclipse we're looking at
        fname = self.lc_obs_fname
        template = 'file_{}'
        for i in range(self.necl):
            this = self.mcmc_input_dict[template.format(i)]
            if fname in this:
                break
            else: i = None
        if i is None:
            print("Couldn't find the index of that file!")
            return
        fileNumber = int(i)
        print('This is file {}'.format(fileNumber))

        parNames = self.parNames

        # If we have s > 0, that means we've read in some chain. Get the last step.
        if self.s > 0:
            params = [p[0] for p in self.selectList][1:-1]

            stepData = []
            for par in parNames:
                # print("par: ",par)

                try:
                    # grab the value from lastStep
                    index = params.index(par)
                    val = self.lastStep[index]
                    # print("I want to get the parameter {}, from index {} in lastStep".format(par, index))
                    # print("Got a value of {} for parameter {}".format(val, par))
                except ValueError:
                    # If the valus isn't in lastStep, take it from the parDict
                    # print("The parameter {} is not fitted. Taking from initial condition:".format(par))
                    val = self.parDict[par][0]
                    # print("Got a value of {} for parameter {}".format(val, par))

                stepData.append(val)

        else:
            print('Getting values from the parameter dict.')
            stepData = [self.parDict[key][0] for key in parNames]
            # for i, j in zip(parNames, stepData):
            #     print("{:>15s}: {}".format(i, j))

        # Set the values of the sliders to the right values
        for par, slider in zip(parNames[:15], self.par_sliders):
            get = par.replace('_0', '_{}'.format(fileNumber))
            index = parNames.index(get)
            param = stepData[index]

            print("Setting the slider for {} to {}".format(get, param))
            slider.remove_on_change('value', self.update_lc_model)
            slider.value = param
            slider.on_change('value', self.update_lc_model)

        # Set the complex values, if needs be
        try:
            if self.complex:
                print("Setting the complex sliders")
                complex_names = ['exp1_{}', 'exp2_{}', 'tilt_{}', 'yaw_{}']
                for get, slider in zip(complex_names, self.par_sliders_complex):
                    get = get.format(fileNumber)
                    index = parNames.index(get)
                    param = stepData[index]

                    print("Setting the slider for {} to {}".format(get, param))
                    slider.remove_on_change('value', self.update_lc_model)
                    slider.value = param
                    slider.on_change('value', self.update_lc_model)
        except ValueError:
            self.complex_button.active = False
            self.update_complex(False)

        print("Setting the GP sliders")
        defaults = {
            'ampin_gp':  [-9.99, -25.0, -1.0],
            'ampout_gp': [-9.99, -25.0, -1.0],
            'tau_gp':    [-5.00, -20.0, -1.0]
        }
        GP_names = ['ampin_gp', 'ampout_gp', 'tau_gp']
        for get, slider in zip(GP_names, self.par_sliders_GP):
                try:
                    index = parNames.index(get)
                    param = stepData[index]
                except:
                    param = defaults[get][0]

                print("Setting the slider for {} to {}".format(get, param))
                slider.value = param

        self.update_lc_model('value', '', '')
        self.lc_isvalid.button_type = 'default'

    def update_like_header(self, gp=False):
        print("res: {} data, err: {} data".format(len(self.lc_obs.data['res']), len(self.lc_obs.data['err'])))
        chisq =  self.lc_obs.data['res'] / self.lc_obs.data['err']
        chisq = np.sum(chisq**2)

        print("Chisq = {}".format(chisq))
        print("Updating header")

        if gp:
            self.lnlike = self.gp.lnlikelihood(self.lc_obs.data['res'])


        print("label text was before: {}".format(self.like_label.text))
        self.like_label.text = "<i>GP ln(likelihood): <b>{:.1f}</b>, Chi Squared: <b>{:.1f}</b></i>".format(self.lnlike, chisq)
        print("label text is now: {}".format(self.like_label.text))

    def update_complex(self, new):
        '''Handler for toggling the complex button. This should just enable/disable the complex sliders '''

        print("Toggling the complex model...")
        self.complex = self.complex_button.active
        print("The complex variable is now {}".format('on' if self.complex else 'off'))

        if self.complex:
            print("Changing to the complex BS model")
            # Complex sliders update the model
            for slider in self.par_sliders_complex:
                slider.on_change('value', self.update_lc_model)
            print("Enabled the comlpex sliders")

            # Initialise a new CV object with the new BS model
            pars = [slider.value for slider in self.par_sliders]
            pars.extend([slider.value for slider in self.par_sliders_complex])
            self.cv = CV(pars)
            print("Re-initialised the CV model")

            self.complex_button.button_type = 'success'

        else:
            print("Changing to the simple BS model")
            # Change the complex sliders to do nothing
            for slider in self.par_sliders_complex:
                slider.remove_on_change('value', self.update_lc_model)
            print("Disabled the comlpex sliders")

            # Initialise a new CV object with the new BS model
            pars = [slider.value for slider in self.par_sliders]
            print("Re-initialised the CV model")

            self.complex_button.button_type = 'danger'

        self.recalc_lc_model()

    def update_lc_model(self, attr, old, new):
        '''Callback to recalculate and redraw the CV model'''
        pass

    def update_lc_obs(self, attr, old, new):
        '''callback to redraw the observations for the lightcurve'''

        print("Redrawing the observations")
        # Re-read the observations
        fname = self.lc_change_fname_button.value
        fname = str(fname)
        self.lc_obs_fname = fname

        print("Plotting data from file {}".format(fname))
        new_obs = read_csv(fname,
                sep=' ', comment='#', header=None, names=['phase', 'flux', 'err']
                )
        # Remove any row with a nan in it
        new_obs.dropna(inplace=True, axis='index', how='any')
        # new_obs['upper'] = new_obs['flux'] + new_obs['err']
        # new_obs['lower'] = new_obs['flux'] - new_obs['err']

        # Figure out which eclipse we're looking at
        template = 'file_{}'
        for i in range(self.necl):
            if self.mcmc_input_dict[template.format(i)] == fname:
                break
        print('This is file {}'.format(i))

        # Set the sliders to the initial guesses for that file
        parNames = ['wdFlux_{}', 'dFlux_{}', 'sFlux_{}', 'rsFlux_{}', 'q', 'dphi', 'rdisc_{}', 'ulimb_{}',
            'rwd', 'scale_{}', 'az_{}', 'fis_{}', 'dexp_{}', 'phi0_{}']
        parNames = [x.format(i) for x in parNames]
        for par, slider in zip(parNames, self.par_sliders):
            slider.remove_on_change('value', self.update_lc_model)
            value = self.parDict[par][0]
            slider.value = value
            slider.on_change('value', self.update_lc_model)

        # Are we complex? If yes, set those too
        if self.complex:
            parNamesComplex = ['exp1_{}', 'exp2_{}', 'tilt_{}', 'yaw_{}']
            parNamesComplex = [x.format(i) for x in parNamesComplex]
            for par, slider in zip(parNamesComplex, self.par_sliders_complex):
                slider.remove_on_change('value', self.update_lc_model)
                value = self.parDict[par][0]
                slider.value = value
                slider.on_change('value', self.update_lc_model)

        # Regenerate the model lightcurve
        pars = [slider.value for slider in self.par_sliders]
        if self.complex:
            pars.extend([slider.value for slider in self.par_sliders_complex])


        self.cv = CV(pars)
        print("Re-initialised the CV model")

        # Total model lightcurve
        new_obs['calc'] = self.cv.calcFlux(pars, np.array(new_obs['phase']))
        new_obs['res']  = new_obs['flux'] - new_obs['calc']
        # Components
        new_obs['sec']   = self.cv.yrs
        new_obs['bspot'] = self.cv.ys
        new_obs['wd']    = self.cv.ywd
        new_obs['disc']  = self.cv.yd

        # GP
        new_obs['GP_up'] = np.zeros_like(new_obs['phase'])
        new_obs['GP_lo'] = np.zeros_like(new_obs['phase'])

        # Push that into the data frame
        self.lc_obs.data = dict(new_obs)
        self.recalc_GP_model()

        # Set the plotting area title
        fname = fname.split('/')[-1]
        print("Trying to change the title of the plot")
        print("Old title: {}".format(self.lc_plot.title.text))
        self.lc_plot.title.text = fname
        print("The title should now be {}!!".format(fname))

        self.update_like_header(gp=self.GP)

    def junk(self, attr, old, new):
        '''Sometimes, you just don't want to do anything'''
        print("Calling the junk pile")
        pass

if __name__ in '__main__':
    print("This script must be run within a bokeh server:")
    print("  bokeh serve --show watchParams.py")
    print("Stopping!")
else:
    mc_fname = 'mcmc_input.dat'

    watcher = Watcher(mcmc_input=mc_fname)
