from os import getcwd

import configobj
import numpy as np
from bokeh.layouts import Spacer, column, gridplot, row
from bokeh.models import ColumnDataSource, Span, Band
from bokeh.models.widgets import Dropdown, Panel, Slider, Tabs, markups
from bokeh.models.widgets.buttons import Button, Toggle
from bokeh.plotting import curdoc, figure
from pandas import DataFrame

import george as g
from CVModel import construct_model

try:
    from lfit import CV
    print("Successfully imported CV class from lfit!")
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
    # Set the initial values of q, rwd, and dphi. These will be used to
    # caclulate the location of the GP changepoints. Setting to initially
    # unrealistically high values will ensure that the first time
    # calcChangepoints is called, the changepoints are calculated.
    _olddphi = 9e99
    _oldq = 9e99
    _oldrwd = 9e99

    # _dist_cp is initially set to whatever, it will be overwritten anyway.
    _dist_cp = 9e99

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
        #TODO:
        # - Parameter reporting table
        # - Physcial params corresponding to params?

        #####################################################
        ############### Information Gathering ###############
        #####################################################

        print("Gathering information about my initial conditions...")
        # Save these, just in case I need to use them again later.
        self.mcmc_fname  = mcmc_input

        print("Looking for the mcmc input '{}'".format(self.mcmc_fname))

        # Parse the mcmc_input file
        self.parse_mcmc_input()
        self.init_data_storage()

        # Create the model inspector tab
        self.create_model_inspector_tab()
        self.update_lc_model('value', '', '')

        ######################################################
        ############# Add the tabs to the figure #############
        ######################################################

        # Make a tabs object
        self.tabs = Tabs(tabs=[self.inspector_tab])#, tab2])
        # Add it
        self.doc = curdoc()
        self.doc.add_root(self.tabs)
        print("Added the tabs to the document!")
        self.doc.title = 'MCMC Chain Supervisor'

        # Is the file open? Check once a second until it is, then once we find it remove this callback.
        self.update_lc_model('value', '', '')

        print("Finished initialising the dashboard!")

    def create_model_inspector_tab(self):
        '''Put together the model inspector tab.
        This tab should have a plot with 18 sliders, one for each
        parameter relevant to this eclipse.

        When plotting a simple BS model, the complex sliders should
        just not do anything.

        The complex model should be toggleable.
        The eclipse should be switchable via a dropdown box.
        '''
        simple_parDesc = {
            'wdFlux': 'White Dwarf Flux',
            'dFlux': 'Disc Flux',
            'sFlux': 'BS Flux',
            'rsFlux': 'Donor Flux',
            'q': 'Mass Ratio',
            'dphi': 'WD Eclipse Width',
            'rdisc': 'Disc Radius',
            'ulimb': 'Limb Dark Coeff.',
            'rwd': 'White Dwarf Radius',
            'scale': 'BS Scale length',
            'az': 'BS Azimuth',
            'fis': 'BS Isotropic Fraction',
            'dexp': 'Disc Exponent',
            'phi0': 'Phase Offset',
        }
        complex_parDesc = {
            'exp1': 'BS Exponent 1',
            'exp2': 'BS Exponent 2',
            'tilt': 'BS Tilt',
            'yaw': 'BS Yaw',
        }
        GP_parDesc = {
            'ln_tau_gp': 'GP Timescale',
            'ln_ampin_gp': 'GP intra-ecl amplitude',
            'ln_ampout_gp': 'GP inter-ecl amplitude'
        }

        print("Creating the model tweaker tab...")
        # I need a myriad of parameter sliders. The ranges on these should be set by the priors.
        self.par_sliders = []
        for name, title in simple_parDesc.items():
            title = simple_parDesc[name]
            param = self.parDict[name]
            print("Slider: {}".format(title))
            print(" -> value, lower limit, upper limit: {}\n".format(param))

            slider = Slider(
                name  = name,
                title = title,
                start = param[1],
                end   = param[2],
                value_throttled = param[0],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                format='0.0000',
                callback_throttle=50,
                callback_policy='mouseup'
            )

            self.par_sliders.append(slider)

        self.par_sliders_complex = []
        for name, title in complex_parDesc.items():
            title = complex_parDesc[name]
            param = self.parDict[name]
            print("Slider: {}".format(title))
            print(" -> value, lower limit, upper limit: {}\n".format(param))

            slider = Slider(
                name  = name,
                title = title,
                start = param[1],
                end   = param[2],
                value_throttled = param[0],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                format='0.0000',
                callback_throttle=50,
                callback_policy='mouseup'
            )

            self.par_sliders_complex.append(slider)

        self.par_sliders_GP = []
        for name, title in GP_parDesc.items():
            title = GP_parDesc[name]
            param = self.parDict[name]
            print("Slider: {}".format(title))
            print(" -> value, lower limit, upper limit: {}\n".format(param))

            slider = Slider(
                name  = name,
                title = title,
                start = param[1],
                end   = param[2],
                value_throttled = param[0],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                format='0.0000',
                callback_throttle=50,
                callback_policy='mouseup'
            )

            self.par_sliders_GP.append(slider)

        # Add the callbacks:
        for slider in self.par_sliders:
            slider.on_change('value_throttled', self.update_lc_model)
        for slider in self.par_sliders_complex:
            slider.on_change('value_throttled', self.update_lc_model)
        for slider in self.par_sliders_GP:
            slider.on_change('value_throttled', self.update_lc_model)
        print("Made the sliders...")

        # Data file picker
        menu = self.menu
        self.lc_change_fname_button = Dropdown(label="Choose Data", button_type="success", menu=menu, width=500)
        self.lc_obs_fname = self.current_eclipse.lc.fname
        self.lc_change_fname_button.on_change('value', self.update_lc_obs)
        print("Made the data picker...")

        # Button to switch from the complex to simple BS model, and vice versa
        col = 'success' if self.complex else 'danger'
        self.complex_button = Toggle(
            label='Complex BS?', width=120,
            button_type=col, active=self.complex
        )
        self.complex_button.on_click(self.update_complex)
        print("Made the complex button...")

        # Button for the GP
        col = 'success' if self.GP else 'danger'
        self.GP_button = Toggle(
            label='Use GP?', width=120,
            button_type=col, active=self.GP
        )
        self.GP_button.on_click(self.update_GP)
        print("Made the GP button...")

        print("Creating the LC plot...", end='')
        # Initialise the figure
        fname = self.current_eclipse.lc.name
        band_name = self.current_eclipse.parent.label
        title_text = "{} --- Band: {}".format(fname, band_name)
        self.lc_plot = figure(title=title_text, plot_height=500, plot_width=1200,
            toolbar_location='above', y_axis_location="left", x_axis_location=None)

        # Plot the lightcurve data
        self.lc_plot.scatter(x='phase', y='flux', source=self.lc_obs, size=5, color='black')

        # also plot residuals
        self.lc_res_plot = figure(plot_height=250, plot_width=1200,
            toolbar_location=None, y_axis_location="left",
            x_range=self.lc_plot.x_range)#, y_range=self.lc_plot.y_range)

        # Plot the lightcurve data
        self.lc_res_plot.scatter(x='phase', y='res', source=self.lc_obs, size=5, color='red')
        self.lc_res_plot.renderers += [Span(location=0, dimension='width', line_color='green', line_width=1)]

        # Plot the GP over the residuals
        band = Band(
            base='phase', lower='GP_lo', upper='GP_up',
            source=self.lc_obs,
            level='underlay', fill_alpha=0.3, line_width=0,
            line_color='black', fill_color='black')
        self.lc_res_plot.add_layout(band)

        # Plot the model
        self.lc_plot.line(
            x='phase', y='calc',
            source=self.lc_obs,
            line_color='red'
        )
        self.lc_plot.line(
            x='phase', y='sec',
            source=self.lc_obs,
            alpha=0.5, line_color='brown'
        )
        self.lc_plot.line(
            x='phase', y='wd',
            source=self.lc_obs,
            alpha=0.5, line_color='blue'
        )
        self.lc_plot.line(
            x='phase', y='bspot',
            source=self.lc_obs,
            alpha=0.5, line_color='green'
        )
        self.lc_plot.line(
            x='phase', y='disc',
            source=self.lc_obs,
            alpha=0.5, line_color='magenta'
        )
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
        inspector_layout = row([
            column([
                row(
                    [self.lc_change_fname_button,
                    self.complex_button, self.GP_button,
                    self.lc_isvalid, self.write2input_button]),
            self.lc_plot, self.lc_res_plot,
            ]),
            column([
                self.like_label,
                gridplot(
                    self.par_sliders, ncols=2,
                    toolbar_options={'logo': None}),
                Spacer(width=200, height=15, sizing_mode='scale_width'),
                gridplot(
                    self.par_sliders_complex, ncols=2,
                    toolbar_options={'logo': None}),
                Spacer(width=200, height=15, sizing_mode='scale_width'),
                gridplot(
                    self.par_sliders_GP, ncols=2,
                    toolbar_options={'logo': None})
            ])
        ])

        self.inspector_tab = Panel(child=inspector_layout, title="Lightcurve Inspector")
        print("Constructed the Lightcurve Inspector tab!")

    def init_data_storage(self):
        print("Grabbing the observations...")
        # Grab the data from the file, to start with just use the first in the list
        self.lc_obs = {}
        self.lc_obs['phase'] = self.current_eclipse.lc.x
        self.lc_obs['flux']  = self.current_eclipse.lc.y
        self.lc_obs['err']  = self.current_eclipse.lc.ye

        self.lc_obs = DataFrame(self.lc_obs)
        self.lc_obs.dropna(inplace=True, axis='index', how='any')

        # Total model lightcurve
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

    def make_header(self):
        '''Update the text at the top of the first tab to reflect mcmc_input, and the user defined stuff.'''

        header  = "I'm working from the directory: <b>{}</b></br>".format(getcwd())
        header += 'This chain has <b>{:,d}</b> burn steps, and <b>{:,d}</b> product steps.</br>'.format(
            self.nBurn, self.nProd)
        header += " We're using <b>{:,d}</b> walkers,".format(self.nWalkers)

        if bool(int(self.mcmc_input_dict['usePT'])):
            header += ' with parallel tempering sampling <b>{:,d}</b> temperatures,'.format(
                int(self.mcmc_input_dict['ntemps']))

        header += ' and running on <b>{:,d}</b> cores.</br>'.format(int(self.mcmc_input_dict['nthread']))

        # self.reportChain_label.text = header

    def write2input(self):
        '''Get the slider values, and modify mcmc_input.dat to match them.'''
        print("I should write the slider value back to the file!")

        to_write = {}

        band = self.current_eclipse.parent
        core = band.parent

        for par, param in self.current_eclipse.ancestor_param_dict.items():
            if par in core.node_par_names:
                parname_label = "{}".format(par)
            if par in band.node_par_names:
                parname_label = "{}_{}".format(par, band.label)
            if par in self.current_eclipse.node_par_names:
                parname_label = "{}_{}".format(par, self.current_eclipse.label)

            # Collect the data
            value = param.currVal
            prior_type = param.prior.type
            p1 = param.prior.p1
            p2 = param.prior.p2
            isVar = param.isVar

            newline = "{:>10s} = {:>12.4f} {:>12} {:>12.4f} {:>12.4f} {:>12}\n".format(
                parname_label,
                value,
                prior_type,
                p1,
                p2,
                isVar
            )

            to_write[parname_label] = newline

        with open(self.mcmc_fname, 'r') as f:
            mcmc_file = f.readlines()

        for key, item in to_write.items():
            print("\npar: {}".format(key))
            print("new line:\n{}".format(item))

        with open("mcmc_input.dat", 'w') as f:
            for line in mcmc_file:

                if not line.startswith('#'):
                    splitted = line.strip().split(' ')

                    if len(splitted) > 0:
                        par = splitted[0]

                        # print("This line in the file starts with: '{}'".format(par))
                        if par in to_write.keys():
                            # print("Replacing the line with the line:")
                            # print(" --> {}".format(to_write[par]))
                            line = to_write[par]

                        if par.lower() == 'usegp':
                            line = "useGP = {}\n".format(int(self.GP_button.active))
                        if par.lower() == 'complex':
                            line = "complex = {}\n".format(int(self.complex))

                # print("Writing the line:\n{}".format(line))
                f.write(line)

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

        # Construct the model
        self.model = construct_model(self.mcmc_fname)

        # Gather the parameters we can use
        self.complex  = bool(int(self.mcmc_input_dict['complex']))
        self.nWalkers = int(self.mcmc_input_dict['nwalkers'])
        self.GP       = bool(int(self.mcmc_input_dict['useGP']))
        self.nBurn    = int(self.mcmc_input_dict['nburn'])
        self.nProd    = int(self.mcmc_input_dict['nprod'])

        try:
            self.necl = int(self.mcmc_input_dict['neclipses'])
        except:
            self.necl = len(self.model.search_node_type("Eclipse"))

        # Query thre model for the eclipses it has
        self.eclipses = list(self.model.search_node_type('Eclipse'))
        self.current_eclipse = self.eclipses[0]

        # Get the parDict set up
        self.update_par_dict()

        if self.complex:
            print("Using the simple BS model!")

        if self.GP:
            print("Using the GP!")

        self.menu = [(ecl.name, ecl.lc.fname) for ecl in self.model.search_node_type('Eclipse')]
        print("The menu looks like this:")
        for m in self.menu:
            print(m)

    def update_par_dict(self):
        '''Take the current eclipse, and update the parDict with its limits
        from the file.'''
        raw_params = self.current_eclipse.ancestor_param_dict

        # Default values for the complex BS, as simple mcmc_input files may not have them:
        parDict = {
            'exp1_0':    [ 1.00, 0.001,  5.0],
            'exp2_0':    [ 2.00, 0.001,  5.0],
            'tilt_0':    [45.00, 0.001,  180],
            'yaw_0':     [ 0.00, -90.0, 90.0],
            'ln_ampin_gp':  [-9.99, -25.0, -1.0],
            'ln_ampout_gp': [-9.99, -25.0, -1.0],
            'ln_tau_gp':    [-5.00, -20.0, -1.0]
        }

        for key, param in raw_params.items():
            currval = param.currVal
            lolim = param.prior.p1
            hilim = param.prior.p2
            parDict[key] = [currval, lolim, hilim]

        self.parDict = parDict

    def reset_sliders(self):
        '''Set the parameters to the initial guesses.'''

        print("resetting the sliders to new values.")
        print("Removing the callbacks...")
        all_sliders = self.par_sliders + self.par_sliders_complex + self.par_sliders_GP

        # Disable the callbacks
        for slider in all_sliders:
            slider.remove_on_change('value_throttled', self.update_lc_model)
        print("Removed all the callbacks.\nSetting the slider values...")

        for par_name, param in self.parDict.items():
            for slider in all_sliders:
                if slider.name == par_name:
                    slider.value_throttled = param[0]
                    slider.value = param[0]
                    slider.start = param[1]
                    slider.end   = param[2]

        # Re-enable the calbacks
        for slider in all_sliders:
            slider.on_change('value_throttled', self.update_lc_model)

        self.update_lc_model('value_throttled', '', '')

        print("Done resetting sliders")

    def update_like_header(self, gp=False):
        print("res: {} data, err: {} data".format(len(self.lc_obs.data['res']), len(self.lc_obs.data['err'])))
        chisq =  self.lc_obs.data['res'] / self.lc_obs.data['err']
        chisq = np.sum(chisq**2)

        print("Chisq = {}".format(chisq))
        print("Updating header")

        print("label text was before: {}".format(self.like_label.text))
        print("label text is now: {}".format(self.like_label.text))

    def update_GP(self, new):
        '''Update the colour of the GP button'''
        self.GP = self.GP_button.active
        self.GP_button.button_type = 'success' if self.GP else 'danger'

    def update_complex(self, new):
        '''Handler for toggling the complex button. This should just enable/disable the complex sliders '''

        print("Toggling the complex model...")
        print("The complex variable was {}".format('on' if self.complex else 'off'))
        self.complex = self.complex_button.active
        print("The complex variable is now {}".format('on' if self.complex else 'off'))

        if self.complex:
            print("Changing to the complex BS model")
            # Complex sliders update the model
            for slider in self.par_sliders_complex:
                slider.on_change('value_throttled', self.update_lc_model)
            print("Enabled the comlpex sliders")

            # Initialise a new CV object with the new BS model
            pars = [slider.value_throttled for slider in self.par_sliders]
            pars.extend([slider.value_throttled for slider in self.par_sliders_complex])
            self.cv = CV(pars)
            print("Re-initialised the CV model")

            self.complex_button.button_type = 'success'

        else:
            print("Changing to the simple BS model")
            # Change the complex sliders to do nothing
            for slider in self.par_sliders_complex:
                slider.remove_on_change('value_throttled', self.update_lc_model)
            print("Disabled the comlpex sliders")

            # Initialise a new CV object with the new BS model
            pars = [slider.value_throttled for slider in self.par_sliders]
            print("Re-initialised the CV model")

            self.complex_button.button_type = 'danger'

        self.update_lc_model('value', '', '')

    def update_lc_model(self, attr, old, new):
        '''Callback to recalculate and redraw the CV model'''
        print("\n\nCALLED UPDATE_LC_MODEL")
        print("I want to update {} with the slider values.".format(
            self.current_eclipse.name
        ))

        # Get the band this eclipse belongs to
        band = self.current_eclipse.parent

        # Get a list of the current model parameter values
        par_vals = self.model.dynasty_par_vals

        # I need to check the complex and GP sliders, as well as the simple
        # ones.
        eclipse_par_sliders = self.par_sliders + self.par_sliders_complex
        eclipse_par_sliders += self.par_sliders_GP

        for i, par_name in enumerate(self.model.dynasty_par_names):
            # print("Param {}: {}".format(i, par_name))
            if par_name.endswith(self.current_eclipse.label):
                for slider in eclipse_par_sliders:
                    if par_name.startswith(slider.name):
                        # print("Slider {} found, taking its value".format(slider.name))
                        par_vals[i] = slider.value_throttled

            if par_name.endswith(band.label):
                for slider in self.par_sliders:
                    if par_name.startswith(slider.name):
                        # print("Slider {} found, taking its value".format(slider.name))
                        par_vals[i] = slider.value_throttled

            if par_name.endswith(self.model.label):
                for slider in self.par_sliders:
                    if par_name.startswith(slider.name):
                        # print("Slider {} found, taking its value".format(slider.name))
                        par_vals[i] = slider.value_throttled

        print("I altered the the following parameter vector components:")
        old_pars = self.model.dynasty_par_vals
        for i, (old_par, new_par) in enumerate(zip(old_pars, par_vals)):
            if old_par != new_par:
                print("parameter {} --- Old value: {:.3f}  ---  New value: {:.3f}".format(
                    i, old_par, new_par
                ))
        self.model.dynasty_par_vals = par_vals

        # Pull out a copy of the observations
        new_obs = dict(self.lc_obs.data)

        # Calculate
        try:
            components = self.current_eclipse.calcComponents()
        except:
            print("Invalid model parameters!")
            self.model.dynasty_par_vals = old_pars
            return

        # Total model lightcurve
        new_obs['calc']  = components[0]
        new_obs['res']   = new_obs['flux'] - new_obs['calc']

        # Components
        new_obs['wd']    = components[1]
        new_obs['sec']   = components[2]
        new_obs['bspot'] = components[3]
        new_obs['disc']  = components[4]

        # Push back into lc_obs
        self.lc_obs.data = dict(new_obs)

        self.recalc_GP_model('')

    def update_lc_obs(self, attr, old, new):
        '''callback to redraw the observations for the lightcurve'''
        print("\n\nCALLED UPDATE_LC_OBS")

        print("\nRedrawing the observations")
        print("I want to take the menu item: {}".format(new))
        for ecl in self.eclipses:
            if ecl.lc.fname == new:
                print("Found the eclipse!")
                self.current_eclipse = ecl


        new_obs = {}
        phi = self.current_eclipse.lc.x
        flx = self.current_eclipse.lc.y
        err = self.current_eclipse.lc.ye

        new_obs['phase'] = phi
        new_obs['flux']  = flx
        new_obs['err']   = err

        # Total model lightcurve
        new_obs['calc']  = np.zeros_like(new_obs['phase'])
        new_obs['res']   = np.zeros_like(new_obs['phase'])
        # Components
        new_obs['sec']   = np.zeros_like(new_obs['phase'])
        new_obs['bspot'] = np.zeros_like(new_obs['phase'])
        new_obs['wd']    = np.zeros_like(new_obs['phase'])
        new_obs['disc']  = np.zeros_like(new_obs['phase'])
        # GP
        new_obs['GP_up'] = np.zeros_like(new_obs['phase'])
        new_obs['GP_lo'] = np.zeros_like(new_obs['phase'])

        self.lc_obs.data = DataFrame(new_obs)

        print("\nUpdate the parDict")
        self.update_par_dict()

        print("\nReset the sliders")
        self.reset_sliders()

        print("\nSet the plotting area title")
        fname = self.current_eclipse.lc.name
        band_name = self.current_eclipse.parent.label
        title_text = "{} --- Band: {}".format(fname, band_name)
        print("Trying to change the title of the plot")
        print("Old title: {}".format(self.lc_plot.title.text))
        self.lc_plot.title.text = title_text
        print("The title should now be {}".format(title_text))

        # self.update_like_header(gp=self.GP)

    def calcChangepoints(self):
        '''Caclulate the WD ingress and egresses, i.e. where we want to switch
        on or off the extra GP amplitude.

        Requires an eclipse object, since this is specific to a given phase
        range.
        '''

        # Also get object for dphi, q and rwd as this is required to determine
        # changepoints
        pardict = {}
        for slider in self.par_sliders:
            pardict[slider.name] = slider.value_throttled

        dphi = pardict['dphi']
        q    = pardict['q']
        rwd  = pardict['rwd']
        phi0 = pardict['phi0']

        # Have they changed significantly?
        # If not, dont bother recalculating dist_cp
        dphi_change = np.fabs(self._olddphi - dphi) / dphi
        q_change = np.fabs(self._oldq - q) / q
        rwd_change = np.fabs(self._oldrwd - rwd) / rwd

        # Check to see if our model parameters have changed enough to
        # significantly change the location of the changepoints.
        if (dphi_change > 1.2) or (q_change > 1.2) or (rwd_change > 1.2):
            # Calculate inclination
            inc = roche.findi(q, dphi)

            # Calculate wd contact phases 3 and 4
            phi3, phi4 = roche.wdphases(q, inc, rwd, ntheta=10)

            # Calculate length of wd egress
            dpwd = phi4 - phi3

            # Distance from changepoints to mideclipse
            dist_cp = (dphi+dpwd)/2.

            # save these values for speed
            self._dist_cp = dist_cp
            self._oldq = q
            self._olddphi = dphi
            self._oldrwd = rwd
        else:
            # Use the old values
            dist_cp = self._dist_cp

        # Find location of all changepoints
        phase = self.lc_obs.data['phase']
        min_ecl = int(np.floor(phase.min()))
        max_ecl = int(np.ceil(phase.max()))

        eclipses = [e for e in range(min_ecl, max_ecl+1)
                    if np.logical_and(e > phase.min(),
                                      e < 1+phase.max()
                                      )
                    ]

        changepoints = []
        for e in eclipses:
            # When did the last eclipse end?
            egress = (e-1) + dist_cp + phi0
            # When does this eclipse start?
            ingress = e - dist_cp + phi0
            changepoints.append([egress, ingress])

        return changepoints

    def create_GP(self):
        """Constructs a kernel, which is used to create Gaussian processes.

        Creates kernels for both inside and out of eclipse,
        works out the location of any changepoints present, constructs a single
        (mixed) kernel and uses this kernel to create GPs

        Requires an Eclipse object to create the GP for. """

        # Get objects for ln_ampin_gp, ln_ampout_gp, ln_tau_gp and find the exponential
        # of their current values
        pardict = {}
        for slider in self.par_sliders_GP:
            pardict[slider.name] = slider.value_throttled

        ln_ampin   = pardict['ln_ampin_gp']
        ln_ampout  = pardict['ln_ampout_gp']
        ln_tau     = pardict['ln_tau_gp']

        ampin_gp   = np.exp(ln_ampin)
        ampout_gp  = np.exp(ln_ampout)
        tau_gp     = np.exp(ln_tau)

        # Calculate kernels for both out of and in eclipse WD eclipse
        # Kernel inside of WD has smaller amplitude than that of outside
        # eclipse.

        # First, get the changepoints
        changepoints = self.calcChangepoints()

        # We need to make a fairly complex kernel.
        # Global flicker
        kernel = ampin_gp * g.kernels.Matern32Kernel(tau_gp)
        # inter-eclipse flicker
        for gap in changepoints:
            kernel += ampout_gp * g.kernels.Matern32Kernel(
                tau_gp,
                block=gap
            )

        # Use that kernel to make a GP object
        georgeGP = g.GP(kernel, solver=g.HODLRSolver)

        return georgeGP

    def recalc_GP_model(self, new):
        '''Update the GP model'''
        lc_obs = dict(self.lc_obs.data)

        phi = lc_obs['phase']
        err = lc_obs['err']
        res = lc_obs['res']

        # Create the GP
        gp = self.create_GP()

        # Compute the matrix
        gp.compute(phi, err)

        # Draw samples from the GP
        samples = gp.sample_conditional(res, phi, size=100)

        # Get the mean, mu, standard deviation, and
        mu = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        lc_obs['GP_up'] = mu + std
        lc_obs['GP_lo'] = mu - std

        self.lc_obs.data = lc_obs

    def junk(self, attr, old, new):
        '''Sometimes, you just don't want to do anything'''
        # print("Calling the junk pile")
        pass

if __name__ in '__main__':
    print("This script must be run within a bokeh server:")
    print("  bokeh serve --show watchParams.py")
    print("Stopping!")
else:
    mc_fname = 'mcmc_input.dat'

    watcher = Watcher(mcmc_input=mc_fname)
