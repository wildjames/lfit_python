import lfit
# from celerite.modeling import Model  # don't actually use, this but be inspired by it
import numpy as np

"""
Kind of like celerites model class (though dont get too bogged down)

But we expand it to encompass the idea of a model optionally having
parents and children. Set parameters from top down, get from bottom up
"""


class Model:
    def __init__(self, parameter_names, parent=None):
        self.parameter_names = parameter_names
        self.parent = parent

    def get_parameter_names(self):
        root = [] if self.parent is None else self.parent.get_parameter_names()
        return root + self.parameter_names

    def get_parameter(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return getattr(self.parent, name)

    def set_parameter(self, name, val):
        if hasattr(self, name):
            self.name = val
        else:
            self.parent.name = val

    def set_parameter_vector(self, vec):
        for name, val in zip(self.parameter_names, vec):
            self.set_parameter(name, val)


class LightCurve:
    def __init__(self, x, dx, y, e):
        self.x = x
        self.dx = dx
        self.y = y
        self.e = e


class Eclipse(Model):

    def __init__(self, lightcurve, band):
        super().__init__(['rdisc', 'bstilt'], band)
        self.lc = lightcurve


class Band(Model):
    def __init__(self, band_name):
        super().__init__(['wdflux_{}'.format(band_name)])


class LCModel(Model):
    def __init__(self, bands):
        assert all(isinstance(b, Band) for b in bands)
        self.bands = list(bands)

    def chisq(self, params):
        self.set_parameter_list(params)
        chisq = 0.0
        for band in self.bands:
            for eclipse in band.eclipses:
                chisq += eclipse.chisq()
        return chisq

    def get_parameter_vector(self):
        """
        Should return list of parameter values like this

        The for loop is needed because get_parameter_vector *should* be on
        abstract class and automatically goes through children
        """
        pars = self.parameters()
        for band in self.bands:
            pars.extend(band.get_parameter_vector())
            for eclipse in band.eclipses:
                pars.extend(eclipse.get_parameter_vector())