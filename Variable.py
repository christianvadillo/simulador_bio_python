# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:43:41 2020

@author: 1052668570
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from itertools import count
from numbers import Number
from utils.tables import MIN_MAX_DEFINED

plt.style.use('grayscale')
mpl.rcParams['lines.linewidth'] = 0.7
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.edgecolor'] = 'grey'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'

# plt.ioff() # dont show plots
plt.ion()


class Variable:
    """ It model a variable for a process.
        As features, a Variable have:
          * name,
          * values [time dependent],
          * unit metric,
          * min value,
          * max value,
          * description """
    _ids = count(0)  # to count instances

    def __init__(self, name, vals, units='', min=0, max=0, desc=''):
        assert isinstance(name, str), "name must be a string"
        assert isinstance(vals, Number), "vals must be an array of numbers"
        assert isinstance(units, str), "units must be a string"
        assert isinstance(min, Number), "min must be a number"
        assert isinstance(max, Number), "max must be a number"
        assert isinstance(desc, str), "desc must be a string"

        self.id = next(self._ids)
        self.name = name
        self.vals = vals
        self.units = units
        self.min = min
        self.max = max
        self.desc = desc

    def initialize_var(self, time, noise=False, sd=0.001):
        """ Transfrom vals from numeric to a numpy array of size lt.
        If noise is set, it will add normal noise with mean vals
        and standard dev. sd.
        If noise is False, it will repeat vals lt times"""
        lt = len(time)
        if noise:
            if self.name == 'da_dqo_in':
                # Specific definition for dqo_in
                T = 1/360
                self.vals = np.array(
                    [20 * np.sin(2*np.pi*T*time[i]) + self.vals
                     for i in range(lt)] + abs(np.random.normal(loc=self.vals,
                                                                scale=sd,
                                                                size=lt))
                    )
            else:
                self.vals = abs(np.random.normal(loc=self.vals,
                                                 scale=sd,
                                                 size=lt))
        else:
            # Specific definition for dqo_in
            if self.name == 'da_dqo_in':
                T = 1/360
                self.vals = np.array(
                    [20 * np.sin(2*np.pi*T*time[i]) + 25
                     for i in range(lt)]
                     )
            else:
                self.vals = np.repeat(self.vals, lt)

            self.set_min()
            self.set_max()

    def set_min(self, min=None):
        """ It set the min value for the variable.
          If value is not passed, it will take the min from MIN_MAX_DEFINED
          table, if MIN_MAX_DEFINED does not exist, the min will set to 0 """
        if min:
            self.min = min
        else:
            try:
                self.min = MIN_MAX_DEFINED["min"][self.name]
            except Exception as e:
                print("Setting min to 0", e)
                self.min = 0

    def set_max(self, max=None):
        """ It set the max value for the variable.
          If value is not passed, it will take the max from MIN_MAX_DEFINED
          table, if MIN_MAX_DEFINED does not exist, the max will set to 0 """
        if max:
            self.max = max
        else:
            try:
                self.max = MIN_MAX_DEFINED["max"][self.name]
            except Exception as e:
                print("Setting max to 0", e)
                self.max = 0

    def plot(self):
        """ To plot the values of the variable """
        plt.figure(self.id)
        plt.plot(self.vals)
        plt.ylabel(self.units)
        plt.xlabel('Time (d)')
        plt.title(self.desc)
        plt.grid()
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return f'Variable: {self.name}'
