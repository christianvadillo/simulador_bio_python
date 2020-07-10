# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:51:07 2020

@author: 1052668570
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.tables import (MIN_MAX_TABLE,
                          MIN_MAX_DEFINED,
                          table_names,
                          plot_labels)

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


def plot_var(TT, data, number, title, ylabel, dias=3600, valor=0, limits=False):
    if limits:
        min_defined = MIN_MAX_DEFINED["min"][table_names(number)]
        max_defined = MIN_MAX_DEFINED["max"][table_names(number)]
        y_min_limit = min_defined
        y_max_limit = max_defined
    
        plt.figure()
        plt.plot(TT, data)
        plt.xlabel('Tiempo (d)')
        plt.ylabel(ylabel)
        plt.ylim(y_min_limit-y_max_limit, y_max_limit*1.10)
        plt.hlines(max_defined, 0, dias, linestyles='-.', color='red', linewidth=1)
        plt.hlines(min_defined, 0, dias, linestyles='-.', color='red', linewidth=1)
        plt.title(title)
        # plt.grid()
        plt.show()
    else:
        plt.figure()
        plt.plot(TT, data)
        plt.xlabel('Tiempo (d)')
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.grid()
        plt.show()

    # plt.savefig(subdir + '\\fig_{}_{}.png'.format(number, valor), dpi=400)


def plot_all_outputs(data, TT, limits=False):
    size = len(data.sim_outputs)
    DIAS = len(TT) // 24
    # Peso molecular del Acetato
    PMAce = 60.052  # gr/mol
    if size == 3:
        plot_var(TT, data.dqo_in,
                 number=3,
                 title='DQO, digestor anaerobio: Entrada',
                 ylabel='$DQO (gL^{-1})$',
                 dias=DIAS, limits=limits)

        plot_var(TT, data.sim_outputs[0, :],
                 number=4,
                 title='Biomasa, digestor anaerobio: Salida',
                 ylabel='$X_{1}(gL^{-1})$',
                 dias=DIAS, limits=limits)
        
        plot_var(TT, data.sim_outputs[1, :],
                 number=5,
                 title='DQO, digestor anaerobio: Salida',
                 ylabel='$DQO (gL^{-1})$',
                 dias=DIAS, limits=limits)
        
        plot_var(TT, data.sim_outputs[2, :] * PMAce,
                 number=6,
                 title='Acetato, digestor anaerobio: Salida, (entrada MEC)',
                 ylabel='$Ace (gL^{-1})$',
                 dias=DIAS, limits=limits)
    else:
        
        plot_var(TT, data.sim_outputs[0, :],
                 number=7,
                 title='Acetato, MEC: Salida',
                 ylabel='$Ace (gL^{-1})$',
                 dias=DIAS, limits=limits)

        plot_var(TT, data.sim_outputs[1, :],
                 number=8,
                 title='Bacterias Anodofílicas, MEC: Salida',
                 ylabel='$X_{a} (mgL^{-1})$',  # mgL o gL? por la conversión de AGV a gL?
                 dias=DIAS, limits=limits)
        
        plot_var(TT, data.sim_outputs[2, :],
                 number=10,
                 title='Bacterias Anodofílicas, MEC: Salida',
                 ylabel='$X_{m} (mgL^{-1})$',  # mgL o gL? por la conversión de AGV a gL?
                 dias=DIAS, limits=limits)

        plot_var(TT, data.sim_outputs[3, :],
                 number=11,
                 title='Bacterias hidrogenotrópicas, MEC: Salida',
                 ylabel='$X_{h} (mgL^{-1})$',  # mgL o gL? por la conversión de AGV a gL?
                 dias=DIAS, limits=limits)

        plot_var(TT, data.sim_outputs[4, :],
                 number=12,
                 title='Mediador de oxidación, MEC: Salida',
                 ylabel='$M_{ox} (L^{-1})$',
                 dias=DIAS, limits=limits)

        plot_var(TT, data.sim_outputs[5, :],
                 number=13,
                 title='Corriente, MEC: Salida',
                 ylabel='$I_{MEC} (A)$',
                 dias=DIAS, limits=limits)

        plot_var(TT, data.qh2,
                 number=14,
                 title='Flujo de hidrógeno, MEC: Salida',
                 ylabel='$Q_{H_{2}} (Ld^{-1})$',
                 dias=DIAS, limits=limits)


def plot_hists(df, log=False, percentils=False):
    n_cols = df.shape[1] // 2
    plt.figure()
    if log:
        for i, var in enumerate(df.columns[:-1]):
            if percentils:
                ulimut = np.percentile(df[var], 99)
                dlimit = np.percentile(df[var], 1)
                data = df.loc[(df[var] > dlimit) & (df[var] < ulimut), var]
            else:
                data = df[var]
            plt.subplot(2,n_cols, i+1)
            sns.distplot(data, hist_kws={'log':True, 'alpha':1}, kde=False)
            plt.tight_layout()
    else:
        for i, var in enumerate(df.columns[:-1]):
            if percentils:
                ulimut = np.percentile(df[var], 99)
                dlimit = np.percentile(df[var], 1)
                data = df.loc[(df[var] > dlimit) & 
                                       (df[var] < ulimut), var]
            else:
                data = df[var] 
            plt.subplot(2, n_cols, i+1)
            sns.distplot(data, kde=False, hist_kws={'alpha':1})
            plt.tight_layout()
