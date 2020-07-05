# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:35:26 2020

@author: 1052668570
"""
from numba import jit
import timeit
import os
from Biorrefineria import Biorrfineria
from Proceso import ProcesoDA, ProcesoMEC
from utils.utils_func import plot_all_outputs, plot_hists
import matplotlib.pyplot as plt
import numpy as np

wd = os.getcwd()

main_folder = 'OOP_TEST'
simulation = "test1" #Change this for every new run
path = wd + '\\' + main_folder + '\\' + simulation

if not os.path.exists(path):
    os.makedirs(path)


# =============================================================================
# Initialize instances
# =============================================================================
def main():
    bio = Biorrfineria('bio1')
    bio.set_time(300)  # Set simulation days
    
    # Setting up the inputs using random noise or change it to constant values
    dil1 = np.random.randn(bio.lt)*0.1 + 0.75  # 0.6
    agv_in = np.random.randn(bio.lt)*1 + 90  # 100
    dil2 = np.random.randn(bio.lt)*0.1 + 1.5  # 1.5
    eapp = np.random.randn(bio.lt)*0.001 + 0.6  # 0.5
    
    
    # Creating Process instances
    da = ProcesoDA('da', dil=dil1, agv_in=agv_in,
                   biomasa=24.8,
                   dqo_out=12.5,
                   agv_out=42.)
    
    da.set_input_vars(['dil', 'agv_in', 'dqo_in'])
    da.set_output_vars(['biomasa', 'dqo_out', 'agv_out'])
    da.initialize_dqo(bio.time)  # Initialize the senoidal array
    da.initialize_outputs(bio.time)  # Initilize the output matrix
    
    
    mec = ProcesoMEC('mec', dil=dil2, eapp=eapp, agv_in=np.zeros(bio.lt),
                     ace_out=2000,
                     xa=1,
                     xm=50,
                     xh=10,
                     mox=100,
                     imec=0,
                     qh2=np.zeros(bio.lt))
    
    mec.set_input_vars(['agv_in', 'dil','eapp'])
    mec.set_output_vars(['ace_out', 'xa', 'xm', 'xh', 'mox', 'imec', 'qh2'])
    mec.initialize_outputs(bio.time)  # Initilize the output matrix
    
    
    # Add process to bio
    bio.add_proceso([da, mec])
    da_acc, mec_acc = bio.simulate(noise=True, batch_size=1)  # noise=True - randomly add big noises to inputs
    
    # plt.figure()
    # plt.plot(da_acc, label='DA')
    # plt.plot(mec_acc, '.', label='MEC')
    # plt.legend()
    
    # plot_all_outputs(bio.procesos['da'], bio.time, limits=False)
    # plot_all_outputs(bio.procesos['mec'], bio.time)
    # df_da, df_mec = bio.save_data(path)
    # plot_hists(df_da, log=True, percentils=True)
    # plot_hists(df_mec, log=True, percentils=True)
    
print(timeit.Timer(main).timeit(number=3))
# Done 106.4159195000002 seg

