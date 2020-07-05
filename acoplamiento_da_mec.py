# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:52:47 2020

@author: 1052668570

FALTA AGREGAR GRAFICAS DE MEC
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import integrate_ode as it

from Proceso import ProcesoDA, ProcesoMEC
from utils.utils_func import plot_all_outputs, plot_hists

plt.style.use('seaborn-whitegrid')

np.random.seed(101)


# =============================================================================
# For saving run
# =============================================================================
# Creación de directorios
wd = os.getcwd()

testing = 'OOP_TEST'
paremeter = "dil1_agv_senoidal_dil2_eapp_normal_dist_randoms_8" #Change this for every new run
subdir = wd + '\\' + testing + '\\' + paremeter
if not os.path.exists(subdir):
    os.makedirs(subdir)

# =============================================================================
# Initializing parameters
# =============================================================================
# Parameter to test
 #Change this for every new run


# # Definicion de tiempo
DIAS = 300
TT = np.arange(0, DIAS, .041666)
lt = len(TT)


# Peso molecular del Acetato
PMAce = 60.052  # gr/mol

# Setting up the inputs using random noise or change it to constant values
dil1 = np.random.randn(lt)*0.1 + 0.75  # 0.6
agv_in = np.random.randn(lt)*1 + 90  # 100
dil2 = np.random.randn(lt)*0.1 + 1.5  # 1.5
eapp = np.random.randn(lt)*0.001 + 0.6  # 0.5

# Constants
# dil1 = np.repeat(0.6, lt)
# agv_in = np.repeat(100.0, lt)
# dil2 = np.repeat(1.5, lt)
# eapp = np.repeat(0.5, lt)

# plt.plot(eapp, '.', alpha=0.4)
# plt.plot(dil2, '.', alpha=0.4)
# plt.plot(dil1, '.', alpha=0.4)
# plt.plot(agv_in, '.', alpha=0.4)


da = ProcesoDA('da', dil=dil1, agv_in=agv_in,
               biomasa=24.8,
               dqo_out=12.5,
               agv_out=42.)

da.set_input_vars(['dil', 'dqo_in', 'agv_in'])
da.set_output_vars(['biomasa', 'dqo_out', 'agv_out'])

# da.initialize_dqo(TT, noise=np.random.randn(lt)*0.1)  # Initialize the senoidal array
da.initialize_dqo(TT)  # Initialize the senoidal array

da.initialize_outputs(TT)  # Initilize the output matrix
# plt.plot(da.dqo_in)


mec = ProcesoMEC('mec', dil=dil2, eapp=eapp, agv_in=np.zeros(lt),
                 ace_out=2000,
                 xa=1,
                 xm=50,
                 xh=10,
                 mox=100,
                 imec=0,
                 qh2=np.zeros(lt))

mec.set_input_vars(['dil', 'agv_in', 'eapp'])
mec.set_output_vars(['ace_out', 'xa', 'xm', 'xh', 'mox', 'imec', 'qh2'])
mec.initialize_outputs(TT)  # Initilize the output matrix

# ========================================================================
# Main loop for Acoplamiento DA y MEC
""" 
%% Modelo de DA en la etapa de acidogénesis
%             _____________ Ace_mec   ___________
% Ace_in --> |             |-------->|           |--> QH2 
% DQO_in --> |  Digestor_A |    D -->|    MEC    |--> Imec
%    Dil --> |_____________|         |___________|
%
%
%1.- Dil != D            
%2.- Concentración de Ace en el digestor (g/L)
%3.- Concentración de Ace en la MEC (mg/L)
%4.- Solo se toman valores positivos de Imec (de acuerdo a la bibliografía)
%5.- Ace_in, DQO_in y Dil fueron tomados del reporte de tesis incluido en la
%    bibliografia (pag 59, fig.6.1).
%6.- Algunos parametros para el calculo de QH2 se ajustaron.
%7.- El modelo de la MEC en un reactor batch, presentado en el articulo adjunto a la
%    bibliografia, fue adaptado para operacion continua en esta simulacion.

"""
# ========================================================================
for i in range(0, lt-1):
    draw = np.random.choice(range(2), 1, p=[0.999, 0.001])
    if draw == 1:
        da.add_random_noise()

    # DA
    t1out, x1out = it.integrate_ode15s(TT[i+0:i+2:1],
                                       da.ode, da.sim_outputs[:, i])
    da.i += 1  # To update dqo_in position for the next iteration
    da.sim_outputs[:, i+1] = x1out[-1:, ]  # Save the integration results
    da.update_outputs_variables()  # update output values from the last sim

    # MEC
    # If not previous noise injected
    if  mec.agv_in[i] == 0:
        mec.agv_in[i] = (da.agv_out * PMAce)

    draw = np.random.choice(range(2), 1, p=[0.999, 0.001])
    if draw == 1:
        mec.add_random_noise()

    t2out, x2out = it.integrate_ode15s(TT[i+0:i+2:1],
                                       mec.ode, mec.sim_outputs[:, i])
    mec.sim_outputs[:, i+1] = x2out[-1:, ]  # Save the integration results
    mec.update_outputs_variables()  # update output values from the last sim
    mec.update_qh2()
    mec.i += 1  # To update qh2 position for the next iteration

    # break
    # if i % 1000 == 0:
    #     print(f'Iteración: {i}')

# Fixing last values for mec.qh2 and mec_agv_in
mec.agv_in[-1] = mec.agv_in[-2]
mec.qh2[-1] = mec.qh2[-2]


# Plotting
# plot_all_outputs(da, TT)
# plot_all_outputs(mec, TT)
# plot_hists(da_df, log=True, percentils=True)
# plot_hists(mec_df, log=True, percentils=False)


# Get and save data as dataframes
df_da = da.save_data(subdir + f"\df_da_{DIAS}.csv")
df_mec = mec.save_data(subdir + f"\df_mec_{DIAS}.csv")

# plt.figure()
# (df_da['agv_out']*PMAce).plot()
# plt.figure()
# df_mec['agv_in'].plot()

