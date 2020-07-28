# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:35:26 2020

@author: 1052668570
"""
import os
import matplotlib.pyplot as plt
import time
import numpy as np

from Biorrefineria import Biorrfineria
from Proceso import ProcesoDA, ProcesoMEC
from Variable import Variable

np.random.seed(101)
wd = os.getcwd()

main_folder = 'OOP_TEST'
simulation = "test1"  # Change this for every new run
path = wd + '\\' + main_folder + '\\' + simulation

if not os.path.exists(path):
    os.makedirs(path)

# =============================================================================
# Initialize instances
# =============================================================================


def main(dias=3600, noise=False, failures=False, ontology=False):
    # DA Variables definition
    dil_da = Variable(name='da_dil', vals=0.6, units="$d^{-1}$",
                      desc="Tasa de dilución: Entrada(DA)")
    agv_in_da = Variable(name='da_agv_in', vals=100.0, units="$mmol L^{-1}$",
                         desc="Concentración de acetato: Entrada(DA)")
    dqo_in = Variable(name='da_dqo_in', vals=25.0, units="$gd^{-1}}$",
                      desc="Demanda Quimica de Oxígeno: Entrada(DA)")
    biomasa = Variable(name='da_biomasa', vals=24.8, units="$X_{1}(gL^{-1})$",
                       desc="Biomasa: Salida(DA)")
    dqo_out = Variable(name='da_dqo_out', vals=12.5, units="$gL^{-1}}$",
                       desc="Demanda Quimica de Oxígeno: Salida(DA)")
    agv_out = Variable(name='da_agv_out', vals=42.0, units="$mmol L^{-1}$",
                       desc="Concentración de acetato: Salida(DA)")

    # MEC Variables definition
    agv_in_mec = Variable(name='mec_agv_in', vals=0.0, units="$gL^{-1}$",
                          desc="Concentración de acetato: Entrada(MEC)")
    dil_mec = Variable(name='mec_dil', vals=1.5, units="$d^{-1}$",
                       desc="Tasa de dilución: Entrada(MEC)")
    eapp = Variable(name='mec_eapp', vals=0.5, units="$V$",
                    desc="Voltaje: Entrada(MEC)")

    ace_out = Variable(name='mec_ace_out', vals=2000.0, units="$gL^{-1}$",
                       desc="Concentración de acetato: Salida(DA)")
    xa = Variable(name='mec_xa', vals=1.0, units="$X_{a} (mgL^{-1})$",
                  desc="Biomasa Anodofílicas: Salida(MEC)")
    xm = Variable(name='mec_xm', vals=50.0, units="$X_{m} (mgL^{-1})$",
                  desc="Biomasa Metanogénicas: Salida(MEC)")
    xh = Variable(name='mec_xh', vals=10.0, units="$X_{h} (mgL^{-1})$",
                  desc="Biomasa Hidrogenotropicas: Salida(MEC)")
    mox = Variable(name='mec_mox', vals=100.0, units="$L^{-1}$",
                   desc="Medidor de oxidación: Salida(MEC)")
    imec = Variable(name='mec_imec', vals=0.0, units="I_{MEC} (A)$",
                    desc="Corriente: Salida(MEC)")
    qh2 = Variable(name='mec_qh2', vals=0.0, units="Q_{H_{2}} ($Ld^{-1})$",
                   desc="Flujo de Hidrógeno: Salida(MEC)")

    # Create a biorefinery
    bio = Biorrfineria('bio1')
    bio.set_time(dias)  # Set simulation days

    # Create Process and add variables
    da = ProcesoDA('da')
    da.set_input_vars([dil_da, agv_in_da, dqo_in])
    da.set_output_vars([biomasa, dqo_out, agv_out])

    mec = ProcesoMEC('mec')
    mec.set_input_vars([agv_in_mec, dil_mec, eapp])
    mec.set_output_vars([ace_out, xa, xm, xh, mox, imec, qh2])

    # Manually initialize DA input variables
    da.input_vars[0].initialize_var(bio.time, noise=noise, sd=0.08)
    da.input_vars[1].initialize_var(bio.time, noise=noise, sd=1)
    da.input_vars[2].initialize_var(bio.time, noise=noise, sd=1)
    # Initialize outputs variables without noise
    da.output_vars[0].initialize_var(bio.time)
    da.output_vars[1].initialize_var(bio.time)
    da.output_vars[2].initialize_var(bio.time)

    # Manually initialize MEC input variables
    mec.input_vars[0].initialize_var(bio.time, noise=False)
    mec.input_vars[1].initialize_var(bio.time, noise=noise, sd=0.08)
    mec.input_vars[2].initialize_var(bio.time, noise=noise, sd=0.04)
    # Initialize outputs variables without noise
    mec.output_vars[0].initialize_var(bio.time)
    mec.output_vars[1].initialize_var(bio.time)
    mec.output_vars[2].initialize_var(bio.time)
    mec.output_vars[3].initialize_var(bio.time)
    mec.output_vars[4].initialize_var(bio.time)
    mec.output_vars[5].initialize_var(bio.time)
    mec.output_vars[6].initialize_var(bio.time)

    # mec.input_vars[2].plot()

    # Initilize the output matrix for the simulation
    da.initialize_outputs(bio.time)
    mec.initialize_outputs(bio.time)

    # Add processes to bio
    bio.add_proceso([da, mec])

    da_acc, mec_acc = bio.simulate(noise=noise,
                                   failures=failures,
                                   ontology=ontology,
                                   batch_size=24)

    return bio, da_acc, mec_acc


# bio, da_acc, mec_acc = main(dias=360, noise=True, failures=True)
times = np.zeros(1)
for i in range(1):
    start_time = time.time()
    bio, da_acc, mec_acc = main(dias=3600,
                                noise=False,
                                failures=False,
                                ontology=False)
    end_time = time.time()
    times[i] = end_time - start_time
    print("--- %s seconds ---" % (times[i]))
    print("--- %s min ---" % ((times[i])/60))

df_da, df_mec = bio.save_data(path)
# matlab_run = np.array([355.931666 ,
# 314.129656 ,
# 304.245377 ,
# 310.033032 ,
# 313.979148 ,
# 315.724433 ,
# 311.426489 ,
# 313.131862 ,
# 315.687116 ,
# 314.887149]
# )

# import pandas as pd
# times_df = pd.DataFrame(data=times, columns=['python_run_time_seg'])
# times_df['matlab_run_time_seg'] = matlab_run
# print(times_df.to_latex())

# da_values_avg_df = pd.DataFrame(data=np.zeros((3,3)),
#                                 columns=['variable', 'python_avg', 'matlab_avg'])

# da_values_avg_df['variable'] = ['biomasa', 'dqo_out', 'agv_out']
# da_values_avg_df['python_avg'] = bio.procesos['da'].sim_outputs.mean(axis=1)
# da_values_avg_df['matlab_avg'] = [59.9596, 14.2128, 140.9430]

# print(da_values_avg_df.to_latex())

# mec_values_avg_df = pd.DataFrame(data=np.zeros((7,3)),
#                                 columns=['variable', 'python_avg', 'matlab_avg'])

# mec_values_avg_df['variable'] = ['agv_out', 'xa', 'xm', 'xh', 'mox', 'imec', 'qh2']
# mec_values_avg_df['python_avg'] = bio.procesos['mec'].sim_outputs.mean(axis=1)
# mec_values_avg_df['matlab_avg'] = [5.6907e+03, 732.0475, 0.0266, 0.0114, 0.1013, 0.0462, 0.4554]

# print(mec_values_avg_df.to_latex())


# plt.figure()
# plt.plot(da_acc, '.-', label='DA', c='red')
# plt.plot(mec_acc, '.-', label='MEC', c='blue')
# plt.legend()

print(df_mec.tail(2).T)
for var in bio.procesos['da'].variables:
     var.plot(limits=False)

for var in bio.procesos['mec'].variables:
    var.plot()

# df_da, df_mec = bio.save_data(path)
# plt.figure(figsize=(12, 6))
# plt.scatter(range(len(df_da)), df_da['da_dqo_out'].values, s=1, alpha=0.3, 
#             c=df_da['labels'], cmap='viridis_r', marker='o')
# plt.grid()

# plt.figure(figsize=(12, 6))
# plt.scatter(range(len(df_mec)), df_mec['mec_qh2'].values, s=1, alpha=0.3, 
#             c=df_mec['labels'], cmap='viridis_r', marker='o')
# plt.grid()

bio.procesos['da'].sim_outputs[:, -2]
