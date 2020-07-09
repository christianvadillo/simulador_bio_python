# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:35:26 2020

@author: 1052668570
"""
import os
import matplotlib.pyplot as plt
import time

from Biorrefineria import Biorrfineria
from Proceso import ProcesoDA, ProcesoMEC
from Variable import Variable


wd = os.getcwd()

main_folder = 'OOP_TEST'
simulation = "test1"  # Change this for every new run
path = wd + '\\' + main_folder + '\\' + simulation

if not os.path.exists(path):
    os.makedirs(path)

# =============================================================================
# Initialize instances
# =============================================================================


def main(dias=3600, noise=False, failures=False):
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

    bio = Biorrfineria('bio1')
    bio.set_time(dias)  # Set simulation days

    # Create Process and add variables
    da = ProcesoDA('da')
    da.set_input_vars([dil_da, agv_in_da, dqo_in])
    da.set_output_vars([biomasa, dqo_out, agv_out])

    mec = ProcesoMEC('mec')
    mec.set_input_vars([agv_in_mec, dil_mec, eapp])
    mec.set_output_vars([ace_out, xa, xm, xh, mox, imec, qh2])

    # Manually initialize the variables
    da.input_vars[0].initialize_var(bio.time, noise=noise, sd=0.08)
    da.input_vars[1].initialize_var(bio.time, noise=noise, sd=1)
    da.input_vars[2].initialize_var(bio.time, noise=noise, sd=1)
    mec.input_vars[0].initialize_var(bio.time, noise=False)
    mec.input_vars[1].initialize_var(bio.time, noise=noise, sd=0.08)
    mec.input_vars[2].initialize_var(bio.time, noise=noise, sd=0.04)
    # mec.input_vars[2].plot()

    # Initilize the output matrix for the simulation
    da.initialize_outputs(bio.time)
    mec.initialize_outputs(bio.time)

    # Add process to bio
    bio.add_proceso([da, mec])

    da_acc, mec_acc = bio.simulate(noise=noise, failures=failures,
                                   batch_size=1)

    return bio, da_acc, mec_acc

# bio, da_acc, mec_acc = main(dias=360, noise=True, failures=True)
for _ in range(5):
    start_time = time.time()
    bio, da_acc, mec_acc = main(dias=3600, noise=True, failures=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s min ---" % ((time.time() - start_time)/60))


# print(timeit.Timer(main).timeit(number=3))
# Done 106.4159195000002 seg


# plt.figure()
# plt.plot(da_acc, '.-', label='DA', c='red')
# plt.plot(mec_acc, '.-', label='MEC', c='blue')
# plt.legend()

# for var in bio.procesos['da'].variables:
#     var.plot()

# for var in bio.procesos['mec'].variables:
#     var.plot()

# df_da, df_mec = bio.save_data(path)
