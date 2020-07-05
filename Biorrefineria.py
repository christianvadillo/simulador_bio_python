# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:28:58 2020

@author: 1052668570
"""
from numba import jit
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
from scipy.integrate import odeint
from utils.rf_classificador import clasificar_estado


class Biorrfineria:
    def __init__(self, name):
        self.__name = name
        self.procesos = dict()
        self.time = np.array([])
        self.lt = 0

    def __str__(self):
        return f"{self.__name}"

    def add_proceso(self, proceso: list()):
        try:
            for p in proceso:
                self.procesos[p.name] = p
        except Exception as e:
            print("Expected [Proceso,], got {proceso}", e)

    def int_ode15s(self, T, function, cond_ini):
        # The ``driver`` that will integrate the ODE(s)
        t0, t1 = T              # start and end
        t = np.linspace(t0, t1, 11)  # the points of evaluation of solution
        y0 = cond_ini               # initial values
        y = np.zeros((len(t), len(y0)))   # array for solution
        y[0, :] = y0
        sol = odeint(func=function, y0=y0, t=t)
        return (t, sol)

    def set_time(self, dias: int):
        # # Definicion de tiempo
        # 24 samples for day
        self.time = np.arange(0, dias, .041666)
        self.lt = len(self.time)

    def save_data(self, path=''):
        dfs = []
        for p in self.procesos.values():
             dfs.append(p.save_data(path + f'\df_{p.name}_{self.lt//24}.csv'))
        return dfs
    
    def simulate(self, batch_size=1, noise=False):
        if batch_size == 1:
            batch_size = self.lt-1
        start = 0
        end = 0
        n_batch = (self.lt-1) // batch_size
        da_acc = np.zeros(n_batch)
        mec_acc = np.zeros(n_batch)
        for batch in range(n_batch):
            start = batch_size * batch
            end =  batch_size * (batch+1)
            n_correct_da = n_correct_mec = n_total = 0
            for i in range(start, end):
                if noise:
                    ## Add random noise to da
                    draw = np.random.choice(range(2), 1, p=[0.999, 0.001])
                    if draw == 1:
                        self.procesos['da'].add_random_noise()
                # DA
                t1out, x1out = self.int_ode15s(self.time[i+0:i+2:1],
                                                self.procesos['da'].ode, 
                                                self.procesos['da'].sim_outputs[:, i])
                
                self.procesos['da'].i += 1  # To update dqo_in position for the next iteration
                self.procesos['da'].sim_outputs[:, i+1] = x1out[-1:, ]  # Save the integration results
                self.procesos['da'].update_outputs_variables()  # update output values from the last sim
            
                # MEC
                # If not previous noise injected
                if  self.procesos['mec'].agv_in[i] == 0:
                    self.procesos['mec'].agv_in[i] = self.procesos['da'].agv_out * self.procesos['da'].PMAce
            
                if noise:
                    ## Add random noise to MEC
                    draw = np.random.choice(range(2), 1, p=[0.999, 0.001])
                    if draw == 1:
                        self.procesos['mec'].add_random_noise()
        
                t2out, x2out = self.int_ode15s(self.time[i+0:i+2:1],
                                                self.procesos['mec'].ode, 
                                                self.procesos['mec'].sim_outputs[:, i])
                
                self.procesos['mec'].sim_outputs[:, i+1] = x2out[-1:, ]  # Save the integration results
                self.procesos['mec'].update_outputs_variables()  # update output values from the last sim
                self.procesos['mec'].update_qh2()
                self.procesos['mec'].i += 1  # To update qh2 position for the next iteration

                            
            df_da = self.procesos['da'].get_batch_data(start, end)
            df_mec = self.procesos['mec'].get_batch_data(start, end)
            y_da = df_da.iloc[:, -1]
            y_mec = df_mec.iloc[:, -1]
            predict = clasificar_estado(df_da.iloc[:, :-1], df_mec.iloc[:, :-1])
            # print(predict)

            n_correct_da += (y_da.values == predict[0]).sum()
            n_correct_mec += (y_mec.values == predict[1]).sum()
            n_total += y_mec.shape[0]
            da_acc[batch] = n_correct_da/n_total
            mec_acc[batch] = n_correct_mec/n_total

            # Fixing last values for mec.qh2 and mec_agv_in
            print("Accuracy-da:",  da_acc[batch])
            print("Accuracy-mec:", mec_acc[batch])
        self.procesos['mec'].agv_in[-1] = self.procesos['mec'].agv_in[-2]
        self.procesos['mec'].qh2[-1] = self.procesos['mec'].qh2[-2]

        print('\nDone')
    
        return da_acc, mec_acc
