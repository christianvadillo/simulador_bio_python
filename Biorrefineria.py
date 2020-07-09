# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:28:58 2020

@author: 1052668570
"""
import pandas as pd
import numpy as np
import datetime as dt
import json

# import matplotlib.pyplot as plt
from scipy.integrate import odeint
from utils import ontoParser as op
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

    def set_time(self, dias: int):
        # # Definicion de tiempo
        # 24 samples for day
        self.time = np.arange(0, dias, .041666)
        self.lt = len(self.time)

    def save_data(self, path=''):
        dfs = []
        time = dt.datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        for p in self.procesos.values():
            name = path + f'/df_{p.name}_{self.lt//24}_{time}.csv'
            dfs.append(p.save_data(name))
        return dfs

    def detect_failures(self, start, end):
        from sklearn.metrics import confusion_matrix
        n_correct_da = n_correct_mec = n_total_da = n_total_mec = 0
        da_acc = mec_acc = 0
        da_tn = da_fp = da_fn = da_tp = 0
        mec_tn = mec_fp = mec_fn = mec_tp = 0

        da_inputs, da_targets = self.procesos['da'].get_batch_data(start,
                                                                   end)
        mec_inputs, mec_targets = self.procesos['mec'].get_batch_data(start,
                                                                      end)
        predict = clasificar_estado(da_inputs, mec_inputs)

        n_correct_da += (da_targets == predict[0]).sum()
        n_correct_mec += (mec_targets == predict[1]).sum()
        n_total_da += da_targets.shape[0]
        n_total_mec += mec_targets.shape[0]
        da_acc = n_correct_da/n_total_da
        mec_acc = n_correct_mec/n_total_mec

        da_tn, da_fp, da_fn, da_tp = confusion_matrix(
            predict[0],
            da_targets,
            labels=[0, 1]
            ).ravel()
        mec_tn, mec_fp, mec_fn, mec_tp = confusion_matrix(
            predict[1],
            mec_targets,
            labels=[0, 1]
            ).ravel()

        print(f'Accuracy         | da: {da_acc:.3f},\
        mec: {mec_acc:.3f}')
        print(f'Normal state| da: {da_tn}({(da_tn/n_total_da)*100:.3f}%), \
        mec: {mec_tn}({(mec_tn/n_total_mec)*100:.3f}%) ')
        print(f'Failures detected| da: {da_tp}({(da_tp/n_total_da)*100:.3f}%),\
        mec: {mec_tp}({(mec_tp/n_total_mec)*100:.3f}%) ')
        print(f'Failures missed: | da: {da_fn}({(da_fn/n_total_da)*100:.3f}%),\
        mec: {mec_fn}({(mec_fn/n_total_mec)*100:.3f}%) ')
        print(f'False positives: | da: {da_fp}({(da_fp/n_total_da)*100:.3f}%),\
        mec: {mec_fp}({(mec_fp/n_total_mec)*100:.3f}%) ')
        print()
        return da_acc, mec_acc

    def recommender(self, start, end):
        da_inputs, da_targets = self.procesos['da'].get_batch_data(start,
                                                                   end)
        mec_inputs, mec_targets = self.procesos['mec'].get_batch_data(start,
                                                                      end)

        cols_da = [var.name for var in self.procesos['da'].variables]
        cols_mec = [var.name for var in self.procesos['mec'].variables]
        df_da = pd.DataFrame(data=da_inputs, columns=cols_da)
        df_mec = pd.DataFrame(data=mec_inputs, columns=cols_mec)
        fails_count_da = 0
        fails_count_mec = 0

        df = pd.concat([df_da, df_mec], axis=1)
        for row in df.iterrows():
            processes_state, onto = op.reasoner(row[1])
            processes_state = json.loads(processes_state)
            # print(processes_state)
            # :2 <- to isolate DA and MEC only
            fails_count_da += 1 if 'Falla' in processes_state[0]['estado'] else 0
            fails_count_mec += 1 if 'Falla' in processes_state[2]['estado'] else 0

            # print('da:', fails_count_da)
            # print(processes_state[0]['estado'])
            # print('mec:', fails_count_mec)
            # print(processes_state[2]['estado'])
        print('da anomalies:', fails_count_da)
        print('mec anomalies', fails_count_mec)

    def int_ode15s(self, T, function, cond_ini):
        # The ``driver`` that will integrate the ODE(s)
        try:
            t0, t1 = T              # start and end
            t = np.linspace(t0, t1, 11)  # the points of evaluation of solution
            y0 = cond_ini               # initial values
            y = np.zeros((len(t), len(y0)))   # array for solution
            y[0, :] = y0
            sol = odeint(func=function, y0=y0, t=t)
            # print(y0.shape)
            # print(sol.shape)
            return (t, sol)
        except Exception as e:
            print("No more values", e)
            return 0

    def simulate(self, noise=False, failures=False, batch_size=1):
        """ It simulate the coupling between DA and MEC processes in batches.
          If no batch_size is passed, by default will simulate run along all
          days defined.
          * If batch_size is passed, it will run the simulation in batches,
          useful for detect failures at certain time. For example:
          batch_size=24 will detect faiulres at each day of simulation.
          * If noise is passed, randomly it will generate noise.
          * If failures is passed, it will detect anomalies on each process
          at certain time defined by batch_size.
        """

        if batch_size == 1:
            print(batch_size)
            batch_size = self.lt-1
        start = 0
        end = 0
        n_batch = (self.lt-1) // batch_size
        # For detection accuracy
        da_acc = np.zeros(n_batch)
        mec_acc = np.zeros(n_batch)

        for batch in range(n_batch):
            start = batch_size * batch
            end = batch_size * (batch+1)
            for i in range(start, end):
                if noise:
                    # # Add random noise to da
                    draw = np.random.choice(range(2), 1, p=[0.999, 0.001])
                    if draw == 1:
                        self.procesos['da'].add_random_noise()
                # Integrate solver: DA
                t1out, x1out = self.int_ode15s(
                    self.time[i+0:i+2:1],
                    self.procesos['da'].ode,
                    self.procesos['da'].sim_outputs[:, i]
                    )
                # To update dqo_in position for the next iteration
                self.procesos['da'].i += 1
                # Save the integration results
                self.procesos['da'].sim_outputs[:, i+1] = x1out[-1, :]

                # Integrate solver: MEC
                # If not previous noise injected
                if self.procesos['mec'].input_vars[0].vals[i] == 0:
                    # pass da_agv_out*PMAce to mec_agv_in as input
                    # input_vars=[0:agv_in, 1:dil, 2:eapp]
                    # sim_outputs=[0:biomasa, 1:dqo_out, 2:agv_out].T
                    self.procesos['mec'].input_vars[0].vals[i] =\
                        self.procesos['da'].sim_outputs[2, i] *\
                        self.procesos['da'].PMAce

                    if noise:
                        # # Add random noise to MEC
                        draw = np.random.choice(range(2), 1,
                                                p=[0.999, 0.001])
                        if draw == 1:
                            self.procesos['mec'].add_random_noise()

                t2out, x2out = self.int_ode15s(
                    self.time[i+0:i+2:1],
                    self.procesos['mec'].ode,
                    self.procesos['mec'].sim_outputs[:, i]
                    )
                # Save the integration results
                self.procesos['mec'].sim_outputs[:, i+1] = x2out[-1, :]
                self.procesos['mec'].update_qh2()
                # To update qh2 position for the next iteration
                self.procesos['mec'].i += 1

            if failures:
                da_acc[batch], mec_acc[batch] = self.detect_failures(start,
                                                                     end)
                self.recommender(start, end)

        # Fixing last values for agv_in_da and qh2_mec
        self.procesos['da'].update_outputs_variables()
        self.procesos['mec'].update_outputs_variables()
        self.procesos['mec'].input_vars[0].vals[-1] =\
            self.procesos['mec'].input_vars[0].vals[-2]
        self.procesos['mec'].output_vars[-1].vals[-1] =\
            self.procesos['mec'].output_vars[-1].vals[-2]
        # print('\nDone')

        return da_acc, mec_acc

