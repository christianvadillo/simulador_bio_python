# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:28:58 2020

@author: 1052668570
"""
import pandas as pd
import numpy as np
import datetime as dt
import json

from scipy.integrate import odeint
from utils import ontoParser as op
from utils.rf_classificador import clasificar_estado
from sklearn.metrics import confusion_matrix


class Biorrfineria:
    """ It model a biorrefinery with two process (DA, MEC).
        As features one Variable have:
          * name,
          * procesos: np.array of Proceso objects,
          * time : number of days to simulate (int),
          * lt : total number of timesteps given the days of simulation,
    """

    def __init__(self, name):
        self.__name = name
        self.procesos = dict()
        self.time = np.array([])
        self.lt = 0

    def __str__(self):
        return f"{self.__name}"

    def add_proceso(self, proceso: list()):
        """ Set the processes used in the simulation """
        try:
            for p in proceso:
                self.procesos[p.name] = p
        except Exception as e:
            print("Expected [Proceso,], got {proceso}", e)

    def set_time(self, dias: int):
        """ Define the time given the days of simulation.
              Arguments:
                dias : total days of simulation (int)
        """
        assert isinstance(dias, int), "dias must be a integer"

        # 24 samples for day
        self.time = np.arange(0, dias, .041666)
        self.lt = len(self.time)

    def save_data(self, path=''):
        """ Save the simulation data of each process.
              Arguments:
                path : path for save the csv files (str)
        """
        dfs = []
        time = dt.datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        for p in self.procesos.values():
            name = path + f'/df_{p.name}_{self.lt//24}_{time}.csv'
            dfs.append(p.save_data(name))
        return dfs

    @staticmethod
    def detect_failures(da, mec, start, end):
        """ To detect anomaly states on each process using a Random Forest
        model.
            Arguments:
                da  : ProcessDA object
                mec : ProcessMEC object
                start : initial position to detect errors
                end  : final position to detect errors

        """

        n_correct_da = n_correct_mec = n_total_da = n_total_mec = 0
        da_acc = mec_acc = 0
        da_tn = da_fp = da_fn = da_tp = 0
        mec_tn = mec_fp = mec_fn = mec_tp = 0

        # Get the chunk of data in the range [start, end]
        da_inputs, da_targets = da.get_batch_data(start, end)
        mec_inputs, mec_targets = mec.get_batch_data(start,  end)
        # Call Random Forest model
        predict = clasificar_estado(da_inputs, mec_inputs)

        # Update limits for onto vars
        if not predict[0].any():
            da.update_limits(da_inputs)
        if not predict[1].any():
            mec.update_limits(mec_inputs)


        # Calculate accuracy
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
        
        print('-'*20)
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
        print('-'*20)
        print()
        return da_acc, mec_acc

    @staticmethod
    def recommender(da, mec, start, end):
        """ To detect anomaly states on each process using an ontology model and
        a Pellet reasoner.
        It also extract new information about the process and variables
              Arguments:
                  start : initial position to detect errors
                  end  : final position to detect errors

        """
        # Get the chunk of data in the range [start, end]
        da_inputs, _ = da.get_batch_data(start, end)
        mec_inputs, _ = mec.get_batch_data(start, end)

        df = np.concatenate([da_inputs, mec_inputs], axis=1)

        for row in df:
            # Call the Pellet reasoner
            states, _ = op.reasoner(row)
            states = json.loads(states)
            print('-'*10)
            print(states)
            da_onto_pred = 1 if 1 in states[0]['estado_code'] else 0
            mec_onto_pred = 1 if 1 in states[2]['estado_code'] else 0
            print(da_onto_pred)
            print(mec_onto_pred)
            print('-'*10)

    @staticmethod
    def int_ode15s(T, function, cond_ini, *args):
        """  The ``driver`` that will integrate the ODE(s).
              Arguments:
                  T : (start, end) - current timesteps to process
                  function  : ode function
                  cond_ini : initial values to run the integration
                  args  : extra arguments to pass it into the ode function

        """
        try:
            t0, t1 = T
            t = np.linspace(t0, t1, 11)  # the points of evaluation of solution
            y0 = cond_ini
            y = np.zeros((len(t), len(y0)))   # array for solution
            y[0, :] = y0
            sol = odeint(func=function, y0=y0, t=t, args=args)
            return (t, sol)
        except Exception:
            print("No more values")
            return 0

    def simulate(self, noise=False, failures=False, ontology=False, batch_size=1):
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
            batch_size = self.lt-1
        start = 0
        end = 0
        n_batch = (self.lt-1) // batch_size
        # For detection accuracy
        da_acc = np.zeros(n_batch)
        mec_acc = np.zeros(n_batch)

        da = self.procesos['da']
        mec = self.procesos['mec']
        choice = np.random.choice

        for batch in range(n_batch):
            start = batch_size * batch
            end = batch_size * (batch+1)
            for i in range(start, end):
                if noise:
                    # # Add random noise to da
                    draw = choice(range(2), 1, p=[0.999, 0.001])
                    if draw == 1:
                        da.add_random_noise()

                ############################################################
                # Integrate solver: DA
                ############################################################
                dil = da.input_vars[0].vals[i]
                agv_in = da.input_vars[1].vals[i]
                dqo_in = da.input_vars[2].vals[i]

                t1out, x1out = self.int_ode15s(
                    self.time[i+0:i+2:1],
                    da.ode,
                    da.sim_outputs[:, i],
                    dil,
                    agv_in,
                    dqo_in
                    )
                # To get correctly the batch_data inside processes
                da.i += 1
                # Save the integration results
                da.sim_outputs[:, i+1] = x1out[-1, :]

                ############################################################
                # Integrate solver: MEC
                ############################################################
                # If not previous noise injected
                if mec.input_vars[0].vals[i] == 0:
                    # pass da_agv_out*PMAce to mec_agv_in as input
                    # input_vars=[0:agv_in, 1:dil, 2:eapp]
                    # sim_outputs=[0:biomasa, 1:dqo_out, 2:agv_out].T
                    mec.input_vars[0].vals[i] =\
                        da.sim_outputs[2, i] * da.PMAce

                    if noise:
                        # # Add random noise to MEC
                        draw = choice(range(2), 1, p=[0.999, 0.001])
                        if draw == 1:
                            mec.add_random_noise()

                agv_in = mec.input_vars[0].vals[i]
                dil = mec.input_vars[1].vals[i]
                eapp = mec.input_vars[2].vals[i]

                t2out, x2out = self.int_ode15s(
                    self.time[i+0:i+2:1],
                    mec.ode,
                    mec.sim_outputs[:, i],
                    agv_in,
                    dil,
                    eapp
                    )
                # Save the integration results
                mec.sim_outputs[:, i+1] = x2out[-1, :]
                mec.update_qh2()
                # To get correctly the batch_data inside processes
                # and calculate qh2
                mec.i += 1

            if failures:
                da_acc[batch], mec_acc[batch] =\
                    self.detect_failures(da, mec, start, end)

            if ontology:
                self.recommender(da, mec, start, end)

        # Fixing last values for agv_in_da and qh2_mec
        da.update_outputs_variables()
        mec.update_outputs_variables()
        mec.input_vars[0].vals[-1] = mec.input_vars[0].vals[-2]
        mec.output_vars[-1].vals[-1] = mec.output_vars[-1].vals[-2]
        # print('\nDone')

        return da_acc, mec_acc
