# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 08:43:59 2020

@author: 1052668570
"""
import numpy as np
import random
import pandas as pd
from numba import jit
from utils.tables import DA_MIN_MAX_DEFINED, MEC_MIN_MAX_DEFINED

class Proceso:
    def __init__(self, name='Proceso'):
        self.__name = name
        self.i = 0
        self.PMAce = 60.052  # gr/mol - Peso molecular del Acetato
        self.sim_outputs = np.array([])
        self.input_vars = []
        self.output_vars = []

    @property
    def name(self):
        return self.__name

    # def add_variable(self, variable):
    #     self.__variables.extend(variable)

    def ode():
        pass

    def initialize_outputs():
        pass

    def update_outputs_variables():
        pass
    
    def transform_negatives(self):
        for var in self.input_vars:
            var = getattr(self, var)
            try:
                var[np.where(var < 0)] = np.mean(var)
            except Exception as e:
                print("No se pudo completar la acción", e)
                print(f'var: {var}')
        for i in range(len(self.sim_outputs)):
            try:
                self.sim_outputs[np.where(self.sim_outputs[i, :] < 0)] = np.mean(self.sim_outputs[i, :])
            except Exception as e:
                print("No se pudo completar la acción", e)


    def __str__(self):
        return f"Proceso {self.__name}"
class ProcesoDA(Proceso):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dil = kwargs.get('dil', 0)
        self.dqo_in = kwargs.get('dqo_in', 0)
        self.agv_in = kwargs.get('agv_in', 0)
        self.biomasa = kwargs.get('biomasa', 0)
        self.dqo_out = kwargs.get('dqo_out', 0)
        self.agv_out = kwargs.get('agv_out', 0)


    def set_input_vars(self, vars_):
        self.input_vars.extend(vars_)

    def set_output_vars(self, vars_):
        self.output_vars.extend(vars_)

    def initialize_dqo(self, tt, noise=0):
        T = 1/360
        lt = len(tt)
        self.dqo_in = np.array([20 * np.sin(2*np.pi*T*tt[i]) + 25
                                for i in range(lt)]) + noise

    def initialize_outputs(self, tt):
        lt = len(tt)
        self.sim_outputs = np.zeros((3, lt))
        self.sim_outputs[:, 0] = np.array([self.biomasa, self.dqo_out, self.agv_out])

    def add_random_noise(self):
        amount_of_noise = 0
        num_of_vars = np.random.randint(0, len(self.input_vars))
        vars_ = random.sample(self.input_vars, k=num_of_vars)
        print(f'{self.name} - Iteration: {self.i}... Adding large noise to variable {vars_}')

        try:
            for var in vars_:
                if var == 'dil':
                    amount_of_noise = np.random.randint(1, 24)
                    noise = (np.random.randn(amount_of_noise)*40)
                    self.dil[self.i:self.i+amount_of_noise] += noise  # Adding the noise

                elif var == 'agv_in':
                    amount_of_noise = np.random.randint(1, 24)
                    noise = (np.random.randn(amount_of_noise)*30)
                    self.agv_in[self.i:self.i+amount_of_noise] += noise  # Adding the noise
                else:
                    amount_of_noise = np.random.randint(1, 24)
                    noise = (np.random.randn(amount_of_noise)*30)
                    self.dqo_in[self.i:self.i+amount_of_noise] += noise  # Adding the noise

            # Changing negative values to zero
            self.transform_negatives()
        except Exception as e:
            print("No se pudo completar la acción", e)

    def update_outputs_variables(self):
        # print(self.sim_outputs[:, self.i])
        self.biomasa = self.sim_outputs[0, self.i]
        self.dqo_out = self.sim_outputs[1, self.i]
        self.agv_out = self.sim_outputs[2, self.i]
        
    def get_batch_data(self, s, e):
        """ the predicter except 
        (dil, dqo_in, agv_in, biomasa, dqo_out, agv_out)
        """
        df = np.hstack((self.dil[s:e].reshape(-1,1),
                        self.agv_in[s:e].reshape(-1,1), 
                        self.dqo_in[s:e].reshape(-1,1),
                        self.sim_outputs.T[s:e, :]))

        df = pd.DataFrame(data=df,
                         columns=self.input_vars + self.output_vars)
        df.set_index(pd.date_range(end='2020-06-29',
                                      periods=len(df), 
                                      freq='H'), inplace=True)

        df['agv_out'] = df['agv_out'] * self.PMAce

        labels = np.zeros(len(df))
        rowError = []
        for i, row in enumerate(df.iterrows()): 
            for key, value in row[1].iteritems():
                if value < DA_MIN_MAX_DEFINED["min"][key]:
                    rowError.append(DA_MIN_MAX_DEFINED["ErrorMin"][key])
                if value > DA_MIN_MAX_DEFINED["max"][key]:
                    rowError.append(DA_MIN_MAX_DEFINED["ErrorMax"][key])
            labels[i] = 1 if rowError else 0
            rowError = []
            
        df['labels'] = labels
        
        return df

    def save_data(self, path):
        df = np.hstack((self.dil.reshape(-1,1),
                        self.agv_in.reshape(-1,1), 
                        self.dqo_in.reshape(-1,1),
                        self.sim_outputs.T))

        df = pd.DataFrame(data=df,
                         columns=self.input_vars + self.output_vars)
        df.set_index(pd.date_range(end='2020-06-29',
                                      periods=len(df), 
                                      freq='H'), inplace=True)

        df['agv_out'] = df['agv_out'] * self.PMAce

        labels = np.zeros(len(df))
        rowError = []
        for i, row in enumerate(df.iterrows()): 
            for key, value in row[1].iteritems():
                if value < DA_MIN_MAX_DEFINED["min"][key]:
                    rowError.append(DA_MIN_MAX_DEFINED["ErrorMin"][key])
                if value > DA_MIN_MAX_DEFINED["max"][key]:
                    rowError.append(DA_MIN_MAX_DEFINED["ErrorMax"][key])
            labels[i] = 1 if rowError else 0
            rowError = []
            
        df['labels'] = labels
        
        df.to_csv(path)
        print(f'data saved in {path}')

        return df

    def ode(self, x, t, *args):
        # print(f'inside ode-da: {x}')
        # print(f'inside ode-da: {t}')
        # Parámetros del modelo de la DA
        # kLaCO2 = 2000  # 1/d
        # KHCO2  = .0271 # M/bar
        # Tr = 308       # K
        # Vr = 4.4       # L
        # Vg = .1
        # R = 8.314e-2   # bar/KM
        u_max1 = 0.31  # d^-1
        k_s1 = 24.0     # gDQO/L
        alpha = 0.18
        k2k1 = 3.8      # mmAGV/gDQO
        # k3 = 611.1     # mmolAGV/gX2
        # Tamb = 298     # K
        # Patm = 1.013   # bar
        # pvH2O = .0557  # bar

        #  Tiempo de retención y Desorción
        # D = V/Q
        u1 = u_max1*(x[1]/(k_s1+x[1]))
        # PCO2g = x(6)*R*Tr
        # qC1 = kLaCO2*(x(5)-KHCO2*PCO2g)

        # Flujo gaseoso
        # Qg = ((R*Tamb*V)/(Patm-pvH2O))*u1 # L/d #Flujo de gas, descomentar

        # Modelo de la disgestión anaerobia
        # AGVin mmolL,

        # print(self.i)
        # print(len(self.dqo_in))

        dX1 = - alpha * self.dil[self.i] * x[0] + u1 * x[0]
        dS1 = self.dqo_in[self.i] * self.dil[self.i] - x[1] * self.dil[self.i] - u1 * x[0]
        dS2 = self.agv_in[self.i] * self.dil[self.i] - x[2] * self.dil[self.i] + k2k1 * u1 * x[0]
        # dZ1 = (Z1 - x(3))*D
        # dC1 = C1*D - x(4)*D  + k3*u1*x(0) - qC1*(V/Vg)
        # dC1g = -x(5)*(Qg/Vg)  + qC1*(V/Vg)
        # self.i += 1

        return [dX1, dS1, dS2]


class ProcesoMEC(Proceso):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dil = kwargs.get('dil', 0)
        self.eapp = kwargs.get('eapp', 0)
        self.agv_in = kwargs.get('agv_in', np.array([]))  # Feeded by da
        self.ace_out = kwargs.get('ace_out', 0)
        self.xa = kwargs.get('xa', 0)
        self.xh = kwargs.get('xh', 0)
        self.xm = kwargs.get('xm', 0)
        self.imec = kwargs.get('imec', 0)
        self.mox = kwargs.get('mox', 0)
        self.qh2 = kwargs.get('qh2', 0)


        # For QH2 calcuations
        self.H2 = 1
        self.Kh = 0.0001
        self.YH2 = 0.9
        self.P = 1
        self.m = 2
        self.R2 = 0.08205
        self.F = 96485
        self.F1 = self.F / (60 * 60 * 24)
        self.T = 298.15
        self.Yh = 0.05
        self.V = 1
        self.umaxh = 0.5
        self.muh = (self.umaxh * self.H2) / (self.Kh + self.H2)

        # Parametros del modelo para ode
        self.qmaxa = 13.14               # d^-1
        self.qmaxm = 14.12               # d^-1
        self.Ksa = 20                    # mg L^-1
        self.Ksm = 80                    # mg L^-1
        self.KM = .01                    # mg L^-1
        self.umaxa = 1.97                # d^-1
        self.Kda = .04                   # d^-1 --> modificado
        self.umaxm = .3                  # d^-1
        self.Kdm = .01                   # d^-1 --> modificado
        self.umaxh = .5                  # d^-1
        self.Kdh = .01                   # d^-1
        self.Kh = .001                   # mg L^-1
        self.H2 = 1                      # mg L^-1
        self.gamma = 663400              # mg-M/mol_med
        self.Vr = 1                      # L
        self.m = 2                       # mol-e/mol-H2
        self.F = 96485                   # C mol-e^-1
        self.F1 = self.F/(60*60*24)           # A d mol-e^-1
        # YM = 34.85                 # mg-M/mg-A --> modificado
        self.YM = 3.3
        self.Yh = 0.05                     # mLH2/mgX/d
        self.Xmax1 = 512.5               # mg L^-1
        self.Xmax2 = (1680+750)/2        # mg L^-1 
        self.Mtotal = 1000               # mgM mg x^-1
        self.beta = 0.5
        self.AsA = 0.01                  # m^2
        # io =.005                   # Am^-2
        self.io = 1
        self.E_CEF = -0.35               # V
        self.E_app = self.eapp               # Fuente de alimentación Default= .5 V == 500mv
        # Si E_app está en 0, existe una falla) Por encima de 10 V, igual hay falla
        # Rmin = 20
        self.Rmin = 2
        # Rmax = 2000
        self.Rmax = 200
        self.KR = .024
        self.R = 8.314                   # J mol^-1 K^-1
        self.R1 = 82.05                  # mL atm mol^-1 K^-1
        self.T = 298.15                  # K

    def set_input_vars(self, vars_):
        self.input_vars.extend(vars_)

    def set_output_vars(self, vars_):
        self.output_vars.extend(vars_)

    def initialize_outputs(self, tt):
        lt = len(tt)
        self.sim_outputs = np.zeros((6, lt))
        self.sim_outputs[:, 0] = np.array([self.ace_out,
                                           self.xa, self.xm, self.xh,
                                           self.mox, self.imec])

    def add_random_noise(self):
        amount_of_noise = 0
        num_of_vars = np.random.randint(0, len(self.input_vars))
        vars_ = random.sample(self.input_vars, k=num_of_vars)
        print(f'{self.name} - Iteration: {self.i}... Adding large noise to variable {vars_}')
        try: 
            for var in vars_:
                if var == 'dil':
                    amount_of_noise = np.random.randint(1, 24)
                    noise = (np.random.randn(amount_of_noise)*10)
                    self.dil[self.i:self.i+amount_of_noise] += noise  # Adding the noise
     
                elif var == 'agv_in':
                    amount_of_noise = np.random.randint(1, 24)
                    noise = (np.random.randn(amount_of_noise)*10)+5000
                    self.agv_in[self.i:self.i+amount_of_noise] += noise  # Adding the noise
                else:
                    amount_of_noise = np.random.randint(1, 24)
                    noise = (np.random.randn(amount_of_noise)*0.01)
                    self.eapp[self.i:self.i+amount_of_noise] += noise  # Adding the noise

            # Changing negative values to zero
            self.transform_negatives()

        except Exception as e:
            print("No se pudo completar la acción", e)

    def update_outputs_variables(self):
        # print(self.sim_outputs[:, self.i])
        self.ace_out = self.sim_outputs[0, self.i]
        self.xa = self.sim_outputs[1, self.i]
        self.xm = self.sim_outputs[2, self.i]
        self.xh = self.sim_outputs[3, self.i]
        self.mox = self.sim_outputs[4, self.i]
        self.imec = self.sim_outputs[5, self.i]

    def update_qh2(self):
        self.qh2[self.i] = self.YH2 * (
            (self.imec * self.R2 * self.T)/(self.m * self.F1 * self.P)
            ) - self.Yh * self.muh * self.V * self.xh / 1000

    def get_batch_data(self, s, e):
        """ the predicter except 
        (agv_in, dil, eapp, xa, xm, xh, mox, imec, qh2)
        """
        df = np.hstack((self.agv_in[s:e].reshape(-1,1),
                        self.dil[s:e].reshape(-1,1),
                        self.eapp[s:e].reshape(-1,1), 
                        self.sim_outputs.T[s:e],
                        self.qh2[s:e].reshape(-1,1)))

        df = pd.DataFrame(data=df,
                             columns=self.input_vars + self.output_vars)
        df.set_index(pd.date_range(end='2020-06-29',
                                      periods=len(df), 
                                      freq='H'), inplace=True)
        
        labels = np.zeros(len(df))
        rowError =[]
        for i, row in enumerate(df.iterrows()): 
            for key, value in row[1].iteritems():
                if value < MEC_MIN_MAX_DEFINED["min"][key]:
                    rowError.append(MEC_MIN_MAX_DEFINED["ErrorMin"][key])
                if value > MEC_MIN_MAX_DEFINED["max"][key]:
                    rowError.append(MEC_MIN_MAX_DEFINED["ErrorMax"][key])
            labels[i] = 1 if rowError else 0
            rowError =[]
        df['labels'] = labels
        
        return df
    
    def save_data(self, path):
        df = np.hstack((self.agv_in.reshape(-1,1),
                        self.dil.reshape(-1,1),
                        self.eapp.reshape(-1,1), 
                        self.sim_outputs.T,
                        self.qh2.reshape(-1,1)))

        df = pd.DataFrame(data=df,
                             columns=self.input_vars + self.output_vars)
        df.set_index(pd.date_range(end='2020-06-29',
                                      periods=len(df), 
                                      freq='H'), inplace=True)
        
        labels = np.zeros(len(df))
        rowError =[]
        for i, row in enumerate(df.iterrows()): 
            for key, value in row[1].iteritems():
                if value < MEC_MIN_MAX_DEFINED["min"][key]:
                    rowError.append(MEC_MIN_MAX_DEFINED["ErrorMin"][key])
                if value > MEC_MIN_MAX_DEFINED["max"][key]:
                    rowError.append(MEC_MIN_MAX_DEFINED["ErrorMax"][key])
            labels[i] = 1 if rowError else 0
            rowError =[]
        df['labels'] = labels
        
        df.to_csv(path)
        print(f'data saved in {path}')

        return df

    def ode(self, x, t):
        # print(x)
        # print("Imec_entrada {}:".format(x[4]))
        # print(Sin)
        # print(Dil)
        A = x[0]
        xa = x[1]
        xm = x[2]
        xh = x[3]
        Mox = x[4]
        Imec = x[5]

    # Ecuaciones cinéticas
        qa = ((self.qmaxa*A)/(self.Ksa+A))*(Mox/(self.KM + Mox))
        qm = (self.qmaxm*A)/(self.Ksm+A)
        mua = ((self.umaxa*A)/(self.Ksa+A))*(Mox/(self.KM + Mox))
        mum = (self.umaxm*A)/(self.Ksm+A)
        muh = (self.umaxh*self.H2)/(self.Kh+self.H2)

        alpha1 = 0.5410
        alpha2 = 0.4894

    # ecuaciones electroquimicas
    #    print("Mox:{}".format(Mox))
        Mred = self.Mtotal - Mox
    #    print("Mtotal:{}".format(Mtotal))
    #    print("Mred:{}".format(Mred))
    #     print("xa:{}".format(xa)) 
        Rint = self.Rmin + (self.Rmax - self.Rmin) * np.exp(- self.KR * xa)
        etha_concA = ((self.R * self.T) / (self.m * self.F)) * np.log(self.Mtotal / Mred)
        etha_actC = ((self.R * self.T)/(self.beta * self.m * self.F)) * np.arcsinh(Imec / (self.AsA * self.io))
        corriente = (self.E_CEF + self.E_app[self.i] - etha_concA - etha_actC)/Rint
    #    print("Corriente:{}".format(Corriente))

    # balance de masas
        # x0[2] = Sin
        dS = self.dil[self.i] * (self.agv_in[self.i] - A) - qa * xa - qm * xm
        dXa = mua * xa - self.Kda * xa - alpha1 * self.dil[self.i] * xa
        dXm = mum * xm - self.Kdm * xm - alpha1 * self.dil[self.i] * xm
        dXh = muh * xh - self.Kdh * xh - alpha2 * self.dil[self.i] * xh
        dMox = ((self.gamma*Imec)/(self.Vr*self.xa*self.m*self.F1)) - self.YM*qa

        return [dS, dXa, dXm, dXh, dMox, (corriente - Imec)]

