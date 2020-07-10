# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 08:43:59 2020

@author: 1052668570
"""
import numpy as np
import pandas as pd

from numba import njit

def _add_noise(vars_):
    pass


class Proceso:
    """ It model a process that use ordinary differential equations (ODE)
    to simulate a specific behaviour.
        As features one Variable have:
          * name,
          * sim_outputs np.array,
          * variables np.array of Variables objects,
          * input_vars np.array of Variables objects,
          * output_vars np.array of Variables objects,
      """

    def __init__(self, name='Proceso'):
        assert isinstance(name, str), "name must be a string"

        self.__name = name
        self.i = 0
        self.sim_outputs = np.array([])
        self.variables = np.array([])
        self.input_vars = np.array([])
        self.output_vars = np.array([])

    @property
    def name(self):
        return self.__name

    def set_input_vars(self, vars_):
        """ Define the inputs of type Variable for the process 
              Arguments:
              * vars_: list []
        """
        self.input_vars = np.append(self.input_vars, vars_)
        self.variables = np.append(self.variables, self.input_vars)

    def set_output_vars(self, vars_):
        """ Define the outputs of type Variable for the process 
              Arguments:
              * vars_: list []
        """
        self.output_vars = np.append(self.output_vars, vars_)
        self.variables = np.append(self.variables, self.output_vars)

    def ode():
        pass

    def initialize_outputs(self, tt):
        """  Set initial values of the outputs to the sim_outputs array used
        for initialize the ode integration
              Arguments:
              * tt: array of timesteps 
        """
        print(f'{self.name}: Initializing outputs for simulation')
        lt = len(tt)
        self.sim_outputs = np.zeros((len(self.output_vars), lt))
        self.sim_outputs[:, 0] = [x.vals[0] for x in self.output_vars]

    def add_random_noise(self):
        """  Add random noise using a normal distribution to a n variables 
        selected randomly. It use self.i to determine at which point will start 
        to add the noise
              Arguments:
              * self
        """
        var_list = self.input_vars
        amount_of_noise = 0
        num_of_vars = np.random.randint(0, len(var_list))
        vars_ = np.random.choice(var_list, size=num_of_vars,
                                 replace=False)
        # Add noise to the selected vars
        for var in vars_:
            # print(f'{self.name}: Adding noise to {var.name}')
            amount_of_noise = np.random.randint(1, 168)
            try:
                # Add bigger noise to agv_in
                if var.name == 'mec_agv_in':
                    std = np.random.uniform(10, 20, 1)
                    noise = np.random.randn(amount_of_noise) * std
                    var.vals[self.i:self.i+amount_of_noise] = noise +\
                        var.vals[self.i-2]
                else:
                    std = np.random.uniform(0, 5, 1)
                    noise = np.random.randn(amount_of_noise) * std
                    # Dealing with negatives by reducing them
                    noise[np.where(noise < 0)[0]] = noise[
                        np.where(noise < 0)[0]
                        ] * 0.1
                    # Adding the noise
                    var.vals[self.i:self.i+amount_of_noise] += noise
            except Exception as e:
                print('Action not completed', e)

    def update_outputs_variables(self):
        """ set the final simulation output values to the correspond
        variable array """
        for idx, var in enumerate(self.output_vars):
            var.vals = self.sim_outputs[idx, :]

    def get_batch_data(self, s, e):
        """
        * Get the chunk with labels (0, 1) to use it in the classifier.
        * The data returned will have the length of batch_size (start, end).
        * The dimensions will be the number of variables in the process plus 1
        (the label colum).
        * Then the final shape will be (batch_size, n_cols + 1)

        The classifier except data in the following order:
            ( [dil, agv_in, dqo_in, biomasa, dqo_out, agv_out],
              [agv_in, dil, eapp, xa, xm, xh, mox, imec, qh2] )
        """
        inp_stack = [var.vals[s:e].reshape(-1, 1) for var in self.input_vars]
        df = np.hstack((np.hstack(inp_stack), self.sim_outputs.T[s:e]))

        if self.name == 'da':
            df[:, -1] = df[:, -1] * self.PMAce  # agv_out conversion

        # Get labels [0, 1]
        labels = np.zeros(df.shape[0])
        vars_min = np.array([var.min for var in self.variables])
        vars_max = np.array([var.max for var in self.variables])

        for i, row in enumerate(df):
            are_below_min = np.any(row < vars_min)
            are_above_max = np.any(row > vars_max)
            # 1 or 0
            labels[i] = are_below_min | are_above_max
        return df, labels

    def save_data(self, path):
        """ Save the simulation data for each process in one pandas.DataFrame,
            * In which each column is a variable of the process and
              last column is the labels for the failure state."""
        inp_stack = [var.vals.reshape(-1, 1) for var in self.input_vars]
        df = np.hstack((np.hstack(inp_stack), self.sim_outputs.T))

        if self.name == 'da':
            df[:, -1] = df[:, -1] * self.PMAce  # agv_out conversion

        # Get labels
        labels = np.zeros(df.shape[0])
        vars_min = np.array([var.min for var in self.variables])
        vars_max = np.array([var.max for var in self.variables])

        for i, row in enumerate(df):
            are_below_min = np.any(row < vars_min)
            are_above_max = np.any(row > vars_max)
            # 1 or 0
            labels[i] = are_below_min | are_above_max

        cols = [var.name for var in self.variables]
        df = pd.DataFrame(data=df,
                          columns=cols)
        df.set_index(pd.date_range(end='2020-06-29',
                                   periods=len(df),
                                   freq='H'), inplace=True)
        df['labels'] = labels
        df.to_csv(path)
        print(f'{self.name}: data saved in {path}')

        return df

    def __str__(self):
        return f"Proceso {self.__name}"


class ProcesoDA(Proceso):
    def __init__(self, name):
        super(ProcesoDA, self).__init__(name)
        self.PMAce = 60.052  # gr/mol - Peso molecular del Acetato

    @staticmethod
    @njit
    def ode(x, t, *args):
        """ Ordinary differential equation (ODE) for Process DA.
            Arguments:
              * X : [biomasa, dqo_out, agv_out]
                t : timesteps
                args: [dil, agv_in, dqo_in]
            Returns:
              np.array([new_biomasa, new_dqo_out, new_agv_out]) """

        # print(f'inside ode-da: {x}')
        # print(f'inside ode-da: {t}')
        # print(self.i)

        dil = args[0]
        agv_in = args[1]
        dqo_in = args[2]

        biomasa = x[0]
        dqo_out = x[1]
        agv_out = x[2]

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
        dX1 = - alpha * dil * biomasa + u1 * biomasa
        dS1 = dqo_in * dil - dqo_out * dil - u1 * biomasa
        dS2 = agv_in * dil - agv_out * dil + k2k1 * u1 * biomasa

        # dZ1 = (Z1 - x(3))*D
        # dC1 = C1*D - x(4)*D  + k3*u1*x(0) - qC1*(V/Vg)
        # dC1g = -x(5)*(Qg/Vg)  + qC1*(V/Vg)

        return np.array([dX1, dS1, dS2])


class ProcesoMEC(Proceso):
    def __init__(self, name, **kwargs):
        super(ProcesoMEC, self).__init__(name)

        # For QH2 calcuations
        self.H2 = 1.0
        self.Kh = 0.0001
        self.YH2 = 0.9
        self.P = 1.0
        self.m = 2.0
        self.R2 = 0.08205
        self.F = 96485.0
        self.F1 = self.F / (60.0 * 60.0 * 24.0)
        self.T = 298.15
        self.Yh = 0.05
        self.V = 1.0
        self.umaxh = 0.5
        self.muh = (self.umaxh * self.H2) / (self.Kh + self.H2)

    def update_qh2(self):
        """ Calculate qh2 at iteration i with the new parameters
             calculated on the Odes
        """
        # sim_outputs[0:ace_out, 1:xa, 2:xm, 3:xh ,4:mox, 5:imec, 6:qh2]'
        self.sim_outputs[-1, self.i] = self.YH2 * (
            (self.sim_outputs[-2, self.i] * self.R2 * self.T) /
            (self.m * self.F1 * self.P)
            ) - self.Yh * self.muh * self.V * self.sim_outputs[3, self.i] / 1000

    @staticmethod
    @njit
    def ode(x, t, *args):
        """ Ordinary differential equation (ODE) for Process DA.
              Arguments:
                  X : [biomasa, dqo_out, agv_out]
                  t : timesteps
                  args: [dil, agv_in, dqo_in]
              Returns:
        np.array([new_ace_oute, new_xa, new_xm, new_xh, new_Mox, new_imec, qh2])
                   """
        agv_in = args[0]
        dil = args[1]
        eapp = args[2]

        A = x[0]
        xa = x[1]
        xm = x[2]
        xh = x[3]
        Mox = x[4]
        Imec = x[5]
        qh2 = x[6]

        # Parametros del modelo para ode
        qmaxa = 13.14               # d^-1
        qmaxm = 14.12               # d^-1
        Ksa = 20.0                    # mg L^-1
        Ksm = 80.0                    # mg L^-1
        KM = .01                    # mg L^-1
        umaxa = 1.97                # d^-1
        Kda = .04                   # d^-1 --> modificado
        umaxm = .3                  # d^-1
        Kdm = .01                   # d^-1 --> modificado
        umaxh = .5                  # d^-1
        Kdh = .01                   # d^-1
        Kh = .001                   # mg L^-1
        H2 = 1.0                      # mg L^-1
        gamma = 663400.0              # mg-M/mol_med
        Vr = 1.0                      # L
        m = 2.0                       # mol-e/mol-H2
        F = 96485.0                   # C mol-e^-1
        F1 = F/(60.0*60.0*24.0)           # A d mol-e^-1
        # YM = 34.85                 # mg-M/mg-A --> modificado
        YM = 3.3
        # Yh = 0.05                     # mLH2/mgX/d
        # Xmax1 = 512.5               # mg L^-1
        # Xmax2 = (1680.0+750.0)/2.0        # mg L^-1
        Mtotal = 1000.0               # mgM mg x^-1
        beta = 0.5
        AsA = 0.01                  # m^2
        # io =.005                   # Am^-2
        io = 1.0
        E_CEF = -0.35               # V
        E_app = eapp   # Default= .5 V == 500mv
        # Si E_app está en 0, existe una falla)
        # Por encima de 10 V, igual hay falla
        # Rmin = 20
        Rmin = 2.0
        # Rmax = 2000
        Rmax = 200.0
        KR = .024
        R = 8.314                   # J mol^-1 K^-1
        # R1 = 82.05                  # mL atm mol^-1 K^-1
        T = 298.15                  # K

        # Ecuaciones cinéticas
        qa = ((qmaxa*A)/(Ksa+A))*(Mox/(KM + Mox))
        qm = (qmaxm*A)/(Ksm+A)
        mua = ((umaxa*A)/(Ksa+A))*(Mox/(KM + Mox))
        mum = (umaxm*A)/(Ksm+A)
        muh = (umaxh*H2)/(Kh+H2)

        alpha1 = 0.5410
        alpha2 = 0.4894

        # ecuaciones electroquimicas
        #    print("Mox:{}".format(Mox))
        Mred = Mtotal - Mox
        #    print("Mtotal:{}".format(Mtotal))
        #    print("Mred:{}".format(Mred))
        #    print("xa:{}".format(xa))
        Rint = Rmin + (Rmax - Rmin) * np.exp(- KR * xa)
        etha_concA = ((R * T) / (m * F)) * np.log(Mtotal / Mred)
        etha_actC = ((R * T)/(beta * m * F)) * np.arcsinh(Imec / (AsA * io))
        corriente = (E_CEF + E_app - etha_concA - etha_actC)/Rint
        #    print("Corriente:{}".format(Corriente))

        # balance de masas

        dS = dil * (agv_in - A) - qa * xa - qm * xm
        dXa = mua * xa - Kda * xa - alpha1 * dil * xa
        dXm = mum * xm - Kdm * xm - alpha1 * dil * xm
        dXh = muh * xh - Kdh * xh - alpha2 * dil * xh
        dMox = ((gamma*Imec)/(Vr*xa*m*F1)) - YM*qa

        return np.array([dS, dXa, dXm, dXh, dMox, (corriente - Imec), qh2])
