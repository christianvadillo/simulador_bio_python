# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:46:11 2020

@author: 1052668570
"""

from numba import jit
import timeit

@jit(nopython=True)
def ode(x, t):
        # print(f'inside ode-da: {x}')
        # print(f'inside ode-da: {t}')
        # Parámetros del modelo de la DA
        # kLaCO2 = 2000  # 1/d
        # KHCO2  = .0271 # M/bar
        # Tr = 308       # K
        # Vr = 4.4       # L
        # Vg = .1
        # R = 8.314e-2   # bar/KM
        u_max1 = .31  # d^-1
        k_s1 = 24     # gDQO/L
        alpha = .18
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

        dX1 = - alpha * 0.6 * x[0] + u1 * x[0]
        dS1 = 24 * 0.6 - x[1] * 0.6 - u1 * x[0]
        dS2 = 100 * 0.6 - x[2] * 0.6 + k2k1 * u1 * x[0]
        # dZ1 = (Z1 - x(3))*D
        # dC1 = C1*D - x(4)*D  + k3*u1*x(0) - qC1*(V/Vg)
        # dC1g = -x(5)*(Qg/Vg)  + qC1*(V/Vg)
        # self.i += 1

        return np.array([dX1, dS1, dS2])


import numpy as np
from scipy.integrate import odeint


# The ``driver`` that will integrate the ODE(s)
def integrate_ode():
    tt = np.arange(0, 300, .041666)
    lt = len(tt)
    T = tt[1+0:1+2:1].astype(np.float32)
    # print(T)
    # print(T.astype(np.float32))
    init_data = np.array([24.8, 12.5, 42.])
    
    # numba_f = numba.jit(ode,nopython=True)
    t0, t1 = T              # start and end
    t = np.linspace(t0, t1, 11)  # the points of evaluation of solution
    y0 = init_data               # initial value
    y = np.zeros((len(t), len(y0)))   # array for solution
    y[0, :] = y0
    sol = odeint(func=ode, y0=y0.astype(np.float32), t=t.astype(np.float32))
    # print(sol)



# from scipy.integrate import odeint
# import numba
# import timeit

# def rober(u,t):
#   k1 = 0.04
#   k2 = 3e7
#   k3 = 1e4
#   y1, y2, y3 = u
#   dy1 = -k1*y1+k3*y2*y3
#   dy2 =  k1*y1-k2*y2*y2-k3*y2*y3
#   dy3 =  k2*y2*y2
#   return [dy1,dy2,dy3]

# u0  = [1.0,0.0,0.0]
# t = [0.0, 1e5]

# numba_f = numba.jit(rober,nopython=True)

# def time_func():
#     sol = odeint(rober,u0,t,rtol=1e-6,atol=1e-6)

print(timeit.Timer(integrate_ode).timeit(number=10000))
# integrate_ode()


import numpy as np
amount_of_noise = 0
input_vars = ['dil', 'agv_in', 'dqo_in']
num_of_vars = np.random.randint(0, len(input_vars))
vars_ =  np.random.choice(input_vars, size=num_of_vars,
                          replace=False)


try:
    for var in vars_:
        if var == 'dil':
            amount_of_noise = np.random.randint(1, 24)
            noise = np.random.randn(amount_of_noise)*40
            dil[i:i+amount_of_noise] += noise  # Adding the noise

        elif var == 'agv_in':
            amount_of_noise = np.random.randint(1, 24)
            noise = np.random.randn(amount_of_noise)*30
            self.agv_in[self.i:self.i+amount_of_noise] += noise  # Adding the noise
        else:
            amount_of_noise = np.random.randint(1, 24)
            noise = np.random.randn(amount_of_noise)*30
            self.dqo_in[self.i:self.i+amount_of_noise] += noise  # Adding the noise

    # Changing negative values to zero
    self.transform_negatives()
except Exception as e:
    print("No se pudo completar la acción", e)


a = np.array([[360.]])
b = np.zeros((2,2))
