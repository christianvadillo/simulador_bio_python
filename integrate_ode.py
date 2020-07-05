# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:05:46 2019

@author: 1052668570
"""
# from scipy import integrate
import numpy as np
from scipy.integrate import odeint


# The ``driver`` that will integrate the ODE(s)
def integrate_ode15s(T, funcion, cond_ini):
    t0, t1 = T              # start and end
    t = np.linspace(t0, t1, 11)  # the points of evaluation of solution
    y0 = cond_ini               # initial value
    y = np.zeros((len(t), len(y0)))   # array for solution
    y[0, :] = y0
    #r = integrate.ode(funcion).set_integrator('vode', method='adams')# specifying the integrator (ode15s equivalent)
    #r = integrate.ode(funcion).set_integrator('lsoda')# specifying the integrator (ode15s equivalent)
    #r.set_initial_value(y0, t0).set_f_params(*args)   # Set initial condition(s): for integrating variable and time!

    sol = odeint(func=funcion, y0=y0, t=t)
#    for i in range(1, t.size):
#        y[i, :] = r.integrate(t[i])  # get one more value, add it to the array
#        if not r.successful():
#            raise RuntimeError("Could not integrate")
#    # plt.plot(t, y)
#    # plt.show()
    #return (t, y)
    return (t, sol)

# sol = odeint(funcion, y0, t, args = tuple(args))
