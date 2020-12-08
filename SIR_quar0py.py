import pandas as pd 
import numpy as np 
#import ACATlib
import importlib
#importlib.reload(ACATlib)
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
from scipy.optimize import curve_fit
from scipy.integrate import odeint
#=================================
inf_0 = 1
sus_0 = (11080000/118)-inf_0
N = 11080000/118
#N_ = [0.85*N,0.75*N,0.5*N]
sus = [sus_0]#,0.75*sus_0,0.5*sus_0]
color = ['b','r','g','k'] 
label = ['no_testing','25%tested','50%_tested','70%_tested']
alpha = 0.7 
beta = 0.08 
gama = 0.002
fr = np.array([0.00001,0.25,0.5,0.7])
#===============================
for i in range(len(fr)) :
     fru = 1-fr[i]
    # selective quar
     def model(sir,t):
        ds_dt = -alpha*sir[0]*sir[2]+gama*sir[3]
        dQ_dt = fr[i]*(alpha*sir[0]*sir[2])-beta*sir[1]
        dUQ_dt = fru*(alpha*sir[0]*sir[2])-beta*sir[2]
        dr_dt = beta*(sir[1]+sir[2])-gama*sir[3]
        dsir_dt = [ds_dt,dQ_dt,dUQ_dt,dr_dt]
        return dsir_dt
     #================================= 
     sir_0 = [sus[0]/N,0/N,inf_0/N,0/N]
 #=================================
 #time span 
     t = np.linspace(0,100,num = 101)
 #=================================
 #ode_solution 
     sir = odeint(model,sir_0,t)
     A = sir[:,1]
     B = sir[:,2]
     c = np.zeros(len(A))
     for j in range(len(A)):
         c[j] = A[j]+B[j]

     plt.plot(t,(c/10),label=label[i],color = color[i])
     plt.xlabel('days')
     plt.ylabel('fraction infected')
plt.legend()
plt.grid()
plt.show()