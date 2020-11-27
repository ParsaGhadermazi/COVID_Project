import pandas as pd 
import numpy as np 
#import ACATlib
import importlib
importlib.reload(ACATlib)
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
from scipy.optimize import curve_fit
from scipy.integrate import odeint
#====================================
''' this work is inspired by the curret global situation with the hope of a more profound understanding which could result in less devastating outcomes
written by : sohaib habib
SIR : The Kermack-McKendrick model'''
#==================================== 
#chinease data 
tspan = np.linspace(0,60,num = 61)
inf = np.array([554,771,1208,1870,2613,4349,5739,7417,9308,11289,13748,16369,19383,22942,26302,28985,
31774,33738,35982,37626,38791,51591,55748,56873,57416,57943,58016,57805,56301,54921,53284,52093,
49824,47765,45600,43258,39919,37414,35129,32616,30004,27423,25353,23784,22179,20533,19016,
17781,16136,14831,13524,12088,10733,9893,8967,8056,7263,6569,6013,5353,5120])
N_c = 11080000/118
sus = N_c-inf[0]
inf_=inf/N_c
# parameters 
beta = 0.34630808
gama = 0.06379039 
#====================================
#qurantine_magicly the suxiptibles 
sus_0 = 165000
inf_0 = 10
N = sus_0 + inf_0 
N_ = [N]#,0.9*N,0.75*N]
sus = [0.85*sus_0]#,0.75*sus_0,0.5*sus_0]
color = ['b']#,'r','g','k'] 
label = ['15%_qur']#,'25%_qur','50%_qur']
day = [0,200]
results = []
fraction_quar = 0.9
I_max = np.zeros(len(sus))
t_max = np.zeros(len(sus))
#=================================== 
for i in range(len(sus )) : 
#model dynamics 
 def model(sir,t):
     dsdt = -beta*sir[0]*sir[1]
     didt = beta*sir[0]*sir[1]-gama*sir[1]
     drdt = gama*sir[1]
     dsirdt = [dsdt,didt,drdt]
     return dsirdt
 #================================= 
 sir_0 = [sus[i]/N,inf_0/N,0/N]
 #=================================
 #time span 
 t = np.linspace(0,60,num = 61)
 #=================================
 #ode_solution 
 sir = odeint(model,sir_0,t)
 A = sir[:,1]
 #=================================
 I_max[i] = np.max(A)
 I_max_loc = np.argmax(A)
 t_max[i] = t[I_max_loc] 
 #=================================
 # results 
 # epidemiological threshold 
R0 = beta/gama
#====================================
#====================================
#fitting 
sir_0 = [sus_0/N,inf_0/N,0/N]
def fitfunc(t,alpha,beta,gama):
    def sir_model(sir,t):
        dsdt = -alpha*sir[0]*sir[1]+gama*sir[2]
        didt = alpha*sir[0]*sir[1]-beta*sir[1]
        drdt = beta*sir[1]-gama*sir[2]
        dsirdt = [dsdt,didt,drdt]
        return dsirdt
    sir_0 = [(N-inf_0)/N,inf_0/N,0/N]
    sir = odeint(sir_model,sir_0,t)
    return sir[:,1]
p_fit, kcov = curve_fit(fitfunc, t, A,p0 = [1,0.34,0.06])
#==========================
#model dynamics 
alpha = p_fit[0]*2.2
beta = p_fit[1]*.96
gama = p_fit[2]*0.11
def sir_model(sir_,t):
    ds_dt = -alpha*sir_[0]*sir_[1]+gama*sir_[2]
    di_dt = alpha*sir_[0]*sir_[1]-beta*sir_[1]
    dr_dt = beta*sir_[1]-gama*sir_[2]
    dsir_dt = [ds_dt,di_dt,dr_dt]
    return dsir_dt
sir_0_ = [(N-inf_0)/N,inf_0/N,0/N]
sir_ = odeint(sir_model,sir_0_,t)
print(p_fit)
plt.figure(2)
plt.plot(tspan,inf_,label='actual data',color = 'b')
plt.plot(tspan,sir_[:,1],label='fitted model',color = 'r')
plt.xlabel('days')
plt.ylabel('fraction infected')
plt.text(15,0,'k1(sus to inf) = 0.6996(1/(individual.day)) ,k2(inf to rec) = 0.0844(day^(-1)) ,k3 (rec to sus) =0.0020(day^(-1)) ')
plt.legend()
plt.grid()
plt.show()





