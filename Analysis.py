import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
Table=pd.read_csv("US_Data.csv")
Positive_Cumulative=Table["positive"]
Recovered_Cumulative=Table["recovered"]
Positive_Cumulative=list(Positive_Cumulative)[::-1]
Recovered_Cumulative=list(Recovered_Cumulative)[::-1]
Active_Cases=[a_i - b_i for a_i, b_i in zip(Positive_Cumulative, Recovered_Cumulative)]
# plt.plot(range(len(Positive_Cumulative)),Positive_Cumulative[::-1],label="Positive_Cumulative")
# plt.plot(range(len(Recovered_Cumulative)),Recovered_Cumulative[::-1],label="Recovered_Cumulative")
j=0
Daily_Recovered=[0]*len(Recovered_Cumulative)

Recovered_Cumulative=Recovered_Cumulative
Daily_Recovered[0]=Recovered_Cumulative[0]
print(Recovered_Cumulative)
for items in Recovered_Cumulative[1:]:
    j += 1
    Daily_Recovered[j]=Recovered_Cumulative[j]-Recovered_Cumulative[j-1]

m,b=np.polyfit(Active_Cases,Daily_Recovered,1)
print(f'm={m} , b={b}')
plt.scatter(Active_Cases,Daily_Recovered,s=5,label="Real Data",c="#5068A8")
plt.text(3000000,140000,'k: I --> R= '+str(m))
plt.plot(Active_Cases,[m*x+b for x in Active_Cases  ],label="label=fitted_line",color= "#E74632")
plt.legend()

plt.ylabel("Daily Recovery")
plt.xlabel("Active Cases")
plt.show()
plt.savefig("Fitting Result")
