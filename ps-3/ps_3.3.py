import numpy as np
import matplotlib.pyplot as plt
from random import random 

t_Tl=3.053*60 #half life
N=1000 #number of decays

#create array of decay times
decays=np.zeros(N, dtype=np.float32)

#extract random times of decay
for i in np.arange(N):
    decays[i]=-t_Tl*np.log(1-random())/np.log(2)

decays=np.sort(decays)

M=10000     #number of points we are going to plot
d=decays[N-1]/M #set the time range and size by looking at the greatest time of decay
x=np.linspace(0,decays[N-1]+d,M) #array of the data points (x coordinates)
Tl=np.zeros(M, dtype=np.float32) #array of the data points (y coordinates)
Tl[0]=1000 #set initial conditions
Pb=np.zeros(M, dtype=np.float32)#array of the data points (y coordinates)
T=0#set initial conditions

#for every time t in our discretized array, compute the number of particles that decayed between t and t-dt
for i in np.arange(M):
    if i>0:
        Tl[i]=Tl[i-1]
        Pb[i]=Pb[i-1]
    dec=0
    while decays[T]<(i+1)*d:
        dec+=1
        T+=1
    Tl[i]-=dec
    Pb[i]+=dec

#plot data
plt.figure(figsize=(8, 6))
plt.plot(x, Tl, label=r'$^{208}Tl$')
plt.plot(x, Pb, label=r'$^{208}Pb$')
plt.title(r'$^{208}Tl \ \to \ ^{208}Pb$ decay')
plt.xlabel('t [s]')
plt.ylabel('N')
plt.legend()
plt.show()
plt.savefig('Tl_decay.png')
