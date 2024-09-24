import numpy as np
import matplotlib.pyplot as plt
from random import random 

dt=1 #time increments
N=20000 #total ties

#probability of decay in a single step, as a function of half-life
def f(t):
    '''
    Compute probability of decay
    '''
    return(1-2**(-dt/t))

#data about the half-lives
t_Bi213=46*60
t_Tl=2.2*60
t_Pb=3.3*60

#channel probability of the two-channel decay
pBi=np.float32(.9791)

#creating arrays of the populations over time
Bi213=np.zeros(N, dtype=np.uint16)
Tl=np.zeros(N, dtype=np.uint16)
Pb=np.zeros(N, dtype=np.uint16)
Bi209=np.zeros(N, dtype=np.uint16)

#setting initial conditions
Bi213[0]=Bi0=10000

#time array for the plot
t=np.linspace(0,N*dt, N)

#starting from the bottom, loop the decay process
for i in np.arange(N):
    #first update population
    if i>0:
        Bi209[i]=Bi209[i-1]
        Pb[i]=Pb[i-1]
        Bi213[i]=Bi213[i-1]
        Tl[i]=Tl[i-1]
    decay=0

    #for each partile, decide wether it decays or not
    for j in np.arange(Pb[i]):
        if random()<=f(t_Pb):
            decay+=1
    
    #update popultation decaied
    Bi209[i]+=decay
    Pb[i]-=decay
    decay=0

    #rinse and repeat
    for k in np.arange(Tl[i]):
        if random()<=f(t_Tl):
            decay+=1
    
    Tl[i]-=decay
    Pb[i]+=decay
    decay=0
    decayTl=0

    for l in np.arange(Bi213[i]):
        if random()<=f(t_Bi213):
            decay+=1
            if random()>pBi:
                decayTl+=1
    Tl[i]+=decayTl
    Pb[i]+=decay-decayTl
    Bi213[i]-=decay

#plot the data
plt.figure(figsize=(8, 6))
plt.plot(t, Bi213, label=r'$^{213}Bi$')
plt.plot(t, Tl, label=r'$^{209}Tl$')
plt.plot(t, Pb, label=r'$^{209}Pb$')
plt.plot(t, Bi209, label=r'$^{209}Bi$')
plt.title('Chain decay')
plt.xlabel('t [s]')
plt.ylabel('N')
plt.legend()
plt.show()
plt.savefig('chain_decay.png')

