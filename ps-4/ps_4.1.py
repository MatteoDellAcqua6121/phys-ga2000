import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as sp

#Defining the constants
#Leaving the power of 10 for later (it will pick up a 10^-1)
k_B=1.380649
theta=428
V=1000
rho=6.022

#number of points for the gaussian
n=50

#prefactor of the integral
def g(x):
    return 9*V*k_B*rho*(x/theta)**3/10

#integrand function
def f(x):
    return x**4*np.exp(x)/(np.exp(x)-1)**2

#final function
def c_T(T,N):
    #generate root and weights
    xp,wp=np.polynomial.legendre.leggauss(N)
    #rescale them according to integration interval
    a=0
    b=theta/T
    xp=0.5*(b-a)*xp+0.5*(b+a)
    wp=0.5*(b-a)*wp
    int_temp=np.zeros(N, dtype=np.float32)
    #temporarily store the contributions and use np.sum
    for i in np.arange(N):
        int_temp[i]=wp[i]*f(xp[i])
    return g(T)*int_temp.sum()

#plot everything
T_min=5
T_max=500

m=1000
step=(T_max-T_min)/m #non è esattissimo
x=np.linspace(T_min, T_max+step, m)
y=np.array([c_T(xi,n) for xi in x])

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title(r'$C_T$(T)')
plt.xlabel('T [K]')
plt.ylabel(r'$C_T$(T) [J/K]')

plt.show()
plt.savefig('Debye_T.png')


T=5 #fix a temperature such that the graph is somewhat intereting
N=[10,20,30,40,50,60,70]

#plot everything
y=np.array([c_T(T,ni) for ni in N])

plt.figure(figsize=(8, 6))
plt.plot(N, y)
plt.title(r'$C_T$(T=5K) vs N')
plt.xlabel('N')
plt.ylabel(r'$C_T$(T) [J/K]')

plt.show()
plt.savefig('Debye_convergence.png')

plt.figure(figsize=(8, 6))
for j in N:
    y=np.array([c_T(xi,j) for xi in x])
    plt.plot(x, y, label=f'N={j}')

plt.title(r'$C_T$(T)')
plt.xlabel('T [K]')
plt.ylabel(r'$C_T$(T) [J/K]')    
plt.legend()
plt.show()

plt.savefig('Debye_conv_T.png')


