import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as sp

N=20 #number of points in gaussian quadrature

#define integrand function
def f(x,a):
    return 1/np.sqrt(a**4-x**4)

def T(a,N):
    #generate weight and roots
    xp,wp=np.polynomial.legendre.leggauss(N)
    #rescale
    xp=0.5*xp*a+0.5*a
    wp=0.5*a*wp
    #sum
    int_temp=np.zeros(N, dtype=np.float32)
    for i in np.arange(N):
        int_temp[i]=wp[i]*f(xp[i],a)
    return np.sqrt(8)*int_temp.sum()

#plot everything
a_min=0
a_max=2

m=1000
step=(a_max-a_min)/m #non è esattissimo
x=np.linspace(step+a_min, a_max, m)
y=np.array([T(xi,N) for xi in x])

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title(r'Anharmonic oscillator')
plt.xlabel('Amplitude [au] (or [m])')
plt.ylabel('T [au] (or [s])')

plt.show()
plt.savefig('Anharmonic.png')