import numpy as np
import matplotlib.pyplot as plt

#define the relevant parameter
a=[2,3,4]
N=10

#define the integrand function, avoiding numerical error for small x
def f(x,j):
    return np.exp((j-1)*np.log(x)-x)

#in the laguerre integral, the weight e^-x is removed
def f_leg(x,j):
    return x**(j-1)

#invers of the change of variable function
def q(z,j):
    return (j-1)*z/(1-z)

#gamma function with gauss integral
def gamma(j):
    xp,wp=np.polynomial.legendre.leggauss(N)
    xp=q(xp,j)
    wp=wp/(j-1)
    int_temp=np.zeros(N, dtype=np.float32)
    def g(x,j):
        return f(x,j)*(j-1+x)**2
    for i in np.arange(N):
        if xp[i]<=0:
            int_temp[i]=0
        else: 
            int_temp[i]=wp[i]*g(xp[i],j)
    return int_temp.sum()

#gamma function with laguerre integral
def gamma_leg(j):
    xp,wp=np.polynomial.laguerre.laggauss(N)
    int_temp=np.zeros(N, dtype=np.float32)
    for i in np.arange(N):
        int_temp[i]=wp[i]*f_leg(xp[i],j)
    return int_temp.sum()

#make the plot
m=1000
x=np.linspace(0.+5./m,5., m)

plt.figure(figsize=(8, 6))
for j in a:
    plt.plot(x, f(x,j), label=f'a={j}')
plt.title(r'$x^{a-1}e^{-x}$')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
plt.legend()
plt.savefig('Gamma_int.png')

#evaluate for different values of a
aa=np.array([3.,6.,10.,1.5])
fact=np.array([2.,120.,362880.,0.5*np.sqrt(np.pi)]) #known results
#evaluation for gaussian integral
gammaa=np.array([gamma(ai) for ai in aa])
print(gammaa,(gammaa-fact)/fact )
#evaluation for laguerre integral
gammaa_leg=np.array([gamma_leg(ai) for ai in aa])
print(gammaa_leg,(gammaa_leg-fact)/fact)