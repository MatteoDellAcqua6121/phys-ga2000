import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as sp

#define limits of the plot and indexes for the hwaveform we will plot
x_min=-4
x_max=4

N=[0,1,2,3]

#define hermite using a mixture of recursion and dynamic programming:
def HH(n, x):
   
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    
    #Create an array to store Hermite values
    temp=np.zeros(n+1, dtype=np.float32)
    temp[0]=1
    temp[1]=2*x
    
    # Fill the table using the recurrence relation
    for i in range(2, n + 1):
        temp[i]=2*x*temp[i-1]-2*(i-1)*temp[i - 2]
    
    return temp[n]

'''
#define hermite using dynamic programming: mixture of recursion and optimization to call HH the least number of times
cache = [lambda x: np.ones(x.shape), lambda x: 2*x]

def HH(n):
    if n >= len(cache):
        def f(x):
            return 2*x*HH(n-1)(x) - 2*(n-1)*HH(n-2)(x)
        
        f(np.array([0])) # THIS f NEEDS TO BE CALLED
        cache.append(f)
 
    return cache[n]


def H0(x):
    return 1

def H1(x):
    return 2*x

HH=[H0,H1]

def H(n):
    if n<len(HH):
        return HH[n]
    else:
        def h(x):
            return 2*x*H(n-1)(x)-2*(n-1)*H(n-2)(x)
        HH.append(h)
        return HH[n]
'''

#define psi in term of the hermite
def psi(n):
    def f(x):
        return np.exp(-0.5*x**2)*HH(n,x)/(np.sqrt(np.sqrt(np.pi)*np.exp2(n)*math.factorial(n)))
    return f

#setup coordinate arrays
m=1000
x=np.linspace(x_min,x_max,m)
y=np.zeros([4,m], dtype=np.float32)
for i in N:
    for j in np.arange(m):
        y[i][j]=psi(i)(x[j])

#plot everything
plt.figure(figsize=(8, 6))
plt.plot(x, y[0], label=r'$\psi_0$')
plt.plot(x, y[1], label=r'$\psi_1$')
plt.plot(x, y[2], label=r'$\psi_2$')
plt.plot(x, y[3], label=r'$\psi_3$')
plt.title('Harmonic Oscillator Wavefunctions')
plt.xlabel('x')
plt.ylabel(r'$\psi_i$')
plt.legend()
plt.show()
plt.savefig('wavefunctions_multiple.png')

#repeat for just one very high value of n
x_min=-10
x_max=10
n=30


#plot everything
m=1000
xx=np.linspace(x_min,x_max,m)
yy=np.array([np.exp(-0.5*xi**2)*HH(n,xi)/(np.sqrt(np.sqrt(np.pi)*np.exp2(n)*math.factorial(n))) for xi in xx])
plt.figure(figsize=(8, 6))
plt.plot(xx, yy, label=r'$\psi_{30}$')
plt.title('Harmonic Oscillator Wavefunction')
plt.xlabel('x')
plt.ylabel(r'$\psi_i$')
plt.legend()
plt.show()
plt.savefig('wavefunctions_high.png')

#
NN=[6,7,100]


def uncertainty(n,NNN):
    xp,wp=np.polynomial.legendre.leggauss(NNN)
    int_temp=np.zeros(NNN, dtype=np.float32)
    zp=xp/(1-xp**2)
    for i in np.arange(NNN):
        int_temp[i]=wp[i]*((1+xp[i]**2)/(1-xp[i]**2)**2)*(zp[i])**2*HH(n,zp[i])**2*np.exp(-(zp[i])**2)/(np.sqrt(np.pi)*np.exp2(n)*math.factorial(n))
    #print(int_temp)
    return int_temp.sum()

def herm_uncertainty(n,NNN):
    xp,wp=np.polynomial.hermite.hermgauss(NNN)
    int_temp=np.zeros(NNN, dtype=np.float32)
    for i in np.arange(NNN):
        int_temp[i]=wp[i]*(xp[i])**2*HH(n,xp[i])**2/(np.sqrt(np.pi)*np.exp2(n)*math.factorial(n))
    #print(int_temp)
    return int_temp.sum()

for i in NN:
    print(np.sqrt(uncertainty(5,i)))
    print(np.sqrt(herm_uncertainty(5,i)))
    print(np.sqrt(5.5)-np.sqrt(uncertainty(5,i)))
    print(np.sqrt(5.5)-np.sqrt(herm_uncertainty(5,i)))