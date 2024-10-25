import numpy as np

#Mass in Kg
m_sun=1.989e30 
m_earth=5.974e24
m_moon=7.348e22
m_jupiter=1.898e27

#Distance in m
r_sun_earth=148.73e9
r_moon_earth=3.844e8

#Create arrays for radius and m'
r=[r_moon_earth, r_sun_earth, r_sun_earth]
m=[m_moon/m_earth, m_earth/m_sun, m_jupiter/m_sun]

#define the polynomial whose root we are trying to find (and its derivative)
def func(x,m):
    return((1.-x)**2-m*x**2-x**3*(1.-x)**2)

def dfunc(x,m):
    return(-2. - 2.*(-1. + m)* x - 3.* x**2 + 8. *x**3 - 5. *x**4)

#import the bracket and newton functions from the class jupiter notebook
def bracket(func,m):
    a = 0.4
    b = 0.6
    maxab = 1.e+7
    while(b - a < maxab):
        d = b - a
        a = a - 0.1 * d
        b = b + 0.1 * d
        if(func(a,m) * func(b,m) < 0.):
            return(a, b)
    return(a, b)

def newton_raphson(xst,m):
    tol = 1.e-10
    maxiter = 100
    x = xst
    for i in np.arange(maxiter):
        delta = - func(x,m) / dfunc(x,m)
        x = x + delta
        if(np.abs(delta) < tol):
            return(x)
        
#run the root-finding (and print the results) for each element of the array
for i in np.arange(len(m)):
    (a, b) = bracket(func,m[i])
    z = newton_raphson(0.5*(a+b),m[i])*r[i]
    print(z)