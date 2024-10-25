import numpy as np
import scipy

#import the braketing function form the class Jupiter notebook
def bracket(func):
    a = 0.
    b = 1.
    maxab = 1.e+7
    while(b - a < maxab):
        d = b - a
        a = a - 0.1 * d
        b = b + 0.1 * d
        if(func(a) * func(b) < 0.):
            return(a,0.5*(a+b), b)
    return(a, 0.5*(a+b),b)
#import the parabolic step function form the class Jupiter notebook
def parabolic_step(func=None, a=None, b=None, c=None):
    """returns the minimum of the function as approximated by a parabola"""
    fa = func(a)
    fb = func(b)
    fc = func(c)
    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)
    # If singular, just return b 
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)
#modify the golden function form the class Jupiter notebook to implement only one step. Achtnung: now the output is the whole new bracketing
def golden_step(func=None, astart=None, bstart=None, cstart=None, tol=1.e-5):
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart
    # Split the larger interval
    if((b - a) > (c - b)):
        x = b
        b = b - gsection * (b - a)
    else:
        x = b + gsection * (c - b)
    fb = func(b)
    fx = func(x)
    if(fb < fx):
        return (a,b,x)
    else:
        return (b,x,c)
       
#define the function we are looking to minimize and ts derivative
def func(x):
    return((x-0.3)**2*np.exp(x))

def dfunc(x):
    return(np.exp(x)* (-0.3 + x) *(1.7 + x))

#define the brent function
def brent(f,astart,bstart,cstart, tol=1.e-5, maxiter=10000):
    a = astart
    b = bstart
    c = cstart
    bold = b + 2. * tol
    niter = 0
    while((np.abs(bold - b) > tol) & (niter < maxiter)):
        bold = b
        #compute the parabolic step
        b = parabolic_step(func=func, a=a, b=b, c=c)
        if(a< b < bold):
            c = bold
        elif(bold<b<c):
            a = bold
        #use the golden step for anomalous cases: eithe b outside of the interval or q=0 (remember that in this case, the parabolic_step function just print out bold)
        else:
            (a,b,c)=golden(func=func, a=a, b=b, c=c)
        niter = niter + 1
    return(b)

#since the bracketing looks for zeroes, and we are looking to minimize, we apply the bracketing to the derivative
brac=bracket(dfunc)
z=brent(func,*brac)
z_sp=scipy.optimize.brent(func,brack=brac)

#compare our result with scipy function: print difference and relative error
print(z,z_sp)
print(z-z_sp)
print((z-z_sp)/z)