import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as sp

x_min=-4
x_max=4

N=[0,1,2,3]

def H0(x):
    return 1

def H1(x):
    return 2*x

HH=[H0,H1, 0, 0]

def H(n):
    if n<len(HH):
        return HH[n]
    else:
        def h(x):
            return 2*x*H(n-1)(x)-2*(n-1)*H(n-2)(x)
        HH[n]=h
        print(HH)
        print(n)
        return HH[n]
    
x = (H(3)(2))

#print(HH)