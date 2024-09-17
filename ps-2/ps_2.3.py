import numpy as np
import matplotlib.pyplot as plt
import timeit

L=100

def V(i, j, k):
    return(-(-1)**(np.absolute(i+j+k))/np.sqrt(i**2+j**2+k**2))

def madelung_for(L):
    #setting the intial value of the Madelung constant and the range
    M=0
    r=np.arange(-L,L+1)

    #run one for loop for each coordinate (and avoid the origin)
    for i in r:
        for j in r:
            for k in r:
                if i|j|k:
                    M+=V(i,j,k)

    #output the Madelung constant
    return(M)

def madelung_not(L):
    #set the range and an array of coordinates
    r=np.arange(-L,L+1)
    coordinates_x, coordinates_y, coordinates_z=np.meshgrid(r,r,r, indexing='xy')
    
    #implement the function on coordinates
    M=np.where(coordinates_x|coordinates_y|coordinates_z, V(coordinates_x, coordinates_y, coordinates_z),0)
    
    #return the sum
    return(M.sum())
 
print("Madelung constant:")
print('using for loops =', madelung_for(L))
print('without for loops =', madelung_not(L))

print('Time needed')
print('using for loops =', timeit.Timer('madelung_for(100)', 'from __main__ import madelung_for').timeit(number=1))
print('without for loops =', timeit.Timer('madelung_not(100)', 'from __main__ import madelung_not').timeit(number=1))
