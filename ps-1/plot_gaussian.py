import numpy as np
import matplotlib.pyplot as plt

#setting upper and lower bounds for x axis
x1=-10
x2=10

#setting the number of points and step size
N=1000
delta=(x2-x1)/N

#setting the mean and variation
mu=0
sigma=3

#defining the gaussian function
def f(x=None):
    """   
    Parameters
    ----------
    
    x : float
        input variable
    
    Returns
    -------
    
    val : float
        output variable
    
    Comments
    --------
    
    Value returned is e^(-(x-mu)^2/(2*sigma^2))/sqrt(2*pi*sigma^2).
    """
    return(np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2))

#creating arrays for the coordinates of plotted points 
xpoints = np.arange(x1, x2+delta, delta)
ypoints = np.array([f(xi) for xi in xpoints])

#plotting (with title and additional labels)
plt.plot(xpoints, ypoints)
plt.xlabel('X (au)')
plt.ylabel('Y (au)')
plt.figtext(.8, .8, r"$\mu$ = " + str(mu))
plt.figtext(.8, .75, r"$\sigma$ = " + str(sigma))
plt.title("Gaussian")
#plt.show()  
plt.savefig('gaussian.png')