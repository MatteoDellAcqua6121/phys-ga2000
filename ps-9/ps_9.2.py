import numpy as np
import matplotlib.pyplot as plt
import scipy
#parameters
A=(0.08)**2*1.22*0.47*9.81
v0=100
class Cannonball(object):
    """Class to calculate the cannonball problem
    
    Parameters
    ----------
    
    C : np.float32
        dimensionless coefficient

    diff : np.float32
        amount for finite difference derivative for potential
    
    Set up to use RK4/5 by default
"""
    def __init__(self, a=A, diff=1.e-7):
        self.a = a
        self.diff = diff
        self.xdiff = np.array([self.diff,0.])
        self.ydiff = np.array([0.,self.diff])
        self.set_ode()
        return 
    def set_ode(self):
        """Setup ODE integrator (RK5)"""
        self.ode = scipy.integrate.ode(self.dwdt)
        self.ode.set_integrator('dopri5') # Runge-Kutta
        return
    def _diff_central(self, func=None, x=None, dx=None, factor=1.):
        """Central difference"""
        return((func(x + 0.5 * dx * factor) - func(x - 0.5 * dx * factor)) /
               (factor * self.diff))
    def _diff_correct(self, func=None, x=None, dx=None):
        """Higher order difference"""
        return((4. * self._diff_central(func=func, x=x, dx=dx, factor=0.5) -
                self._diff_central(func=func, x=x, dx=dx)) / 3.)

    def dwdt(self, t, w):
        """Phase space time derivative
        
        Parameters
        ----------
        
        t : np.float32
            current time
            
        w : ndarray of np.float32
            [4] phase space coords (positions and velocities)
        
        Returns
        -------
        
        dwdt : ndarray of np.float32
            [4] time derivatives to integrate
"""
        x = w[:2]
        v = w[2:]
        dwdt = np.zeros(4)
        dwdt[:2] = v
        dwdt[2:] = np.array([-0.5*np.pi*self.a*v[0]*np.sqrt(v[0]**2+v[1]**2), -1-0.5*np.pi*self.a*v[1]*np.sqrt(v[0]**2+v[1]**2)])
        return(dwdt)
    
    def integrate(self, w0=None, t0=0., dt=0.1, nt=100):
        """Integrate the equations
        
        Parameters
        ----------
        
        t0 : np.float32
            initial time
            
        w0 : ndarray of np.float32
            [4] initial phase space coords (position and velocity)
            
        dt : np.float32
            time interval to integrate per output
            
        nt : np.int32
            number of intervals
"""
        self.ode.set_initial_value(w0, t0)
        w = np.zeros((nt, 4))
        t = np.zeros(nt)
        w[0, :] = w0
        t[0]= t0
        for indx in np.arange(nt - 1) + 1:
            t[indx] = t[indx - 1] + dt
            self.ode.integrate(self.ode.t + dt)
            w[indx, :] = self.ode.y
        return(t, w)
    
cannonball = Cannonball()
w0 = np.array([0.,0.,np.sqrt(3.)*0.5*v0/9.81, 0.5*v0/9.81])
(t, w) = cannonball.integrate(w0=w0, nt=500)
w=9.81*w
#error1 = w[:, 0] - np.cos(t)
'''
plt.plot(t, w[:,0])
#plt.plot(t, np.cos(t))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
plt.clf()
plt.plot(t, w[:,1])
#plt.plot(t, np.cos(t))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
plt.clf()
plt.plot(t, w[:,2])
#plt.plot(t, np.cos(t))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
plt.clf()
plt.plot(t, w[:,3])
#plt.plot(t, np.cos(t))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
plt.clf()
'''
#plot until it hits the ground
x=[]
y=[]
i=0
while w[i,1]>=0:
    x.append(w[i,0])
    y.append(w[i,1])
    i+=1
X=w[i,0]
print(X)
plt.plot(x, y)
#plt.plot(t, np.cos(t))
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.savefig('projectile.png')
plt.clf()


m=np.arange(1,6)
XX=[]
for j in m:
    cannonball = Cannonball(a=A/j)
    w0 = np.array([0.,0.,np.sqrt(3.)*0.5*v0/9.81, 0.5*v0/9.81])
    (t, w) = cannonball.integrate(w0=w0, nt=500)
    w=9.81*w
    x=[]
    y=[]
    i=0
    while w[i,1]>=0:
        x.append(w[i,0])
        y.append(w[i,1])
        i+=1
    XX.append(w[i,0])
    plt.plot(x, y, label=f'm={j} kg')
    #plt.plot(t, np.cos(t))
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend()
plt.savefig('multiple-projectile.png')
plt.clf()

plt.plot(m,XX)
plt.xlabel('$m$ [kg]')
plt.ylabel('landing spot [m]')
plt.savefig('landing-projectile.png')
plt.clf()