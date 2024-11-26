import numpy as np
import matplotlib.pyplot as plt
import scipy

#4th order RK taken from class notebook
class Harmonic(object):
    """Class to calculate the (An)Harmonic oscillators
    
    Parameters
    ----------
    
    omega : np.float32
        frequencies

    diff : np.float32
        amount for finite difference derivative for potential
    
    Set up to use RK4/5 by default
"""
    def __init__(self, omega=1., diff=1.e-7):
        self.omega = omega
        self.diff = diff
        self.diff = np.array([self.diff])
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
    def gradient(self, x=None):
        """Returns gradient
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            position
        
        Returns
        -------
        
        grad : ndarray of np.float32
           gradient of potential
"""
        g = self._diff_correct(func=self.potential, x=x, dx=self.diff)
        return g
    def potential(self, x=None):
        """Returns potential
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            [3]-d position
        
        Returns
        -------
        
        phi : np.float32
            potential
"""
        return 0.5*self.omega**2*x**2
    def dwdt(self, t, w):
        """Phase space time derivative
        
        Parameters
        ----------
        
        t : np.float32
            current time
            
        w : ndarray of np.float32
            [2] phase space coords (position and velocity)
        
        Returns
        -------
        
        dwdt : ndarray of np.float32
            [2] time derivatives to integrate
"""
        x = w[:1]
        v = w[1:]
        dwdt = np.zeros(2)
        dwdt[:1] = v
        dwdt[1:] = - self.gradient(x)
        return(dwdt)
    
    def integrate(self, w0=None, t0=0., dt=0.1, nt=100):
        """Integrate the equations
        
        Parameters
        ----------
        
        t0 : np.float32
            initial time
            
        w0 : ndarray of np.float32
            [2] initial phase space coords (position and velocity)
            
        dt : np.float32
            time interval to integrate per output
            
        nt : np.int32
            number of intervals
"""
        self.ode.set_initial_value(w0, t0)
        w = np.zeros((nt, 2))
        t = np.zeros(nt)
        w[0, :] = w0
        t[0]= t0
        for indx in np.arange(nt - 1) + 1:
            t[indx] = t[indx - 1] + dt
            self.ode.integrate(self.ode.t + dt)
            w[indx, :] = self.ode.y
        return(t, w)
    
class Anharmonic(object):
    """Class to calculate the (An)Harmonic oscillators
    
    Parameters
    ----------
    
    omega : np.float32
        frequencies

    diff : np.float32
        amount for finite difference derivative for potential
    
    Set up to use RK4/5 by default
"""
    def __init__(self, omega=1., diff=1.e-7):
        self.omega = omega
        self.diff = diff
        self.diff = np.array([self.diff])
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
    def gradient(self, x=None):
        """Returns gradient
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            position
        
        Returns
        -------
        
        grad : ndarray of np.float32
           gradient of potential
"""
        g = self._diff_correct(func=self.potential, x=x, dx=self.diff)
        return g
    def potential(self, x=None):
        """Returns potential
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            [3]-d position
        
        Returns
        -------
        
        phi : np.float32
            potential
"""
        return 0.25*self.omega**2*x**4
    def dwdt(self, t, w):
        """Phase space time derivative
        
        Parameters
        ----------
        
        t : np.float32
            current time
            
        w : ndarray of np.float32
            [2] phase space coords (position and velocity)
        
        Returns
        -------
        
        dwdt : ndarray of np.float32
            [2] time derivatives to integrate
"""
        x = w[:1]
        v = w[1:]
        dwdt = np.zeros(2)
        dwdt[:1] = v
        dwdt[1:] = - self.gradient(x)
        return(dwdt)
    
    def integrate(self, w0=None, t0=0., dt=0.1, nt=100):
        """Integrate the equations
        
        Parameters
        ----------
        
        t0 : np.float32
            initial time
            
        w0 : ndarray of np.float32
            [2] initial phase space coords (position and velocity)
            
        dt : np.float32
            time interval to integrate per output
            
        nt : np.int32
            number of intervals
"""
        self.ode.set_initial_value(w0, t0)
        w = np.zeros((nt, 2))
        t = np.zeros(nt)
        w[0, :] = w0
        t[0]= t0
        for indx in np.arange(nt - 1) + 1:
            t[indx] = t[indx - 1] + dt
            self.ode.integrate(self.ode.t + dt)
            w[indx, :] = self.ode.y
        return(t, w)
    
class VanderPol(object):
    """Class to calculate the (An)Harmonic oscillators
    
    Parameters
    ----------
    
    omega : np.float32
        frequencies

    mu : np.float32
        viscosity

    diff : np.float32
        amount for finite difference derivative for potential
    
    Set up to use RK4/5 by default
"""
    def __init__(self, omega=1.,mu=1.,diff=1.e-7):
        self.omega = omega
        self.mu=mu
        self.diff = diff
        self.diff = np.array([self.diff])
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
    def gradient(self, x=None):
        """Returns gradient
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            position
        
        Returns
        -------
        
        grad : ndarray of np.float32
           gradient of potential
"""
        g = self._diff_correct(func=self.potential, x=x, dx=self.diff)
        return g
    def potential(self, x=None):
        """Returns potential
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            [3]-d position
        
        Returns
        -------
        
        phi : np.float32
            potential
"""
        return 0.25*self.omega**2*x**4
    def dwdt(self, t, w):
        """Phase space time derivative
        
        Parameters
        ----------
        
        t : np.float32
            current time
            
        w : ndarray of np.float32
            [2] phase space coords (position and velocity)
        
        Returns
        -------
        
        dwdt : ndarray of np.float32
            [2] time derivatives to integrate
"""
        x = w[:1]
        v = w[1:]
        dwdt = np.zeros(2)
        dwdt[:1] = v
        dwdt[1:] = - self.omega**2*x+self.mu*(1-x**2)*v
        return(dwdt)
    
    def integrate(self, w0=None, t0=0., dt=0.01, nt=1000):
        """Integrate the equations
        
        Parameters
        ----------
        
        t0 : np.float32
            initial time
            
        w0 : ndarray of np.float32
            [2] initial phase space coords (position and velocity)
            
        dt : np.float32
            time interval to integrate per output
            
        nt : np.int32
            number of intervals
"""
        self.ode.set_initial_value(w0, t0)
        w = np.zeros((nt, 2))
        t = np.zeros(nt)
        w[0, :] = w0
        t[0]= t0
        for indx in np.arange(nt - 1) + 1:
            t[indx] = t[indx - 1] + dt
            self.ode.integrate(self.ode.t + dt)
            w[indx, :] = self.ode.y
        return(t, w)

#plots
harmonic = Harmonic()
w0 = np.array([1., 0.])
(t, w) = harmonic.integrate(w0=w0, nt=500)
error1 = w[:, 0] - np.cos(t)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w[:, 0])
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(w[:,0], w[:, 1])
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
plt.tight_layout()
plt.savefig('harmonic.png')
plt.clf()

w0 = np.array([1., 0.])
(t, w) = harmonic.integrate(w0=w0, nt=500)
error1 = w[:, 0] - np.cos(t)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w[:, 0], label='x0=1')
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(w[:,0], w[:, 1], label='x0=1')
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
fft=np.absolute(scipy.fft.rfft(w[:, 0]))
f0=np.float32(np.argmax(fft)*10./500.)
w0 = np.array([2., 0.])
(t, w) = harmonic.integrate(w0=w0, nt=500)
error12 = w[:, 0] - 2*np.cos(t)
ax1.plot(t, w[:, 0], label='x0=2')
#plt.plot(t, np.cos(t))
ax2.plot(w[:,0], w[:, 1], label='x0=2')
plt.tight_layout()
plt.legend()
plt.savefig('harmonic-amplitude.png')
plt.clf()
fft=np.absolute(scipy.fft.rfft(w[:, 0]))
f1=np.float32(np.argmax(fft)*10./500.)
print(f0-f1)

anharmonic = Anharmonic()
w0 = np.array([1., 0.])
(t, w) = anharmonic.integrate(w0=w0, nt=500)
error1 = w[:, 0] - np.cos(t)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w[:, 0])
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(t, w[:, 1])
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
plt.tight_layout()
plt.savefig('anharmonic.png')
plt.clf()

w0 = np.array([1., 0.])
(t, w) = anharmonic.integrate(w0=w0, nt=500)
error1 = w[:, 0] - np.cos(t)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w[:, 0],label='x0=1')
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(w[:,0], w[:, 1],label='x0=1')
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
w0 = np.array([2., 0.])
(t, w) = anharmonic.integrate(w0=w0, nt=500)
error12 = w[:, 0] - 2*np.cos(t)
ax1.plot(t, w[:, 0],label='x0=2')
#plt.plot(t, np.cos(t))
ax2.plot(w[:,0], w[:, 1],label='x0=2')
plt.tight_layout()
plt.legend()
plt.savefig('anharmonic-amplitude.png')
plt.clf()


vanderpol = VanderPol()
w0 = np.array([1., 0.])
(t, w) = vanderpol.integrate(w0=w0, nt=2000)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w[:, 0])
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(w[:,0], w[:, 1])
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
plt.tight_layout()
plt.savefig('vanderpol.png')
plt.clf()


vanderpol = VanderPol(mu=2)
w0 = np.array([1., 0.])
(t, w1) = vanderpol.integrate(w0=w0, nt=2000)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w1[:, 0])
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(w1[:,0], w1[:, 1])
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
plt.tight_layout()
plt.savefig('vanderpol2.png')
plt.clf()


vanderpol = VanderPol(mu=4)
w0 = np.array([1., 0.])
(t, w2) = vanderpol.integrate(w0=w0, nt=2000)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, w2[:, 0])
#plt.plot(t, np.cos(t))
ax1.set_xlabel('time [au]')
ax1.set_ylabel('$x$ [au]')
ax2.plot(w2[:,0], w2[:, 1])
ax2.set_xlabel('$x$ [au]')
ax2.set_ylabel('$v$ [au]')
plt.tight_layout()
plt.savefig('vanderpol4.png')
plt.clf()


plt.plot(w[:,0], w[:, 1], label=r'$\mu$=1')
plt.plot(w1[:,0], w1[:, 1], label=r'$\mu$=2')
plt.plot(w2[:,0], w2[:, 1], label=r'$\mu$=4')
plt.tight_layout()
plt.legend()
plt.savefig('vanderpol-compare.png')
plt.clf()