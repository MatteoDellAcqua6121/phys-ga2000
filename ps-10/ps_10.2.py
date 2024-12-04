import numpy as np
import matplotlib.pyplot as plt
import scipy
import dcst
import matplotlib.animation as animation

#define the parameters
M=9.109             #e-31 kg
L=1                 #e-8 m
sigma=0.01          #e-8m
kappa=500               #e8 m-1
N=1000              #number of steps
a=L/N               #grid size
h=1                 #e-18s
hbar=1.054571817    #e-34 J/s
T=1000              #number of time interations (step size=h)
x0=0.5*L

#defining the normal modes
def E(k):
    return (0.5*hbar*(np.pi*k)**2/(M*L**2))*10**(-5) #adjust the unit (taking into account h!!!)

#defining the initial condition
psi0_real = np.array([np.real(np.exp(- (i/N - x0)**2 / (2. * sigma**2)) *np.exp((1.j) * i*kappa/N)) for i in np.arange(N)])
psi0_im= np.array([np.imag(np.exp(- (i/N - x0)**2 / (2. * sigma**2)) *np.exp((1.j) * i*kappa/N)) for i in np.arange(N)])

#founding its coefficients
a0=dcst.dst(psi0_real)
n0=dcst.dst(psi0_im)

#inverting the fourier
psi_real = np.zeros((T, N), dtype=np.float32)
def coeff(t):
    return np.array([a0[i]*np.cos(E(i)*t*h)+n0[i]*np.sin(E(i)*t*h) for i in np.arange(N)])
for i in np.arange(T):
    psi_real[i, :] = dcst.idst(coeff(i))

#create the animation
x=np.linspace(0,L, N)
c=1.
fig, ax = plt.subplots()
ax.set_xlim(( 0., L))
ax.set_ylim((-c,c))
ax.set_title('Spectral')
ax.set_xlabel(r'$x$ [$10^{-8}$ m]')
ax.set_ylabel(r'$\psi$ [au]')
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

def frame(i):
    line.set_data(x,psi_real[i, :])
    return (line,)

anim = animation.FuncAnimation(fig, frame, init_func=init,
                               frames=N, interval=40,
                               blit=True)

plt.show()
#anim.save(filename="Spectral.gif")

sample=np.array([0,100, 200, 300, 400, 500])

# Plotting 
fig,axes = plt.subplots(2,3,figsize=(12,8))


for i, ax in enumerate(axes.flat):
        ax.plot(x, psi_real[sample[i], :])
        
        ax.set_title(rf't = {sample[i]*h} [$10^{-18}$ s]', fontsize=15)
        ax.set_xlim(( 0., L))
        ax.set_ylim((-c,c))
        
        # Axes label
        if i == 3 or i == 4 or i == 5:
            ax.set_xlabel(r'$x$ [$10^{-8}$ m]')
        if i == 0 or i == 3:
           ax.set_ylabel(r'$\psi$ [au]')
        
        # Ticks font size
        ax.tick_params(axis='both', labelsize=15)  

# To prevent overlap
plt.tight_layout()
plt.savefig("Spectral.png")
plt.show()