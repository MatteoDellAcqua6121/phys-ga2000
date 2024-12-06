import numpy as np
import matplotlib.pyplot as plt
import scipy
import banded
import matplotlib.animation as animation

#define the parameters
M=9.109             #e-31 kg
L=1                 #e-8 m
sigma=0.01          #e-8m
k=500               #e8 m-1
N=1000              #number of steps
a=L/N               #grid size
h=1                 #e-18s
hbar=1.054571817    #e-34 J/s
T=1000              #number of time interations (step size=h)
x0=0.5*L

#defining the A and B matrices
Delta=0.5*h*hbar/(M*L**2)*(N**2/100000)
a1=complex(1,Delta)
a2=complex(0,-0.5*Delta)
b1=complex(1,-Delta)
b2=complex(0,0.5*Delta)
B = scipy.sparse.diags([b2, b1, b2], offsets=[-1, 0, 1], shape=(N,N))
B=B.toarray()
A = np.zeros((3,N), dtype=np.complex64)
for i in np.arange(N):
    if i>0:
        A[0,i]=a2
    A[1,i]=a1
    if i<N-1:
        A[2,i]=a2

#defining the initial condition
psi0 = np.array([np.complex64(np.exp(- (i/N - x0)**2 / (2. * sigma**2)) *np.exp((1.j) * i*k/N)) for i in np.arange(N)])

#one-step Crank-Nicolson solver: multiply by B and invert A
def CrankNicolson_step(psi0=psi0, A=A, B=B):
    v=np.dot(B,psi0)
    psi=banded.banded(A,v, 1, 1)
    return psi

#running the actual solver
psi = np.zeros((T, N), dtype=np.complex64)
psi[0, :] = psi0
for i in np.arange(T - 1):
    psi[i + 1, :] = CrankNicolson_step(psi0=psi[i, :])

#create the animation
x=np.linspace(0,L, N)
c=1.
fig, ax = plt.subplots()
ax.set_xlim(( 0., L))
ax.set_ylim((-c,c))
ax.set_title('Crank-Nicolson')
ax.set_xlabel(r'$x$ [$10^{-8}$ m]')
ax.set_ylabel(r'$\psi$ [au]')
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

def frame(i):
    line.set_data(x, np.real(psi[i, :]))
    return (line,)

anim = animation.FuncAnimation(fig, frame, init_func=init,
                               frames=N, interval=40,
                               blit=True)

plt.show()
#anim.save(filename="CrankNicolson.gif")

sample=np.array([0,100, 200, 300, 400, 500])

# Plotting 
fig,axes = plt.subplots(2,3,figsize=(12,8))


for i, ax in enumerate(axes.flat):
        ax.plot(x, np.real(psi[sample[i], :]))
        
        ax.set_title(rf't = {sample[i]*h} [$10^{{-18}}$ s]', fontsize=15)
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
plt.savefig("CrankNicolson.png")
plt.show()