import numpy as np
import matplotlib.pyplot as plt

#Set number of implementations (T), grid size (N)
T=100
N=1000

def mandelbrot_humble():
    #Set up the array needed: grid (c), mandelbrot variable (z), and convergence check (Converge)
    arr=np.linspace(-2, 2, num=N, dtype=np.float32)
    real, im=np.meshgrid(arr, arr, indexing='xy')
    c=real + im*1j
    z=np.zeros((N,N),dtype=np.complex128)
    Converge=np.zeros((N,N), dtype=np.int32)
    
    #Run the iteration loop
    for i in np.arange(T):
        z=np.where(np.absolute(z)>=2, z, z**2+c)
        
    #Evaluate the convergence
    Converge=np.where(np.absolute(z)>=2,np.int32(1.),np.int32(0.))

    #Plot everything
    fig, ax = plt.subplots()
    f = ax.pcolormesh(arr, arr, Converge)
    ax.set_title('The Mandelbrot plot')
    # set the limits of the plot to the limits of the data
    ax.axis([arr.min(), arr.max(), arr.min(), arr.max()])
    ax.set_xlabel('Re(c) [au]')
    ax.set_ylabel('Im(c) [au]')
    ax.text(.9, 1.85, str(T)+' iterations')
    ax.text(.9, 1.7, str(N)+ ' grid points' )
    #fig.colorbar(f, ax=ax)
    plt.show()
    plt.savefig('Mandelbort_BW.png')

def mandelbrot_fancy():
    #Set up the array needed: grid (c), mandelbrot variable (z), and convergence check (Converge)
    arr=np.linspace(-2, 2, num=N, dtype=np.float32)
    real, im=np.meshgrid(arr, arr, indexing='xy')
    c=real + im*1j
    z=np.zeros((N,N),dtype=np.complex128)
    Converge=np.zeros((N,N), dtype=np.int32)
    
    #Run the iteration loop, and evaluate converge
    for i in np.arange(T):
        I=np.ones((N,N), dtype=np.float32)*(T-i)#very memory inefficient, I know
        z=np.where(np.absolute(z)>=2, z, z**2+c)
        Converge=np.where(np.absolute(z)>=2,np.maximum(I,Converge), 0.1)
    
    Converge=np.where(Converge>0.1,Converge**3, 0.)

    #Plot everything
    fig, ax = plt.subplots()
    f = ax.pcolormesh(arr, arr, Converge)
    ax.set_title('The Fancy Mandelbrot plot')
    # set the limits of the plot to the limits of the data
    ax.axis([arr.min(), arr.max(), arr.min(), arr.max()])
    fig.colorbar(f, ax=ax, label='(100 - Convergence time)^3')
    ax.set_xlabel('Re(c) [au]')
    ax.set_ylabel('Im(c) [au]')
    ax.text(.8, 1.85, str(T)+' iterations')
    ax.text(.8, 1.7, str(N)+ ' grid points' )
    plt.show()
    plt.savefig('Mandelbort_Colour.png')

mandelbrot_humble()
mandelbrot_fancy()



