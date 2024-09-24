import numpy as np
import matplotlib.pyplot as plt
import timeit

#define teh naive multiplication
def mult(A,B):
    """For two 2D NumPy arrays, return multiplcation
    
    Inputs:
    ------
    A,B : 2D NumPy arrays
        arrays to multiply (in order)
        
    Returns:
    -------
    C : NumPy array
       matrix multiplication of imputa
    """
    N1=A.shape[0]
    N2=A.shape[1]
    if N2!=B.shape[0]:
        return('Error: wrong dimensions!')
    else:
        N3=B.shape[1]
        C=np.zeros([N1,N3], dtype=np.float32)

        for i in np.arange(N1):
            for j in np.arange(N3):
                for k in np.arange(N2):
                    C[i][k]+=A[i][k]*B[k][j]

        return(C)

samples_for=30 #number of test runned for the naive
samples_dot=300 #number of tests runned for the dot
increments=3   #increments of steps

#defines arrays which are going to store the data of time vs size (which is what we are going to plot)
t_for=np.zeros(samples_for, dtype=np.float32)
t_dot=np.zeros(samples_dot, dtype=np.float32)
size_for=np.zeros(samples_for, dtype=np.float32)
size_dot=np.zeros(samples_dot, dtype=np.float32)

#run the functions for random matrices (whith entries in [-1,1])
for i in np.arange(samples_for):
    N=(i+1)*increments
    A=2*np.random.random((N,N))-1
    B=2*np.random.random((N,N))-1
    #print(mult(A,B))
    size_for[i]=N
    t_for[i]=timeit.timeit(lambda:mult(A,B), globals=globals(), number=1)
    
for i in np.arange(samples_dot):
    N=(i+1)*increments
    A=2*np.random.random((N,N))-1
    B=2*np.random.random((N,N))-1
    #print(mult(A,B))
    size_dot[i]=N
    t_dot[i]=timeit.timeit(lambda:np.dot(A,B), globals=globals(), number=1)

# linear plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(size_for, t_for)
ax1.set_title('for')
ax1.set_xlabel('N')
ax1.set_ylabel('t [s]')

ax2.plot(size_dot, t_dot)
ax2.set_title('dots')
ax2.set_xlabel('N')
ax2.set_ylabel('t [s]')

plt.tight_layout()
plt.show()
plt.savefig('matrix_mult_linear.png')

#create linear graph attemping to match the scaling factor
log_fit_for=3*np.log(size_for)+np.ones(samples_for, dtype=np.float32)*(np.log(t_for)[3]-3*np.log(size_for)[3])
log_fit_dot=2.8*np.log(size_dot)+np.ones(samples_dot, dtype=np.float32)*(np.log(t_dot)[100]-2.8*np.log(size_dot)[100])

#log plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(np.log(size_for), np.log(t_for), label='data')
ax1.plot(np.log(size_for), log_fit_for , label=r"$\alpha$=3")
ax1.set_title('for')
ax1.set_xlabel('log(N)')
ax1.set_ylabel('log(t [s])')
ax1.legend()

# Second subplot
ax2.plot(np.log(size_dot), np.log(t_dot), label='data')
ax2.plot(np.log(size_dot), log_fit_dot, label=r"$\alpha$=2.8")
ax2.set_title('dots')
ax2.set_xlabel('log(N)')
ax2.set_ylabel('log(t [s])')
ax2.legend()

plt.tight_layout()
plt.show()
plt.savefig('matrix_mult_log.png')