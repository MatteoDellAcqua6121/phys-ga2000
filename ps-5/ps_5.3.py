import numpy as np
import matplotlib.pyplot as plt

error=2.0

#import the data
data = np.genfromtxt('signal.dat',
                     skip_header=1,
                     skip_footer=0,
                     dtype=np.float32,
                     delimiter='|')
#print(data)

#for some reason, only the second and third colums are the actual data, the others are filled with NaNs
x=np.array(data[:,1], dtype=np.float32)
y=np.array(data[:,2], dtype=np.float32)

#print((np.max(x)-np.min(x))/7.5)

#rescale the independent variable
x=(x-np.mean(x))/np.std(x)

#plot the data
plt.figure(figsize=(8, 6))
plt.plot(x,y,'o' )
plt.title('Signal')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
#plt.show()
plt.savefig('Signal.png')

#chose the degree of the polnomial
N=4

#implement the SVD
A = np.zeros((len(x), N))
for i in np.arange(N):
    if i==0:
        A[:, i] = 1.
    else:
        A[:, i] = x**i

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
weff=np.where(w!=0, w, np.inf)
ainv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
b = ainv.dot(y)
bm=A.dot(b)
#print the condition number
c=np.max(w)/np.min(w)
print(c)

#plot polynomial fit and its error
plt.figure(figsize=(8, 6))
plt.plot(x,y,'o',label='data')
plt.plot(x,bm, 'o', label='fit')
plt.title('SVD-fit')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
#plt.show()
plt.legend()
plt.savefig('SVD-fit.png')

plt.figure(figsize=(8, 6))
plt.plot(x,bm-y , 'o')
plt.title('Error')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
#plt.show()
plt.savefig('Error.png')

#setup and implement the sinusoidal fit
t=np.amax(x)
M=20

A = np.zeros((len(x), 2*M+1))
for i in np.arange(2*M+1):
    if i==0:
        A[:, i] = 1.
    elif i<=M:
        A[:, i] = np.sin(i*np.pi*x/t)
    else:
        A[:, i] = np.cos((i-M)*np.pi*x/t)

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
weff=np.where(w!=0, w, np.inf)
ainv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
b = ainv.dot(y)
bm=A.dot(b)
#print the condition number
c=np.max(w)/np.min(w)
print(c)
#print(b)

#plot sinusoidal fit and its errors
plt.figure(figsize=(8, 6))
plt.plot(x,y,'o',label='data')
plt.plot(x,bm, 'o', label='fit')
plt.title('SVD-fit')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
#plt.show()
plt.legend()
plt.savefig('SVD-fit-sin.png')

plt.figure(figsize=(8, 6))
plt.plot(x,bm-y,'o' )
plt.title('Error')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
#plt.show()
plt.savefig('Error-sin.png')



