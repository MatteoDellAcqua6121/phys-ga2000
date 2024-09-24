import numpy as np
import matplotlib.pyplot as plt

N=500#number of semples per times
M=10000 #number of ensambles (i.e. how many times I test the CLT)
bins=50 #number of bins in histogram

#define a function that extract the mean y M times (y being the mean of N iid variables destrebuted as a normal exponential)
def CLT(N,M):
    y=np.zeros(M, dtype=np.float32)
    for i in np.arange(M):
        y_temp=np.random.standard_exponential(N)
        y[i]=y_temp.sum()/N
    return(y)
    
y=CLT(N,M)
y=np.sqrt(N)*(y-1)

#create a standard gaussian
n=1000
x=np.linspace(-3,3,n)
normal=np.array([M*(np.amax(y)-np.min(y))/bins*np.exp(-(xi)**2/2)/np.sqrt(2*np.pi) for xi in x])#*np.exp(-N*(x-1)**2/2)/np.sqrt(2*np.pi/N)

#plot them agains each other
plt.figure(figsize=(8, 6))
plt.hist(y, bins=bins, color='blue', edgecolor='black')
plt.plot(x, normal)
plt.title('CLT')
plt.xlabel('N')
plt.ylabel('y')
plt.show()
plt.savefig('CLT-pdf.png')

#Create coordinates arrays
x=np.linspace(1,N+1, N)
mean=np.zeros(N, dtype=np.float64)
variance=np.zeros(N, dtype=np.float64)
skewness=np.zeros(N, dtype=np.float64)
kurtosis=np.zeros(N, dtype=np.float64)

#s=0
#t=0

#Extract a bunch of y at defferent levels of i and compute the momenta (unbiased)
for i in range(N):
    y=CLT(i+1,M)
    mean[i]=y.sum()/M
    var_temp=(y-np.ones(M, dtype=np.float32)*mean[i])**2
    v=var_temp.sum()
    variance[i]=v/(M-1)
    skewness_temp_num=(y-np.ones(M, dtype=np.float32)*mean[i])**3
    skewness[i]=skewness_temp_num.sum()/v**(3/2)*M/(M-1)/(M-2)*M**(3/2)
    kurtosis_temp=(y-np.ones(M, dtype=np.float32)*mean[i])**4
    kurtosis[i]=(M+1)*(M-1)/(M-2)/(M-3)/M*kurtosis_temp.sum()/variance[i]**2-3*(M-1)**2/(M-2)/(M-3)
    '''
    if (skewness[i]<0.02):
        s+=1
    if (i>20 and t==0 and kurtosis[i]<0.06):
        t=1
        print('N_t='+str(i))
        '''

#print(s)


#plot everything   
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8, 6))
ax1.plot(x, mean, label='for')
ax2.plot(x, variance, label='for')
ax3.plot(x, skewness, label='for')
ax4.plot(x, kurtosis, label='for')
ax1.set_title('Mean')
ax1.set_ylabel(r'$\hat{\mu}$ [au]')
ax1.set_xlabel('N')
ax2.set_title('Variance')
ax2.set_ylabel(r'$\hat{\sigma}^2$ [au]')
ax2.set_xlabel('N')
ax3.set_title('Skewness')
ax3.set_ylabel(r'G$_1$ [au]')
ax3.set_xlabel('N')
ax4.set_title('Kurtosis')
ax4.set_ylabel(r'G$_2$ [au]')
ax4.set_xlabel('N')


plt.show()
plt.savefig('CLT-momenta.png')