import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math

#import data
data = np.genfromtxt('dow.txt',
                     skip_header=0,
                     skip_footer=0,
                     dtype=np.float32)

#plot raw data
plt.plot(np.arange(len(data)),data, label='data')
plt.xlabel('Buisness day [day]')
plt.ylabel('Index value [$]')
plt.title('Dow Jones')
plt.legend()
plt.tight_layout()
plt.savefig('Dow.png')

#transform it (and remove the tail)
fft=sp.fft.rfft(data)
r=len(fft)
for i in np.arange(np.uint(math.ceil(r*0.9))):
    fft[r-i-1]=0

#ivert transform
approx=sp.fft.irfft(fft)

#plot the approximation
plt.clf()
plt.plot(np.arange(len(data)),data, label='data')
plt.plot(np.arange(len(data)),approx, label=r'$10\%$ appprox' )
plt.xlabel('Buisness day [day]')
plt.ylabel('Index value [$]')
plt.title('Dow Jones')
plt.legend()
plt.tight_layout()
plt.savefig('Dow_10.png')

#repeat for the further approximation
for i in np.arange(np.uint(math.ceil(r*0.98))):
    fft[r-i-1]=0

approx=sp.fft.irfft(fft)

plt.clf()
plt.plot(np.arange(len(data)),data, label='data')
plt.plot(np.arange(len(data)),approx, label=r'$2\%$ appprox' )
plt.xlabel('Buisness day [day]')
plt.ylabel('Index value [$]')
plt.title('Dow Jones')
plt.legend()
plt.tight_layout()
plt.savefig('Dow_2.png')

