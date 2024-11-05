import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#define global quantities
sample_rate=44100 #Hz

#import piano data
data = np.genfromtxt('piano.txt',
                     skip_header=0,
                     skip_footer=0,
                     dtype=np.float32)

#plot it
time=np.linspace(0,len(data)/sample_rate,len(data))
plt.plot(time,data, label='data')
plt.xlabel('time [s]')
plt.ylabel('y [au]')
plt.title('Piano waveform')
plt.legend()
plt.tight_layout()
plt.savefig('piano.png')

#discrete transform
fft=np.absolute(sp.fft.rfft(data))
f=np.linspace(0,sample_rate/2, len(fft))
print(np.argmax(fft)*sample_rate/len(data))

#plot the transform
plt.clf()
plt.plot(f,fft, label='fft')
plt.yscale('log')
plt.xlabel('f [Hz]')
plt.ylabel('y [au]')
plt.title('Piano FFT')
plt.legend()
plt.tight_layout()
plt.savefig('piano_fft.png')

#do the same for trumpet
data = np.genfromtxt('trumpet.txt',
                     skip_header=0,
                     skip_footer=0,
                     dtype=np.float32)

plt.clf()
plt.plot(time,data, label='data')
plt.xlabel('time [s]')
plt.ylabel('y [au]')
plt.title('Trunpet waveform')
plt.legend()
plt.tight_layout()
plt.savefig('trumpet.png')

fft=np.absolute(sp.fft.rfft(data))
print(np.argmax(fft)*sample_rate/len(data))

plt.clf()
plt.plot(f,fft, label='fft')
plt.yscale('log')
plt.xlabel('f [Hz]')
plt.ylabel('y [au]')
plt.title('Trumpet FFT')
plt.legend()
plt.tight_layout()
plt.savefig('trumpet_fft.png')