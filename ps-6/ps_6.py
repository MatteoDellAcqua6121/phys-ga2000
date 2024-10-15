import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time 

#Upload the data
hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
fluxx = hdu_list['FLUX'].data
Nw=len(logwave)
#Ng=len(flux[:,0])
Ng=500 #analyze only a smoller set of galaxies due to computational power: for the full set, de-comment previous line
flux=np.array([fluxx[xi,:] for xi in np.arange(Ng)])
wave=10**logwave/10 #linear-scale of wavelenghts for plots

#plot the first five galaxies
plt.figure(figsize=(8, 6))
for i in np.arange(5):
    plt.plot(wave, flux[i,:], label=f'{i+1}')
plt.title('Galaxies (flux vs wavelenght)')
plt.xlabel(r'Wavelength $[\lambda]=[\AA]$')
plt.ylabel(r'Flux $[10^{−17}\ \text{erg}\ s^{−1}\ \text{cm}^{−2} \AA^{−1}]$')   
plt.legend(title='Galaxy number')
plt.savefig('Galaxies.png')

#Normalize the fluxes
integral=np.zeros(Ng,dtype=np.float32)
for i in np.arange(Ng):
    for j in np.arange(Nw-1):
        integral[i]+=flux[i,j+1]*(wave[j+1]-wave[j]) #compute the rienman integral
flux_norm=np.zeros(np.shape(flux),dtype=np.float32)
flux_res=np.zeros(np.shape(flux),dtype=np.float32)
flux_mean=np.zeros(Ng,dtype=np.float32)
for i in np.arange(Ng):
    flux_norm[i,:]=flux[i,:]/integral[i]
for i in np.arange(Ng):
    flux_mean[i]=np.mean(flux_norm[i,:])
for i in np.arange(Ng):
    flux_res[i,:]=flux_norm[i,:]-flux_mean[i]

#plot the normlaized fluxes
plt.figure(figsize=(8, 6))
for i in np.arange(5):
    plt.plot(wave, flux_res[i,:],label=f'{i+1}')
plt.title('Normalized galaxies (flux vs wavelenght)')
plt.xlabel(r'Wavelength $[\lambda]=[\AA]$')
plt.ylabel(r'Normalized flux $[au]$')  
plt.legend(title='Galaxy number') 
plt.savefig('Galaxies_norm.png')


#create the correlation matrix, and diagonalize it
t0=time.time()
C=np.dot(np.transpose(flux_res),flux_res)
#print(np.shape(C))
D, V = np.linalg.eig(C)
t1=time.time()
print(f'Covariance condition number: {np.max(np.abs(D))/np.min(np.abs(D))}') #condition value
print(f'Time taken to diagonalize: {t1-t0}')#computational cost
#V=np.transpose(V)
#print(np.shape(V))

#plot the first 5 eigenvectors. ACTUNG: eigh prints eigenvector sorted according to increasing eigenvlue!
plt.figure(figsize=(8, 6))
for i in np.arange(5):
    plt.plot(wave,V[:,i], label=rf'$\lambda_{i}$={D[i]}')
plt.title('Eigenvectors')
plt.xlabel(r'Wavelength $[\lambda]=[\AA]$')
plt.ylabel(r'Normalized flux [au]')    
plt.legend()
plt.savefig('Eigenvectors.png')

#compute svd of R. ACHTUNG: the R in the text, is the transpose of my flux_res
t2=time.time()
(u, w, vt) = np.linalg.svd(flux_res, full_matrices=True)
t3=time.time()
#print(np.shape(vt), np.shape(u))
print(f'R condition number: {np.max(np.abs(w))/np.min(np.abs(w))}')#condition value
print(f'Time taken to SVD: {t3-t2}')#computational cost

#plot the first 5 eigenvectors (SVD)
plt.figure(figsize=(8, 6))
for i in np.arange(5):
    plt.plot(wave,vt[i,:], label=rf'$\lambda_{i}$={D[i]}')
plt.title('Eigenvectors-SVD')
plt.xlabel(r'Wavelength $[\lambda]=[\AA]$')
plt.ylabel(r'Normalized flux [au]')    
plt.legend()
plt.savefig('Eigenvectors-SVD.png')

#Reconstruct the signal by the first Nc principal values
Nc=20
flux_approximate=np.zeros((Ng,Nw,Nc), dtype=np.float32)
#compute the coefficient matrix
c=np.dot(flux_norm,V) #Ng Nw
#store the incresingly more accurate approximation for each of the Ng galaxies, Nw fluxes and Nc orders of approc
for i in np.arange(Nc):
    for j in np.arange(Ng):
        for k in np.arange(Nw):
            if i==0:
                flux_approximate[j][k][i]=c[j][i]*V[k][i]
            else:
                flux_approximate[j][k][i]=flux_approximate[j][k][i-1]+c[j][i]*V[k][i]
#up to this point, the flux are normalized: let's undo the normalization!
for i in np.arange(Ng):
    for j in np.arange(Nc):
        flux_approximate[i,:,j]=(flux_approximate[i,:,j]+flux_mean[i])*integral[i]

#plot c0 vs c1
plt.figure(figsize=(8, 6))
plt.plot(c[:,1],c[:,0], 'o')
plt.title(r'$c_1$ vs $c_0$')
plt.xlabel(r'$c_1$ [au]')
plt.ylabel(r'$c_0$ [au]')   
plt.savefig('c1.png')

#plot c0 vs c2
plt.figure(figsize=(8, 6))
plt.plot(c[:,2],c[:,0], 'o')
plt.title(r'$c_2$ vs $c_0$')
plt.xlabel(r'$c_2$ [au]')
plt.ylabel(r'$c_0$ [au]')   
plt.savefig('c2.png')

#compute the error
rme=np.zeros((Ng,Nc), dtype=np.float32)
for i in np.arange(Nc):
    for j in np.arange(Ng):
        rme[j][i]=np.sqrt(np.mean(np.square(flux_approximate[j,:,i]-flux[j,:])))

#plot the error as a function of the order of approximation Nc, for the first five galaxies
plt.figure(figsize=(8, 6))
for i in np.arange(5):
    plt.plot(np.arange(Nc)+1, rme[i,:], label=f'{i}')
plt.title('Reconstruction Error (rms)')
plt.xlabel(r'$N_c$ [au]')
plt.ylabel(r'Flux error $[10^{−17}\ \text{erg}\ s^{−1}\ \text{cm}^{−2} \AA^{−1}]$')
plt.legend(title='Galaxy number')   
plt.savefig('Error.png')

for i in np.arange(5):
    print(f'RME-{i}(Nc=20): {rme[i,19]}')

#plot the inreasingly more precise approximation for the first galaxy and the first five orders of approx
plt.figure(figsize=(8, 6))
plt.plot(wave, flux[0,:], label='data')
for i in np.arange(5):
    plt.plot(wave, flux_approximate[0,:,i], label=rf'$N_c$={i}')
plt.title('Signal reconstruction')
plt.xlabel(r'Wavelength $[\lambda]=[\AA]$')
plt.ylabel(r'Flux $[10^{−17}\ \text{erg}\ s^{−1}\ \text{cm}^{−2} \AA^{−1}]$')   
plt.legend()
plt.savefig('Galaxies_approzimate.png')

