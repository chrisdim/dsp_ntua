# Digital Signal Processing - hwk3 - Part 3.4
# Christos Dimopoulos - 03117037
# Periodogram Compare

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')
counter =0

def classic_periodogram(sig):
    DFT = np.fft.fft(sig)
    P = (np.abs(DFT)**2)/np.size(sig)
    return P

def energy(sig):
    E = 0
    for i in range(np.size(sig)):
        E = E + sig[i]**2
    return E/np.size(sig)

def modified_periodogram(sig, window):
    DFT = np.fft.fft(sig*window)
    U = energy(window)
    P = (np.abs(DFT)**2)/(np.size(sig)*U)
    return P    

for N in [64, 128]:
    n = np.arange(0,N,1) #time index
    # Frequencies:
    v1 = 0.2*np.pi
    v2 = 0.3*np.pi
    classic_final = np.zeros(N)
    modified_final = np.zeros(N)
    w = signal.get_window('hamming',N)
    
    
    for i in range(50):
        # Phases Random Uniform Distribution
        phase1 = 2*np.pi*np.random.rand(1)
        phase2 = 2*np.pi*np.random.rand(1)
        
        # AWGN with mean_value = 0, std = 1
        noise = np.random.normal(0,1,N) 
        
        # Define signal x[n]
        x = 0.1*np.sin(v1*n +phase1) + np.sin(v2*n+phase2)
        for k in range(np.size(x)):
            x[k]=x[k]+noise[k]

        classic = classic_periodogram(x)
        classic_final = classic_final + classic
        
        modified = modified_periodogram(x,w)
        modified_final = modified_final + modified
    
    classic_final = classic_final/50
    modified_final = modified_final/50    
        
    
    f1 = np.fft.fftfreq(np.size(x))
    counter=counter+1
    plt.figure(counter)
    plt.plot(np.fft.fftshift(f1),np.fft.fftshift(20*np.log10(classic_final)),'m',label='Classic Periodogram')
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Magnitude [dB]')
    if N == 64:
        plt.title('Power Spectrum Estimation - N = 64 samples')
    else:
        plt.title('Power Spectrum Estimation - N = 128 samples')
    plt.legend()
    plt.show()
    
    counter=counter+1
    plt.figure(counter)
    plt.plot(np.fft.fftshift(f1),np.fft.fftshift(20*np.log10(modified_final)),'g',label='Modified Periodogram')
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Magnitude [dB]')
    if N == 64:
        plt.title('Power Spectrum Estimation - N = 64 samples')
    else:
        plt.title('Power Spectrum Estimation - N = 128 samples')    
    plt.legend()
    plt.show()