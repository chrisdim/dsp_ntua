# Digital Signal Processing - Lab 2 - Part 3
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165
# PCA & k-Means Algorithm

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal

plt.close('all')
counter =0

# Part 3
# USE THIS .py FILE TO CREATE mxd MATRIX 
# MIGHT TAKE A LOT OF TIME (one file per min) :) 
# mxd MATRIX IS SAVED IN FILE data.npy

#3.1 Creation of mxd matrix
result = np.zeros((40,56))
signals1 = np.zeros((40,6,12000)) #for acc-xyz and gyr-xyz
signals2 = np.zeros((40,3000)) #for hrm

#Load files.npz
stepfiles = ['step_00.npz','step_01.npz','step_02.npz','step_03.npz','step_04.npz',
             'step_05.npz','step_06.npz','step_07.npz','step_08.npz','step_09.npz',
             'step_10.npz','step_11.npz','step_12.npz','step_13.npz','step_14.npz',
             'step_15.npz','step_16.npz','step_17.npz','step_18.npz','step_19.npz',]

sleepfiles = ['sleep_00.npz','sleep_01.npz','sleep_02.npz','sleep_03.npz','sleep_04.npz',
             'sleep_05.npz','sleep_06.npz','sleep_07.npz','sleep_08.npz','sleep_09.npz',
             'sleep_10.npz','sleep_11.npz','sleep_12.npz','sleep_13.npz','sleep_14.npz',
             'sleep_15.npz','sleep_16.npz','sleep_17.npz','sleep_18.npz','sleep_19.npz',]


for i in range(20):
    data = np.load(stepfiles[i])
    acc = data['acc']
    gyr = data['gyr']
    hrm = data['hrm']
    data.close()
    signals1[i,0,:np.size(acc[:,0])] = acc[:,0]
    signals1[i,1,:np.size(acc[:,1])] = acc[:,1]
    signals1[i,2,:np.size(acc[:,2])] = acc[:,2]
    signals1[i,3,:np.size(gyr[:,0])] = gyr[:,0]
    signals1[i,4,:np.size(gyr[:,1])] = gyr[:,1]
    signals1[i,5,:np.size(gyr[:,2])] = gyr[:,2]
    signals2[i,:np.size(hrm)] = hrm
    
    data = np.load(sleepfiles[i])
    acc = data['acc']
    gyr = data['gyr']
    hrm = data['hrm']
    data.close()
    signals1[i+20,0,:np.size(acc[:,0])] = acc[:,0]
    signals1[i+20,1,:np.size(acc[:,1])] = acc[:,1]
    signals1[i+20,2,:np.size(acc[:,2])] = acc[:,2]
    signals1[i+20,3,:np.size(gyr[:,0])] = gyr[:,0]
    signals1[i+20,4,:np.size(gyr[:,1])] = gyr[:,1]
    signals1[i+20,5,:np.size(gyr[:,2])] = gyr[:,2]
    signals2[i+20,:np.size(hrm)] = hrm

# Teager-Kaiser Operator
def teo(x):
    y = np.zeros(np.size(x))
    for i in range(1,np.size(x)-1):
        y[i] = (x[i])**2 - (x[i-1])*x[i+1]
    y[0] = y[1]
    y[np.size(x)-1] = y[np.size(x)-2]
    return y

# Filter Bank
def gaborfilt(x, fc, a, fs):
    b = a/fs
    N = (3/b)+1
    n = np.arange(-N,N+1,1)
    h = np.exp(-(b**2)*(n**2))*np.cos((2*np.pi*fc/fs)*n)
    out = signal.convolve(x,h, mode="same")
    return out

#Smoothing Binomial Lowpass Filter
def smooth(x):
    h = np.array([0.25, 0.5, 0.25])
    return signal.lfilter(h,1,x)

def mean_energy(x):
    E = 0
    counter = 0
    for i in range(np.size(x)):
        if (x[i]>0):   #only positive values
            E = E + x[i]
            counter = counter+1
    return E/counter

def teager_energy(sig,fs):
    K = 25 #nuber of filters
    a = fs/(2*K)
    fcmin = a/2
    fcmax = (fs -a)/2
    step = (fcmax-fcmin)/K
    fc = np.arange(fcmin,fcmax, step) #linear index of fc for 25 filters

    #Windowing input signal with Hamming Window
    twin = 20 #sec
    tshift = 5 #sec
    winlen = twin*fs    #400 samples / 100 samples for hrm
    winshift = tshift*fs #100 samples / 25 samples for hrm
    window_hamming = sp.signal.get_window("hamming", winlen, 'true')
       
    total = np.size(sig)//(winshift)-4 #number of windows    
    y = np.zeros((total,winlen))
    for i in range(total):
          y[i,:] = sig[i*(winshift):i*(winshift)+winlen]*window_hamming

    # Use Gabor Lowpass Filters
    z = np.zeros((np.size(fc),winlen)) # 25x400
    MTE = np.zeros(total)    
    energies = np.zeros((total,np.size(fc)))
    for j in range(total): #117 windows
        for i in range(np.size(fc)): #25 filters
            z[i,:] = (gaborfilt(y[j,:], fc[i], a, fs))
            #(a) Teager-Kaiser Operator
            z[i,:] = teo(z[i,:])
            #(b) Binomial smoothing filter
            z[i,:] = smooth(z[i,:])
            z[i,:] = smooth(z[i,:]) #twice
            #(c) Energy Calculation
            energies[j,i] = mean_energy(z[i,:])
        #(d) Mean multiband Teager energy
        MTE[j] = np.max(energies[j,:])
    return MTE

def ste(x, win):
    #Compute short-time energy.
    if isinstance(win, str):
      win = sp.signal.get_window(win, max(1, len(x) // 8))
    return sp.signal.convolve(x**2, win, mode="same")

M = 400 #Window length of 20sec --> 400 samples
window_hamming = sp.signal.get_window("hamming", int(M))
M2 = 100 #Window length of 20sec --> 100 samples
window_hamming2 = sp.signal.get_window("hamming", int(M2))

 
energies = np.zeros((40,56))
for i in range(40):
    print("fileno:",i)
    for j in range(0,48,8):
        energies[i,j]=np.mean(teager_energy(signals1[i,j//8,:],20))
        energies[i,j+1]=np.mean(ste(signals1[i,j//8,:],window_hamming))
        energies[i,j+2]=np.min(teager_energy(signals1[i,j//8,:],20))
        energies[i,j+3]=np.min(ste(signals1[i,j//8,:],window_hamming))
        energies[i,j+4]=np.max(teager_energy(signals1[i,j//8,:],20))
        energies[i,j+5]=np.max(ste(signals1[i,j//8,:],window_hamming))
        energies[i,j+6]=np.std(teager_energy(signals1[i,j//8,:],20))
        energies[i,j+7]=np.std(ste(signals1[i,j//8,:],window_hamming))
    k = 48 #the rest for the hrm signal
    energies[i,k]=np.mean(teager_energy(signals2[i,:],5))
    energies[i,k+1]=np.mean(ste(signals2[i,:],window_hamming2))
    energies[i,k+2]=np.min(teager_energy(signals2[i,:],5))
    energies[i,k+3]=np.min(ste(signals2[i,:],window_hamming2))
    energies[i,k+4]=np.max(teager_energy(signals2[i,:],5))
    energies[i,k+5]=np.max(ste(signals2[i,:],window_hamming2))
    energies[i,k+6]=np.std(teager_energy(signals2[i,:],5))
    energies[i,k+7]=np.std(ste(signals2[i,:],window_hamming2))

        
np.save("data_mxd.npy", energies)
    