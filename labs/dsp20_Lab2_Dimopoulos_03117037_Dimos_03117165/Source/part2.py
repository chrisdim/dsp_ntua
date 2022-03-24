# Digital Signal Processing - Lab 2 - Part 2
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165
# Teager - Kaiser Energy Operator - Gabor Filterbanks

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal

plt.close('all')
counter =0

# Part 2

#2.1 Load compressed file "step_00.npz" once again
data = np.load('step_00.npz')
#data.files
acc = data['acc']
gyr = data['gyr']
hrm = data['hrm']
data.close()
accx = acc[:,0]
gyrx = gyr[:,0]

#2.2 Defining teo() function 
def teo(x):
    y = np.zeros(np.size(x))
    for i in range(1,np.size(x)-1):
        y[i] = (x[i])**2 - (x[i-1])*x[i+1]
    y[0] = y[1]
    y[np.size(x)-1] = y[np.size(x)-2]
    return y

#2.3 Filter Bank
def gaborfilt(x, fc, a, fs):
    b = a/fs
    N = (3/b)+1
    n = np.arange(-N,N+1,1)
    h = np.exp(-(b**2)*(n**2))*np.cos((2*np.pi*fc/fs)*n)
    out = sp.signal.convolve(x,h, mode="same")
    #out = signal.lfilter(h,1,x)
    return out

times = 0
for sig in [accx, gyrx, hrm]:
    times = times+1
    if (times==3):
        fs = 5 #HRM sampling frequency
    else:
        fs = 20 # accx or gyrx signals
    inputs = ['Acceleration-X', 'Gyroscope-X', 'HRM']
    print("\nCASE ", times, ": ", inputs[times-1], 'signal.')
    Ts=1/fs
    
    K = 25 #nuber of filters
    a = fs/(2*K)
    fcmin = a/2
    fcmax = (fs -a)/2
    step = (fcmax-fcmin)/K
    fc = np.arange(fcmin,fcmax, step) #linear index of fc for 25 filters

    #Plot spectrum of signal
    counter = counter+1
    plt.figure(counter)
    DFT = np.fft.rfft(sig)
    f = np.fft.rfftfreq(np.size(sig)) * fs
    labels = ['Spectrum of Signal Acceleration X', 'Spectrum of Signal Angular Velocity X', 'Spectrum of Signal Heart Rate Variability']
    titles = ['Gabor Filterbank of 25 Filters - Acceleration X','Gabor Filterbank of 25 Filters - Gyroscope X','Gabor Filterbank of 25 Filters - HRM',]
    plt.plot(f,20*np.log10(np.abs(DFT)),'b', label=labels[times-1])
    plt.xlabel('Frequency (Hertz)') 
    plt.ylabel('Magnitude (dB)')
    plt.title(titles[times-1])
    plt.legend()
    
    #Plot spectrums of 25 Gabor filters
    b = a/fs
    N = (3/b)+1
    n = np.arange(-N,N+1,1)
    for fx in fc:
        h = np.exp(-(b**2)*(n**2))*np.cos((2*np.pi*fx/fs)*n)
        f = np.fft.rfftfreq(np.size(h)) * fs
        H = np.fft.rfft(h)
        plt.plot(f,20*np.log10(np.abs(H)), 'r')
        plt.ylim(bottom=-20)
    plt.show()
    
#2.4 Smoothing Binomial Lowpass Filter
    def smooth(x):
        h = np.array([0.25, 0.5, 0.25])
        return sp.signal.lfilter(h,[1],x)
    
#2.5 Windowing input signal with Hamming Window
    twin = 20 #sec
    tshift = 5 #sec
    
    if (times==3): #HRM
    #insert 2 elemenets to have exactly 600sec-->3000 samples
        for i in range(2):
            sig = np.insert(sig,np.size(sig),sig[np.size(sig)-1])
        
    else:
    #insert 8 elemenets to have exactly 600sec-->12000 samples
        for i in range(8):
            sig = np.insert(sig,np.size(sig),sig[np.size(sig)-1])
    
    winlen = twin*fs    #400 samples / 100 samples for hrm
    winshift = tshift*fs #100 samples / 25 samples for hrm
    window_hamming = signal.get_window("hamming", winlen)
       
    total = np.size(sig)//(winshift)-3 #number of windows
    y = np.zeros((total,winlen))
    for i in range(total):
          y[i,:] = sig[i*(winshift):i*(winshift)+winlen]*window_hamming
    
    
#2.6 Use Gabor Lowpass Filters
    z = np.zeros((np.size(fc),winlen)) # 25x400
    MTE = np.zeros(total)
    
    def mean_energy(x):
        E = 0
        amount = 0
        for i in range(np.size(x)):
            if (x[i]>0):   #only positive values
                E = E + x[i]
                amount = amount+1
        return E/amount
    
    energies = np.zeros((total,np.size(fc)))
    for j in range(total): #117 windows
        for i in range(np.size(fc)): #25 filters
            z[i,:] = (gaborfilt(y[j,:], fc[i], a, fs))
            

#2.7 Get Ready for some action:
            #(a) Teager-Kaiser Operator
            z[i,:] = teo(z[i,:])
     
            #(b) Binomial smoothing filter
            z[i,:] = smooth(z[i,:])
            z[i,:] = smooth(z[i,:]) #twice
    
            #(c) Energy Calculation
            energies[j,i] = mean_energy(z[i,:])
            
        #(d) Mean multiband Teager energy
        MTE[j] = np.max(energies[j,:])
        position = np.argmax(energies[j,:])
        print("Window No",j+1,": MTE = ",round(MTE[j],3)," for fc = ",round(fc[position],3),"Hz")
    
    def ste(x, win):
      #Compute short-time energy.
      if isinstance(win, str):
        win = sp.signal.get_window(win, max(1, len(x) // 8))
      return sp.signal.convolve(x**2, win, mode="same")
  
    #Plot the result:
    counter = counter+1
    plt.figure(counter)
    step = np.size(sig)*Ts/np.size(MTE)
    m = np.arange(0,np.size(MTE)*step, step)
        
    M = 20/Ts #Window length of 20sec --> 400 samples
    window_hamming = sp.signal.get_window("hamming", int(M))
    ste_signal = ste(sig, window_hamming)
    t = np.arange(0,np.size(sig)*Ts, Ts) #time index
    
    titles = ['Short-time Energy & Mean Teager Energy of Acceleration Signal (X-axis)', 'Short-time Energy & Mean Teager Energy of Gyroscope Signal (X-axis)','Short-time Energy & Mean Teager Energy of HRM Signal']
    plt.plot(t, 20*np.log10(np.abs(sig)), 'b', label='Original Signal')
    plt.plot(m,20*np.log10(np.abs(MTE)),'orange', label="Mean multiband Teager Energy")
    plt.plot(t, 20*np.log10(np.abs(ste_signal)), 'r', label='Short Time Energy')
    plt.xlabel('Time [sec]')
    plt.ylabel('Magnitude (dB)')
    plt.title(titles[times-1])
    plt.legend()
    plt.show()