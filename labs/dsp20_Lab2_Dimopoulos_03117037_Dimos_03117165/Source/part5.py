# Digital Signal Processing - Lab 2 - Part 5 (BONUS)
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165
# Powers Spectrum Estimation with three periodograms

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal

plt.close('all')
counter =0

# Part 5.1
fs1=5
Ts1=1/fs1
hrm_orig = np.load("hrm_orig.npy")
moments = np.cumsum(hrm_orig)*0.001 #convert to sec
interp = sp.interpolate.interp1d(moments,hrm_orig,kind = 'cubic')

t = np.arange(moments[0],moments[-1], Ts1) #time index
hrm = interp(t)
counter=counter+1
plt.figure(counter)
plt.plot(t,hrm, label='Sampling Frequency = 5Hz')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude [msec]')
plt.title('HRM Signal after Cubic Interpolation')
plt.legend()
plt.show()


# Part 5.2
N = 1000
tsampling = np.sort(100*np.random.rand(N), kind='mergesort')
noise = 0.1*np.random.rand(N)
sinusoids = np.sin(2*np.pi*tsampling)+np.cos(2*np.pi*tsampling/3)
x = sinusoids+noise
    
interp = sp.interpolate.interp1d(tsampling,x,kind = 'linear')
fs2 = 10
Ts2 = 1/fs2
t = np.arange(tsampling[0],tsampling[-1], Ts2) #time index
xnew = interp(t)

counter=counter+1
plt.figure(counter)
plt.plot(t,xnew, label='Sampling Frequency = 10Hz')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude [V]')
plt.title('Signal x(t) = sin(2πt)+cos(2πt/3)+v(t) after Linear Interpolation')
plt.legend()
plt.show()

counter=counter+1
plt.figure(counter)
DFT = np.fft.rfft(xnew)
freqs = np.fft.rfftfreq(np.size(xnew))*fs2
plt.plot(freqs,np.abs(DFT))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [V]')
plt.title('FFT of Interpolated Signal x(t) = sin(2πt)+cos(2πt/3)+v(t)')
plt.show()

# Part 5.3
def periodogram_schuster(x):
    DFT = np.fft.fft(x)
    P = (np.abs(DFT)**2)/np.size(x)
    return P

schuster1 = periodogram_schuster(hrm)
schuster2 = periodogram_schuster(xnew)
f1 = np.fft.fftfreq(np.size(hrm),1/fs1)
f2 = np.fft.fftfreq(np.size(xnew))*fs2

counter=counter+1
plt.figure(counter)
plt.plot(np.fft.fftshift(f1),np.fft.fftshift(20*np.log10(schuster1)),'m',label='Schuster Periodogram')
plt.xlabel('Frequency [Hertz]')
plt.ylabel('Magnitude [dB]')
plt.title('Power Spectrum Estimation of HRM signal')
plt.legend()
plt.show()

counter=counter+1
plt.figure(counter)
plt.plot(np.fft.fftshift(f2),np.fft.fftshift(20*np.log10(schuster2)),'m', label='Schuster Periodogram')
plt.xlabel('Frequency [Hertz]')
plt.ylabel('Magnitude [dB]')
plt.title('Power Spectrum Estimation of Interpolated Signal x(t)')
plt.legend()
plt.show()

# Part 5.4
L = 100 #winlen
D = 50 # overlap
bartlett = signal.get_window('bartlett',L)
times =0
for sig in [hrm, xnew]:
    times = times+1
    if times==1: 
        fso = fs1
    else: 
        fso = fs2
    total = np.size(sig)//D # number of windows
    
    y = np.zeros((total,L))
    z = np.zeros((total-1,L))
    for i in range(total-1):
          y[i,:] = sig[i*(D):i*(D)+L]*bartlett
          z[i,:] = periodogram_schuster(y[i,:])
    
    welch = np.zeros(L)
    for i in range(total-1):  
        welch = welch+z[i,:]
    welch = welch/(L)
    
    counter=counter+1
    plt.figure(counter)
    f = np.fft.fftfreq(np.size(welch))*fso
    plt.plot(np.fft.fftshift(f),np.fft.fftshift(20*np.log10(welch)),'g', label='Welch Periodogram')
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Magnitude [dB]')
    titles = ['Power Spectrum Estimation of HRM Signal','Power Spectrum Estimation of Interpolated Signal x(t)']
    plt.title(titles[times-1])
    plt.legend()
    plt.show()

# 5.5
def lombscargle(times,sig,omega):
    N = np.size(sig)
    a,b=0,0
    for i in range(N):
        a = a + np.sin(2*omega*times[i])
        b = b+ np.cos(2*omega*times[i])
    taf = np.arctan(a/b)/(2*omega)
    
    A,B,C,D=0,0,0,0
    for i in range(N):
        A = A + sig[i]*np.cos(omega*(times[i]-taf))
        B = B + sig[i]*np.sin(omega*(times[i]-taf))
        C = C + np.cos(omega*(times[i]-taf))**2
        D = D + np.sin(omega*(times[i]-taf))**2
    
    Pls = ((A**2)/C+(B**2)/D)/2
    return Pls

radials1 = np.linspace(-5*np.pi,5*np.pi, 1000)
Estimation1 = lombscargle(moments,hrm_orig,radials1)
counter=counter+1
plt.figure(counter)
plt.plot(radials1,20*np.log10(Estimation1),'red', label='Lomb-Scargle Periodogram')
plt.xlabel('Radial Frequency Ω [rad/sec]')
plt.ylabel('Magnitude [dB]')
plt.title('Power Spectrum Estimation of Original HRM Signal')
plt.legend()
plt.show()

radials2 = np.linspace(-10*np.pi,10*np.pi, 1000)
Estimation2 = lombscargle(tsampling,x,radials2)
counter=counter+1
plt.figure(counter)
plt.plot(radials2,20*np.log10(Estimation2),'red', label='Lomb-Scargle Periodogram')
plt.xlabel('Radial Frequency Ω [rad/sec]')
plt.ylabel('Magnitude [dB]')
plt.title('Power Spectrum Estimation of Original Signal x(t)')
plt.legend()
plt.show()

times=0
# Compare spectograms for different values of N
for N in [100,500,5000,10000]:
    times=times+1
    tsampling = np.sort(100*np.random.rand(N), kind='mergesort')
    noise = 0.1*np.random.rand(N)
    sinusoids = np.sin(2*np.pi*tsampling)+np.cos(2*np.pi*tsampling/3)
    x = sinusoids+noise
    interp = sp.interpolate.interp1d(tsampling,x,kind = 'cubic')
    fs = 10
    Ts = 1/fs
    t = np.arange(tsampling[0],tsampling[-1], Ts) #time index
    xnew = interp(t)
    
    #Schuster
    schuster = periodogram_schuster(xnew)
    f1 = np.fft.fftfreq(np.size(xnew))*fs
    
    #Welch
    total = np.size(xnew)//D
    y = np.zeros((total,L))
    z = np.zeros((total-1,L))
    for i in range(total-1):
          y[i,:] = xnew[i*(D):i*(D)+L]*bartlett
          z[i,:] = periodogram_schuster(y[i,:])
    welch = np.zeros(L)
    for i in range(total-1):  
        welch = welch+z[i,:]
    welch = welch/(L)
    f2 = np.fft.fftfreq(np.size(welch))*fs
    
    #Lomb - Scargle
    radials = np.linspace(-10*np.pi,10*np.pi, 1000)
    lomb = lombscargle(tsampling,x,radials)
    
    counter = counter+1
    plt.figure(counter)
    titles=['Power Spectrum Estimations of Signal x(t) for N = 100','Power Spectrum Estimations of Signal x(t) for N = 500','Power Spectrum Estimations of Signal x(t) for N = 5000','Power Spectrum Estimations of Signal x(t) for N = 10000']
    plt.suptitle(titles[times-1])
    plt.subplot(311)
    plt.plot(np.fft.fftshift(f1),np.fft.fftshift(20*np.log10(schuster)),'m', label = 'Schuster Periodogram')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(np.fft.fftshift(f2),np.fft.fftshift(20*np.log10(welch)),'g', label = 'Welch Periodogram')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(radials,20*np.log10(lomb),'r', label = 'Lomb-Scargle Periodogram')
    plt.xlabel('Radial Frequency Ω [rad/sec]')
    plt.ylabel('Magnitude [dB]')
    plt.legend()
    plt.show()