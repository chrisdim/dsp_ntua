# Digital Signal Processing - hwk2 - Part 2.2
# Christos Dimopoulos - 03117037

# Exercise 2.2: Linear Prediction Covariance and Autocorrelation Method vs DFT
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
plt.close('all')
counter = 0;

# Part (a): DFT Calculation
n = np.arange(0,101,1);
s = 10*np.cos(0.24*np.pi*n+0.2*np.pi) + 12*np.sin(0.26*np.pi*n - 0.8*np.pi)

# Zero-padding
N = 1024
x = np.zeros(N)
x[0:101] = s

# Plot DFT X[k]
counter = counter+1
plt.figure(counter)
fs = 1
DFT = np.fft.rfft(x)
f = np.fft.rfftfreq(np.size(x))*fs
plt.plot(f,20*np.log10(np.abs(DFT)),'b', label='N = 1024 samples')
plt.xlabel('Frequency (Hertz)') 
plt.ylabel('Magnitude (dB)')
plt.title('DFT of signal x[n] after zero-paading')
plt.legend()

# Part (b): Linear Prediction Covariance Method
p = 4
n = np.arange(-p,101,1);
s = 10*np.cos(0.24*np.pi*n+0.2*np.pi) + 12*np.sin(0.26*np.pi*n - 0.8*np.pi)

def cross_correlate(i,k,sig, N,n):
    corr = 0
    for m in range(101):
        a = sig[n+m-i]
        b = sig[n+m-k]
        corr = corr + a*b
    return corr

covariance_matrix = np.zeros((p,p))
for i in range(1,p+1):
    for j in range(1,p+1):
        covariance_matrix[i-1,j-1] = cross_correlate(i,j,s,np.size(s),p)
        
psi_matrix = np.zeros((p,1))
for i in range(1,p+1):
    psi_matrix[i-1,0] = cross_correlate(i,0,s,np.size(s),p)
    

LPC = np.matmul(np.linalg.inv(covariance_matrix),psi_matrix)
print("Covariance Method: \n", np.transpose(LPC))

# Part (c): Autocorrelation Method
n = np.arange(0,101,1);
s = 10*np.cos(0.24*np.pi*n+0.2*np.pi) + 12*np.sin(0.26*np.pi*n - 0.8*np.pi)

win_hamming = np.zeros(101)
win_hamming[:] = sp.signal.get_window("hamming", 101, 'true')
v = s*win_hamming
counter = counter+1
plt.figure(counter)
plt.stem(n,v,'b',label='w[n]: Hamming Window', use_line_collection=True)
plt.xlabel('Discrete Time n') 
plt.ylabel('Amplitude (V)')
plt.title('Windowed Signal v[n] = s[n]w[n]')
plt.legend()

def autocorrelation(k,N,sig):
    result = 0
    for m in range(p,N-k):
        result = result + sig[m]*sig[m+k]
    return result


Emin0 = autocorrelation(0,np.size(v), v)

# Level 1
k1 = -(autocorrelation(1,np.size(v),v))/Emin0
LPC1 = np.zeros(1)
LPC1[0] = -k1
Emin1 = (1-k1**2)*Emin0

def levinson(sig, prevLPC, prevEmin, level):
    temp = 0
    newLPC = np.zeros(level)
    for j in range(1,level):
        temp = temp+prevLPC[j-1]*autocorrelation(level-j,np.size(sig),sig)
    k = -(autocorrelation(level,np.size(sig),sig)-temp)/prevEmin
    newLPC[level-1] = -k
    for j in range(1,level):
        newLPC[j-1] = prevLPC[j-1]+k*prevLPC[level-j-1]
    newEmin = (1-k**2)*prevEmin
    return (newEmin,newLPC,k)


# Level 2
(Emin2,LPC2,k2) = levinson(v,LPC1,Emin1,2)
# Level 3
(Emin3,LPC3,k3) = levinson(v,LPC2,Emin2,3)
# Level 4
(Emin4,LPC4,k4) = levinson(v,LPC3,Emin3,4)

print("\nAutocorrelation Method: \n", LPC4)

# Part (d): Time for some plotting
win_hamming = np.ones(101)
win_hamming = sp.signal.get_window("hamming", 101, 'true')
x[0:101] = x[0:101]*win_hamming

LPCcov = np.ones(p+1)
LPCcov [1:] = -LPC[:,0]
predict1 = sp.signal.lfilter(LPCcov,1,x)
ws1, Acov = sp.signal.freqz(LPCcov, 1, fs = 1)

LPCaut = np.ones(p+1)
LPCaut [1:] = -LPC4
predict2 = sp.signal.lfilter(LPCaut,[1],x)
ws2, Aaut = sp.signal.freqz(LPCaut, 1, fs = 1)


counter = counter+1
plt.figure(counter)
plt.plot(f,20*np.log10(np.abs(DFT)),'b', label='Spectrum of windowed signal v[n]')
plt.plot(ws1,20*np.log10(1/np.abs(Acov)),'r', label='1/|Acov| - Covariance Method')
plt.plot(ws2,20*np.log10(np.abs(1/Aaut)),'--g', label='1/|Aaut| - Autocorrelation Method')
plt.xlabel('Frequency (Hertz)') 
plt.ylabel('Magnitude (dB)')
plt.title('Spectrum of signal s[n] and LPC Analysis Amplitude Responses')
plt.legend()
plt.show()

print("\n")
roots_cov = np.roots(LPCcov)
roots_aut = np.roots(LPCaut)
print("Zeros of Acov:\n ",roots_cov)
print("Zeros of Aaut:\n ",roots_aut)