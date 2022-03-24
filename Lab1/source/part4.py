# Digital Signal Processing - Lab 1 - Part 4 (BONUS)
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import librosa
import sounddevice as sd

plt.close('all')
counter = 0

# Part 4 (Bonus)

#4.1 Open .wav file of salsa music signal 1

salsa1, fs = librosa.load('salsa_excerpt1.mp3')
sd.play(salsa1, fs) #kommatara :)
Ts = 1/fs # fs = 22050Hz sampling frequency

segment = salsa1[10000:75536] #segment of 2^16=65536 samples
t = np.arange(0,np.size(segment)*Ts, Ts) #time index

counter = counter+1
plt.figure(counter)
plt.plot(t,segment, 'b', label = 'Samples L=2^16')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Segment of "salsa_excerpt1.mp3"')
plt.legend()

#4.2 Discrete Wavelet Transform
from pywt import wavedec
coeffs = wavedec(segment, 'db1', level=7)/np.sqrt(2)
ya7, yd7, yd6, yd5, yd4, yd3, yd2, yd1 = coeffs

#4.3 Envelope Detection

#(a) Absolute Value
absolutes = np.abs(coeffs)
za7 = absolutes[0]
zd7 = absolutes[1]
zd6 = absolutes[2]
zd5 = absolutes[3]
zd4 = absolutes[4]
zd3 = absolutes[5]
zd2 = absolutes[6]
zd1 = absolutes[7]

#(b) Lowpass Filter
a0 = 0.006
a = np.zeros(7)
for i in range(1,8):
    a[i-1] = a0*(2**(i+1))
    

def envelope(signal, absolute, a):
    x = np.zeros(np.size(signal))
    x[0] = a*absolute[0]
    for i in range(1,np.size(x)):
        x[i] = (1-a)*x[i-1] + a*absolute[i]
    x = x - np.mean(x)
    return x

xa7 = envelope(ya7, za7, a[6])
xd7 = envelope(yd7, zd7, a[6])
xd6 = envelope(yd6, zd6, a[5])
xd5 = envelope(yd5, zd5, a[4])
xd4 = envelope(yd4, zd4, a[3])
xd3 = envelope(yd3, zd3, a[2])
xd2 = envelope(yd2, zd2, a[1])
xd1 = envelope(yd1, zd1, a[0])

n = np.arange(0,np.size(yd3),1) #number of samples
counter=counter+1
plt.figure(counter)
plt.plot(n, yd3, 'b', label = 'Detal yd3[n]')
plt.plot(n, xd3, 'r', label = 'Envelope xd3[n]')
plt.xlabel('Samples (2^13 = 8192)')
plt.ylabel('Amplitude')
plt.title('Envelope Detection of Detail yd3')
plt.show()
plt.legend()

counter=counter+1
plt.figure(counter)
n = np.arange(0,np.size(yd6),1) #number of samples
plt.plot(n, yd6, 'b', label = 'Detail yd6[n]')
plt.plot(n, xd6, 'r', label = 'Envelope xd6[n]')
plt.xlabel('Samples (2^10 = 1024)')
plt.ylabel('Amplitude')
plt.title('Envelope Detection of Detail yd6')
plt.show()
plt.legend()

#4.4 Sum of Envelopes and Autocorrelation
nvalues = np.arange(0, 32768, 1)
n = np.arange(0, 32768, 1)
xd1 = np.interp(nvalues, n, xd1)

n = np.arange(0, 16384, 1)
xd2 = np.interp(nvalues, n, xd2)

n = np.arange(0, 8192, 1)
xd3 = np.interp(nvalues, n, xd3)

n = np.arange(0, 4096, 1)
xd4 = np.interp(nvalues, n, xd4)

n = np.arange(0, 2048, 1)
xd5 = np.interp(nvalues, n, xd5)

n = np.arange(0, 1024, 1)
xd6 = np.interp(nvalues, n, xd6)

n = np.arange(0, 512, 1)
xd7 = np.interp(nvalues, n, xd7)

n = np.arange(0, 512, 1)
xa7 = np.interp(nvalues, n, xa7)

xsum = xd1+xd2+xd3+xd4+xd5+xd6+xd7+xa7
autocorrelation = np.correlate(xsum,xsum, 'full')[len(xsum)-1:]

autocorrelation = sp.ndimage.filters.gaussian_filter1d(autocorrelation,150)

counter = counter+1
plt.figure(counter)
t = np.arange(Ts,np.size(autocorrelation)*Ts*2, 2*Ts) #time index
plt.plot(t, autocorrelation)
plt.xlabel('Time [sec]')
plt.title('Autocorrelation of Salsa Excerpt 1')



#Find the maximums of Autocorrelation
maximums = np.array(sp.signal.argrelextrema(autocorrelation, np.greater))

#Keep every two of them - Maximums of great amplitude will show as the beat
maximums = maximums[0,::2]

#Calculate number of samples between every two peaks of autocorrelation
samplesbetween = np.zeros(np.size(maximums))
for i in range(1,np.size(maximums)):
    samplesbetween[i] = maximums[i]-maximums[i-1]
samplesbetween = samplesbetween[1:(np.size(samplesbetween))]

#Find the mean number of samples between every two peaks of autocorrelation
samplebeat = np.mean(samplesbetween)
print('Salsa1: Autocorrelation peaks every %i samples.' %samplebeat)

#Convert to time
timebeat = samplebeat*2*Ts*1000 #msec
print('Salsa1: Autocorrelation peaks approximately every %d msec.' %timebeat)

#Calculate BPM os salsa1
bpm_rate = 60*(1000/(timebeat))
print('Salsa1: Beats Per Minute Rate = %d bpm.' %bpm_rate)

#Visualise BPM of salsa1 with help of plotting
counter = counter+1
plt.figure(counter)
plt.plot(60/t,autocorrelation)
plt.xlim(20, 180)
plt.xlabel('Beats Per Minute (BPM)')
plt.ylabel('Autocorrelation')
plt.title('BPM of Salsa Excerpt 1')



####################    SALSA 2         #####################

#4.1 Open .wav file of salsa music signal 2

salsa2, fs = librosa.load('salsa_excerpt2.mp3')
#sd.play(salsa2, fs)
Ts = 1/fs # fs = 22050Hz sampling frequency

segment = salsa2[60000:125536] #segment of 2^16=65536 samples
t = np.arange(0,np.size(segment)*Ts, Ts) #time index

counter = counter+1
plt.figure(counter)
plt.plot(t,segment, 'b', label = 'Samples L=2^16')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Segment of "salsa_excerpt2.mp3"')
plt.legend()

#4.2 Discrete Wavelet Transform
from pywt import wavedec
coeffs = wavedec(segment, 'db1', level=7)/np.sqrt(2)
ya7, yd7, yd6, yd5, yd4, yd3, yd2, yd1 = coeffs

#4.3 Envelope Detection

#(a) Absolute Value
absolutes = np.abs(coeffs)
za7 = absolutes[0]
zd7 = absolutes[1]
zd6 = absolutes[2]
zd5 = absolutes[3]
zd4 = absolutes[4]
zd3 = absolutes[5]
zd2 = absolutes[6]
zd1 = absolutes[7]

#(b) Lowpass Filter
a0 = 0.003
a = np.zeros(7)
for i in range(1,8):
    a[i-1] = a0*(2**(i+1))
    

def envelope(signal, absolute, a):
    x = np.zeros(np.size(signal))
    x[0] = a*absolute[0]
    for i in range(1,np.size(x)):
        x[i] = (1-a)*x[i-1] + a*absolute[i]
    x = x - np.mean(x)
    return x

xa7 = envelope(ya7, za7, a[6])
xd7 = envelope(yd7, zd7, a[6])
xd6 = envelope(yd6, zd6, a[5])
xd5 = envelope(yd5, zd5, a[4])
xd4 = envelope(yd4, zd4, a[3])
xd3 = envelope(yd3, zd3, a[2])
xd2 = envelope(yd2, zd2, a[1])
xd1 = envelope(yd1, zd1, a[0])

n = np.arange(0,np.size(yd3),1) #number of samples
counter=counter+1
plt.figure(counter)
plt.plot(n, yd3, 'b', label = 'Detal yd3[n]')
plt.plot(n, xd3, 'r', label = 'Envelope xd3[n]')
plt.xlabel('Samples (2^13 = 8192)')
plt.ylabel('Amplitude')
plt.title('Envelope Detection of Detail yd3')
plt.show()
plt.legend()

counter=counter+1
plt.figure(counter)
n = np.arange(0,np.size(yd6),1) #number of samples
plt.plot(n, yd6, 'b', label = 'Detail yd6[n]')
plt.plot(n, xd6, 'r', label = 'Envelope xd6[n]')
plt.xlabel('Samples (2^10 = 1024)')
plt.ylabel('Amplitude')
plt.title('Envelope Detection of Detail yd6')
plt.show()
plt.legend()

#4.4 Sum of Envelopes and Autocorrelation
nvalues = np.arange(0, 32768, 1)
n = np.arange(0, 32768, 1)
xd1 = np.interp(nvalues, n, xd1)

n = np.arange(0, 16384, 1)
xd2 = np.interp(nvalues, n, xd2)

n = np.arange(0, 8192, 1)
xd3 = np.interp(nvalues, n, xd3)

n = np.arange(0, 4096, 1)
xd4 = np.interp(nvalues, n, xd4)

n = np.arange(0, 2048, 1)
xd5 = np.interp(nvalues, n, xd5)

n = np.arange(0, 1024, 1)
xd6 = np.interp(nvalues, n, xd6)

n = np.arange(0, 512, 1)
xd7 = np.interp(nvalues, n, xd7)

n = np.arange(0, 512, 1)
xa7 = np.interp(nvalues, n, xa7)

xsum = xd1+xd2+xd3+xd4+xd5+xd6+xd7+xa7
autocorrelation = np.correlate(xsum,xsum, 'full')[len(xsum)-1:]

autocorrelation = sp.ndimage.filters.gaussian_filter1d(autocorrelation,130)

counter = counter+1
plt.figure(counter)
t = np.arange(Ts,np.size(autocorrelation)*Ts*2, 2*Ts) #time index
plt.plot(t, autocorrelation)
plt.xlabel('Time [sec]')
plt.title('Autocorrelation of Salsa Excerpt 2')



#Find the maximums of Autocorrelation
maximums = np.array(sp.signal.argrelextrema(autocorrelation, np.greater))

#Keep every two of them - Maximums of great amplitude will show as the beat
maximums = maximums[0,::2]

#Calculate number of samples between every two peaks of autocorrelation
samplesbetween = np.zeros(np.size(maximums))
for i in range(1,np.size(maximums)):
    samplesbetween[i] = maximums[i]-maximums[i-1]
samplesbetween = samplesbetween[1:(np.size(samplesbetween))]

#Find the mean number of samples between every two peaks of autocorrelation
samplebeat = np.mean(samplesbetween)
print('Salsa2: Autocorrelation peaks every %i samples.' %samplebeat)

#Convert to time
timebeat = samplebeat*2*Ts*1000 #msec
print('Salsa2: Autocorrelation peaks approximately every %d msec.' %timebeat)

#Calculate BPM os salsa1
bpm_rate = 60*(1000/(timebeat))
print('Salsa2: Beats Per Minute Rate = %d bpm.' %bpm_rate)

#Visualise BPM of salsa1 with help of plotting
counter = counter+1
plt.figure(counter)
plt.plot(60/t,autocorrelation)
plt.xlim(20, 180)
plt.xlabel('Beats Per Minute (BPM)')
plt.ylabel('Autocorrelation')
plt.title('BPM of Salsa Excerpt 2')


####################    RUMBA         #####################

#4.1 Open .wav file of rumba music signal

rumba, fs = librosa.load('rumba_excerpt.mp3')
#sd.play(rumba,fs)

Ts = 1/fs # fs = 22050Hz sampling frequency

segment = rumba[350000:415536] #segment of 2^16=65536 samples
t = np.arange(0,np.size(segment)*Ts, Ts) #time index

counter = counter+1
plt.figure(counter)
plt.plot(t,segment, 'b', label = 'Samples L=2^16')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Segment of "rumba_excerpt.mp3"')
plt.legend()

#4.2 Discrete Wavelet Transform
from pywt import wavedec
coeffs = wavedec(segment, 'db1', level=7)/np.sqrt(2)
ya7, yd7, yd6, yd5, yd4, yd3, yd2, yd1 = coeffs

#4.3 Envelope Detection

#(a) Absolute Value
absolutes = np.abs(coeffs)
za7 = absolutes[0]
zd7 = absolutes[1]
zd6 = absolutes[2]
zd5 = absolutes[3]
zd4 = absolutes[4]
zd3 = absolutes[5]
zd2 = absolutes[6]
zd1 = absolutes[7]

#(b) Lowpass Filter
a0 = 0.0005
a = np.zeros(7)
for i in range(1,8):
    a[i-1] = a0*(2**(i+1))
    

def envelope(signal, absolute, a):
    x = np.zeros(np.size(signal))
    x[0] = a*absolute[0]
    for i in range(1,np.size(x)):
        x[i] = (1-a)*x[i-1] + a*absolute[i]
    x = x - np.mean(x)
    return x

xa7 = envelope(ya7, za7, a[6])
xd7 = envelope(yd7, zd7, a[6])
xd6 = envelope(yd6, zd6, a[5])
xd5 = envelope(yd5, zd5, a[4])
xd4 = envelope(yd4, zd4, a[3])
xd3 = envelope(yd3, zd3, a[2])
xd2 = envelope(yd2, zd2, a[1])
xd1 = envelope(yd1, zd1, a[0])

n = np.arange(0,np.size(yd3),1) #number of samples
counter=counter+1
plt.figure(counter)
plt.plot(n, yd3, 'b', label = 'Detal yd3[n]')
plt.plot(n, xd3, 'r', label = 'Envelope xd3[n]')
plt.xlabel('Samples (2^13 = 8192)')
plt.ylabel('Amplitude')
plt.title('Envelope Detection of Detail yd3')
plt.show()
plt.legend()

counter=counter+1
plt.figure(counter)
n = np.arange(0,np.size(yd6),1) #number of samples
plt.plot(n, yd6, 'b', label = 'Detail yd6[n]')
plt.plot(n, xd6, 'r', label = 'Envelope xd6[n]')
plt.xlabel('Samples (2^10 = 1024)')
plt.ylabel('Amplitude')
plt.title('Envelope Detection of Detail yd6')
plt.show()
plt.legend()

#4.4 Sum of Envelopes and Autocorrelation
nvalues = np.arange(0, 32768, 1)
n = np.arange(0, 32768, 1)
xd1 = np.interp(nvalues, n, xd1)

n = np.arange(0, 16384, 1)
xd2 = np.interp(nvalues, n, xd2)

n = np.arange(0, 8192, 1)
xd3 = np.interp(nvalues, n, xd3)

n = np.arange(0, 4096, 1)
xd4 = np.interp(nvalues, n, xd4)

n = np.arange(0, 2048, 1)
xd5 = np.interp(nvalues, n, xd5)

n = np.arange(0, 1024, 1)
xd6 = np.interp(nvalues, n, xd6)

n = np.arange(0, 512, 1)
xd7 = np.interp(nvalues, n, xd7)

n = np.arange(0, 512, 1)
xa7 = np.interp(nvalues, n, xa7)

xsum = xd1+xd2+xd3+xd4+xd5+xd6+xd7+xa7
autocorrelation = np.correlate(xsum,xsum, 'full')[len(xsum)-1:]

autocorrelation = sp.ndimage.filters.gaussian_filter1d(autocorrelation,250)

counter = counter+1
plt.figure(counter)
t = np.arange(Ts,np.size(autocorrelation)*Ts*2, 2*Ts) #time index
plt.plot(t, autocorrelation)
plt.xlabel('Time [sec]')
plt.title('Autocorrelation of Rumba Excerpt')



#Find the maximums of Autocorrelation
maximums = np.array(sp.signal.argrelextrema(autocorrelation, np.greater))

#Calculate number of samples between every two peaks of autocorrelation
samplesbetween = np.zeros(np.size(maximums))
for i in range(1,np.size(maximums)):
    samplesbetween[i] = maximums[0,i]-maximums[0,i-1]
samplesbetween = samplesbetween[1:(np.size(samplesbetween))]

#Find the mean number of samples between every two peaks of autocorrelation
samplebeat = np.mean(samplesbetween)
print('Rumba: Autocorrelation peaks every %i samples.' %samplebeat)

#Convert to time
timebeat = samplebeat*2*Ts*1000 #msec
print('Rumba: Autocorrelation peaks approximately every %d msec.' %timebeat)

#Calculate BPM os salsa1
bpm_rate = 60*(1000/(timebeat))
print('Rumba: Beats Per Minute Rate = %d bpm.' %bpm_rate)

#Visualise BPM of salsa1 with help of plotting
counter = counter+1
plt.figure(counter)
plt.plot(60/t,autocorrelation)
plt.xlim(20, 180)
plt.xlabel('Beats Per Minute (BPM)')
plt.ylabel('Autocorrelation')
plt.title('BPM of Rumba Excerpt')