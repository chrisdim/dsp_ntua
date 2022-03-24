# Digital Signal Processing - Lab 1 - Part 2
# Dimitris Dimos - 03117165
# Christos Dimopoulos - 03117037

import numpy as np
from numpy import random
import librosa
import matplotlib.pyplot as plt
import pywt
import sounddevice as sd
from scipy import signal as sg
import math

plt.close('all')

# 2.1

# QUESTION (a)

# sampling parameters
Fs = 1000  
Ts = 1/Fs  
start = 0
end = 2
samples = int (end/Ts)

# AWGN
v = random.normal(loc = 0.0, scale = 1.0, size = int (samples)) 

# noisy sampled signal
n = np.arange(start, end, Ts) / Ts
x = 2 * np.cos(2*np.pi*70*Ts*n) + 3 * np.sin(2*np.pi*140*Ts*n) + 0.15 * v

# sounds like
sd.play(x, 10000)

# looks like
plt.figure(1,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.plot(n, x, label = 'x[n] = 2cos(2π0.07n) + 3sin(2π0.14n) + 0.15v[0.001n]')
plt.xlabel('Time [n]')
plt.ylabel('x[n]')
plt.title('Sampled noisy signal of Question 2.1')
plt.legend()
plt.show()


# QUESTION (b)

# STFT Parameters
window = 0.04
overlap = 0.02
one = int (window / Ts)
two = int (overlap / Ts)

# STFT creation
G = librosa.stft(x, n_fft = one, hop_length = two)
print(G.shape)

# plot STFT
t = np.linspace(0, end, 101)
f = np.linspace(0, Fs/2, 21)
plt.figure(2,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.pcolormesh(t,f,np.abs(G))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of the noisy signal')


# QUESTION (c)

# Continuous Wavelet Transform
s = np.power(2, np.linspace(1, math.log(Fs/15.625, 2), 20*5))
coefs,freqs = pywt.cwt(x, s, 'cmor3.0-1.0')
print(coefs.shape)

# Plot CWT - frequency
t = np.linspace(0, samples/Fs, samples)
f = freqs * Fs
plt.figure(3,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.pcolormesh(t,f,np.abs(coefs))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('CWT of the signal segment')

# Plot CWT - scale
t = np.linspace(0, samples/Fs, samples)
plt.figure(4,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.pcolormesh(t, s, np.abs(coefs))
plt.xlabel('Time (sec)')
plt.ylabel('Scales')
plt.title('CWT of the signal segment')


# 2.2

# Parameters

Fs = 1000
Ts = 1/Fs
start = 0
end = 2
samples = int (end/Ts)

# QUESTION (a)

# AWGN
v = random.normal(loc = 0.0, scale = 1.0, size = int (samples))  # AWGN

# Diracs
burst1 = sg.unit_impulse(samples, 625)
burst2 = sg.unit_impulse(samples, 800)

# final signal
n = np.arange(start, end, Ts) / Ts
x = 1.7 * (burst1 + burst2) + 0.15*v + 1.7*np.cos(2*np.pi*90*Ts*n)

# which sounds like
sd.play(x, 5000)

# and looks like
plt.figure(5,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.plot(n, x, label =
         'x[n] = 1.7cos(2π0.09n)+0.15v(0.001n)+1.7[δ(0.001n − 0.625)+δ(0.001n − 0.800)]')
plt.xlabel('Time [n]')
plt.ylabel('x[n]')
plt.title('Sampled noisy signal of Question 2.2')
plt.legend()
plt.show()


# QUESTION (b)

# STFT Parameters

window = 0.04
overlap = 0.02
one = int (window / Ts)
two = int (overlap / Ts)

# STFT of noisy signal
G = librosa.stft(x, n_fft = one, hop_length = two)
print(G.shape)

# plot STFT
t = np.linspace(0, end, 101)
f = np.linspace(0, Fs/2, 21)
plt.figure(6,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.contour(t, f, np.abs(G), 16)
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('STFT of the noisy signal')


# QUESTION (c)

# CWT

s = np.power(2, np.linspace(1, math.log(Fs/15.625, 2), 20*5))
coefs,freqs = pywt.cwt(x, s, 'cmor3.0-1.0')
print(coefs.shape)

# plot in relation to frequency
t = np.linspace(0, samples/Fs, samples)
f = freqs * Fs
plt.figure(7,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.contour(t, f, np.abs(coefs), 16)
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('CWT of the signal segment')

# plot in relation to scale

t = np.linspace(0, samples/Fs, samples)
plt.figure(8,
           dpi = 200,
           figsize = (15.0, 7.0))
plt.contour(t, s, np.abs(coefs), 16)
plt.xlabel('Time (sec)')
plt.ylabel('Scales')
plt.title('CWT of the signal segment')
