# Digital Signal Processing - Lab 1 - Part 3
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sounddevice as sd

plt.close('all')
counter =0

# Part 3

#3.1: Short-Time Energy & Zero Crossing of speech signal

# Open .wav file of speech signal
from scipy.io import wavfile
fs, speech = wavfile.read('speech_utterance.wav')

Ts = 1/fs # fs = 16kHz sampling frequency
t = np.arange(0,np.size(speech)*Ts, Ts) #time index

# Plot speech signal
counter = counter+1
plt.figure(counter)
plt.plot(t,speech,'b', label = 'Sampling Fs = 16kHz')
plt.ylabel('Amplitude of Speech Signal')
plt.xlabel('Time [sec]')
plt.legend()
plt.title('‘Αλλά και για να επέμβει χρειάζεται συγκεκριμένη καταγγελία’')
plt.show()
sd.play(speech,fs)
sd.wait(6)

# Function that calculates sgn(x) = (1 if x>=0| -1 if x<0)
def sgn(x):
  y = np.zeros_like(x)
  y[np.where(x >= 0)] = 1.0
  y[np.where(x < 0)] = -1.0
  return y

# Function that calculates Short-Time Zero Crossings of signal x with window win
def stzerocr(x, win):
  #Compute short-time zero crossing rate.
  if isinstance(win, str):
    win = sp.signal.get_window(win, max(1, len(x) // 8))
  win = 0.5 * win / len(win)
  x1 = np.roll(x, 1)
  x1[0] = 0.0
  abs_diff = np.abs(sgn(x) - sgn(x1))
  return sp.signal.convolve(abs_diff, win, mode="same")

# Function that calculates Short-Time Energy of signal x with window win
def stenergy(x, win):
  #Compute short-time energy.
  if isinstance(win, str):
    win = sp.signal.get_window(win, max(1, len(x) // 8))
  win = win / len(win)
  return sp.signal.convolve(x**2, win, mode="same")


M = 0.025/Ts #Window length of 25msec --> 400 samples
window_hamming = sp.signal.get_window("hamming", int(M))
window_rectangular = sp.signal.get_window("boxcar", int(M))

speech = np.array(speech, dtype=float)

#Calculate Short-Time Energy of Speech Signal
energy = stenergy(speech,window_hamming)

# Divide with max values to normalize axis
maxspeech = max(abs(speech))
maxenergy = max(abs(energy))

counter = counter+1
plt.figure(counter)
plt.plot(t, speech/maxspeech, 'b', label='Speech Signal')
plt.plot(t, energy/maxenergy, 'r', label='Short Time Energy')
plt.xlabel('Time [sec]')
plt.ylabel('Normalised Amplitude')
plt.title('Short-Time Energy of Speech Signal with Window Length = 25msec')
plt.legend()
plt.show()

#Calculate Short-Time Zero Crossings of Speech Signal
zerocross = stzerocr(speech,window_rectangular)

# Divide with max values to normalize axis
maxspeech = max(abs(speech))
maxzerocross = max(abs(zerocross))

counter = counter+1
plt.figure(counter)
plt.plot(t, speech/maxspeech, 'b', label='Speech Signal')
plt.plot(t, zerocross/maxzerocross, 'g', label='Zero Crossings')
plt.xlabel('Time [sec]')
plt.ylabel('Normalised Amplitude')
plt.title('Short-Time Zero Crossings of Speech Signal with Window Length = 25msec')
plt.legend()
plt.show()

# Changing Window length to see how it affects energy and zero crossings
N = np.array([50, 100, 200, 400, 800, 1000])

# Short-Time Energy:
counter = counter+1
plt.figure(counter)
for i in range(np.size(N)):
    window_hamming = sp.signal.get_window("hamming", N[i])
    #Calculate Short-Time Energy of Speech Signal
    energy = stenergy(speech,window_hamming)
    plt.subplot(np.size(N), 1, i+1)
    plt.plot(t, energy, 'r', label='Window length = %i samples' %N[i])
    plt.legend()

plt.xlabel('Time [sec]')
plt.suptitle('Short-Time Energy of Speech Signal for different Window Lengths')
plt.show()
    
# Short-Time Zero Crossings:
counter = counter+1
plt.figure(counter)
for i in range(np.size(N)):
    window_rectangular = sp.signal.get_window("boxcar", N[i])
    #Calculate Short-Time Energy of Speech Signal
    zerocross = stzerocr(speech,window_rectangular)
    plt.subplot(np.size(N), 1, i+1)
    plt.plot(t, zerocross, 'g', label='Window length = %i samples' %N[i])
    plt.legend()

plt.xlabel('Time [sec]')
plt.suptitle('Short-Time Zero Crossings of Speech Signal for different Window Lengths')
plt.show()

#3.2: Short-Time Energy & Zero Crossing of music signal

# Open .wav file of music signal
fs, music = wavfile.read('music.wav')

Ts = 1/fs # fs = 44100Hz sampling frequency
t = np.arange(0,np.size(music)*Ts, Ts) #time index

# Plot music signal
counter = counter+1
plt.figure(counter)
plt.plot(t,music,'b', label = 'Sampling Fs = 44100Hz')
plt.ylabel('Amplitude of Music Signal')
plt.xlabel('Time [sec]')
plt.legend()
plt.title('Music Signal')
plt.show()
sd.play(music,fs)


M = 0.025/Ts #Window length of 25msec --> 1102.5 samples
window_hamming = sp.signal.get_window("hamming", int(M))
music = np.array(music, dtype=float)

#Calculate Short-Time Energy of Music Signal
energy = stenergy(music,window_hamming)

# Divide with max values to normalize axis
maxmusic = max(abs(music))
maxenergy = max(abs(energy))

counter = counter+1
plt.figure(counter)
plt.plot(t, music/maxmusic, 'b', label='Music Signal')
plt.plot(t, energy/maxenergy, 'r', label='Short Time Energy')
plt.xlabel('Time [sec]')
plt.ylabel('Normalised Amplitude')
plt.title('Short-Time Energy of Music Signal with Window Length = 25msec')
plt.legend()
plt.show()

#Calculate Short-Time Zero Crossings of Music Signal
zerocross = stzerocr(music,window_rectangular)

# Divide with max values to normalize axis
maxmusic = max(abs(music))
maxzerocross = max(abs(zerocross))

counter = counter+1
plt.figure(counter)
plt.plot(t, music/maxmusic, 'b', label='Music Signal')
plt.plot(t, zerocross/maxzerocross, 'g', label='Zero Crossings')
plt.xlabel('Time [sec]')
plt.ylabel('Normalised Amplitude')
plt.title('Short-Time Zero Crossings of Music Signal with Window Length = 25msec')
plt.legend()
plt.show()

# Changing Window length to see how it affects energy and zero crossings
N = np.array([50, 100, 200, 400, 800, 1000])

# Short-Time Energy:
counter = counter+1
plt.figure(counter)
for i in range(np.size(N)):
    window_hamming = sp.signal.get_window("hamming", N[i])
    #Calculate Short-Time Energy of Music Signal
    energy = stenergy(music,window_hamming)
    plt.subplot(np.size(N), 1, i+1)
    plt.plot(t, energy, 'r', label='Window length = %i samples' %N[i])
    plt.legend()

plt.xlabel('Time [sec]')
plt.suptitle('Short-Time Energy of Music Signal for different Window Lengths')
plt.show()
    
# Short-Time Zero Crossings:
counter = counter+1
plt.figure(counter)
for i in range(np.size(N)):
    window_rectangular = sp.signal.get_window("boxcar", N[i])
    #Calculate Short-Time Energy of Music Signal
    zerocross = stzerocr(music,window_rectangular)
    plt.subplot(np.size(N), 1, i+1)
    plt.plot(t, zerocross, 'g', label='Window length = %i samples' %N[i])
    plt.legend()

plt.xlabel('Time [sec]')
plt.suptitle('Short-Time Zero Crossings of Music Signal for different Window Lengths')
plt.show()