# Digital Signal Processing - Lab 2 - Part 1
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165
# Short-Time Energy and STFT Calculation of Movement & Heart Signals

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa

plt.close('all')
counter =0

# Part 1

#1.1 Load compressed file "step_00.npz"
data = np.load('step_00.npz')
#data.files
acc = data['acc']
gyr = data['gyr']
hrm = data['hrm']
data.close()

# Acceleration
accx = acc[:,0]
accy = acc[:,1]
accz = acc[:,2]

fs1 = 20 #sampling freq
Ts1 = 1/fs1
t = np.arange(0,np.size(accx)*Ts1, Ts1) #time index

counter = counter+1
plt.figure(counter)
plt.suptitle('Acceleration of walking person - Duration of 10 min')
plt.subplot(311)
plt.plot(t,accx,'b', label = 'X axis')
plt.xlabel('Time [sec]')
plt.ylabel('[m/sec^2]')
plt.legend()

plt.subplot(312)
plt.plot(t,accy,'r', label = 'Y axis')
plt.xlabel('Time [sec]')
plt.ylabel('[m/sec^2]')
plt.legend()

plt.subplot(313)
plt.plot(t,accz,'g', label = 'Z axis')
plt.xlabel('Time [sec]')
plt.ylabel('[m/sec^2]')
plt.legend()

#Gyroscope
gyrx = gyr[:,0]
gyry = gyr[:,1]
gyrz = gyr[:,2]

t = np.arange(0,np.size(gyrx)*Ts1, Ts1) #time index

counter = counter+1
plt.figure(counter)
plt.suptitle('Angular Velocity of walking person - Duration of 10 min')
plt.subplot(311)
plt.plot(t,gyrx,'b', label = 'X axis')
plt.xlabel('Time [sec]')
plt.ylabel('[deg/sec]')
plt.legend()

plt.subplot(312)
plt.plot(t,gyry,'r', label = 'Y axis')
plt.xlabel('Time [sec]')
plt.ylabel('[deg/sec]')
plt.legend()

plt.subplot(313)
plt.plot(t,gyrz,'g', label = 'Z axis')
plt.xlabel('Time [sec]')
plt.ylabel('[deg/sec]')
plt.legend()

#Heart Rate Variability
fs2 = 5
Ts2 = 1/fs2
t2 = np.arange(0,np.size(hrm)*Ts2, Ts2)
counter = counter+1
plt.figure(counter)
plt.title('Heart Rate Variability of walking person - Duration of 10 min')
plt.plot(t2,hrm,'b', label = 'Sampling Freq = 5 Hz')
plt.xlabel('Time [sec]')
plt.ylabel('HRM [msec]')
plt.legend()

#1.2 Short-Time Energy function

# Function that calculates Short-Time Energy of signal x with window win
def ste(x, win):
  #Compute short-time energy.
  if isinstance(win, str):
    win = signal.get_window(win, max(1, len(x) // 8))
  return signal.convolve(x**2, win, mode="same")

M = 20/Ts1 #Window length of 20sec --> 400 samples
window_hamming = signal.get_window("hamming", int(M))

ste_accx = ste(accx, window_hamming)
ste_gyrx = ste(gyrx, window_hamming)
ste_hrm = ste(hrm, window_hamming)

counter = counter+1
plt.figure(counter)
plt.plot(t, (np.abs(accx)/np.max(accx)), 'b', label='Acceleration of X-axis')
plt.plot(t, (np.abs(ste_accx)/np.max(ste_accx)), 'r', label='Short Time Energy')
plt.xlabel('Time [sec]')
plt.ylabel('Normalized Amplitude')
plt.title('Short-Time Energy of Acceleration (X-axis) - Window Length=20sec')
plt.legend()
plt.show()

counter = counter+1
plt.figure(counter)
plt.plot(t, (np.abs(gyrx)/np.max(gyrx)), 'b', label='Angular Velocity of X-axis')
plt.plot(t, (np.abs(ste_gyrx)/np.max(ste_gyrx)), 'r', label='Short Time Energy')
plt.xlabel('Time [sec]')
plt.ylabel('Normalized Amplitude')
plt.title('Short-Time Energy of Angular Velocity (X-axis) - Window Length=20sec')
plt.legend()
plt.show()

counter = counter+1
plt.figure(counter)
plt.plot(t2, hrm/np.abs(np.max(hrm)), 'b', label='Heart Rate Variability')
plt.plot(t2, ste_hrm/np.abs(np.max(ste_hrm)), 'r', label='Short Time Energy')
plt.xlabel('Time [sec]')
plt.ylabel('Normalized Amplitude')
plt.title('Short-Time Energy of Heart Rate Variability - Window Length=20sec')
plt.legend()
plt.show()

#1.3 STFT of Accx and Heart-Rate Variability

# STFT Parameters
window = 20
overlap = 10
one = int (window / Ts1)
two = int (overlap / Ts1)

# STFT creation
G = librosa.stft(accx, n_fft = one, hop_length = two)
#print(G.shape)

# Plot STFT of Accelartion (x-axis)
t = np.linspace(0, np.size(accx)*Ts1, 60)
f = np.linspace(0, fs1/2, 201)
counter = counter+1
plt.figure(counter)
plt.pcolormesh(t,f,np.log10(np.abs(G)))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('log-STFT of Acceleration (X-axis)')

one = int (window / Ts2)
two = int (overlap / Ts2)

# STFT creation
G = librosa.stft(hrm, n_fft = one, hop_length = two)
#print(G.shape)

# Plot STFT of HRM
t = np.linspace(0, np.size(hrm)*Ts2, 60)
f = np.linspace(0, fs1/2, 51)
counter = counter+1
plt.figure(counter)
plt.pcolormesh(t,f,np.log10(np.abs(G)))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('log-STFT of Heart Rate Variability')


#1.4 Compare Sleep and Walking

#CASE 1: Walking
data = np.load('step_03.npz')
acc = data['acc']
gyr = data['gyr']
hrm = data['hrm']
data.close()

# Acceleration
accx = acc[:,0]
accy = acc[:,1]
accz = acc[:,2]
ste_accx = ste(accx, window_hamming)
ste_accy = ste(accy, window_hamming)
ste_accz = ste(accz, window_hamming)
#Gyroscope
gyrx = gyr[:,0]
gyry = gyr[:,1]
gyrz = gyr[:,2]
ste_gyrx = ste(gyrx, window_hamming)
ste_gyry = ste(gyry, window_hamming)
ste_gyrz = ste(gyrz, window_hamming)
#Heart Rate Variation
ste_hrm = ste(hrm, window_hamming)

print("CASE 1: Walking")
#Acceleration
print("[Acceleration X] Mean Value = ", np.mean(ste_accx))
print("[Acceleration Y] Mean Value = ", np.mean(ste_accy))
print("[Acceleration Z] Mean Value = ", np.mean(ste_accz))
print("")
print("[Acceleration X] Minimum Value = ", np.min(ste_accx))
print("[Acceleration Y] Minimum Value = ", np.min(ste_accy))
print("[Acceleration Z] Minimum Value = ", np.min(ste_accz))
print("")
print("[Acceleration X] Maximum Value = ", np.max(ste_accx))
print("[Acceleration Y] Maximum Value = ", np.max(ste_accy))
print("[Acceleration Z] Maximum Value = ", np.max(ste_accz))
print("")
print("[Acceleration X] Standard Deviation = ", np.std(ste_accx))
print("[Acceleration Y] Standard Deviation = ", np.std(ste_accy))
print("[Acceleration Z] Standard Deviation = ", np.std(ste_accz))

#Gyroscope
print("")
print("[Gyroscope X] Mean Value = ", np.mean(ste_gyrx))
print("[Gyroscope Y] Mean Value = ", np.mean(ste_gyry))
print("[Gyroscope Z] Mean Value = ", np.mean(ste_gyrz))
print("")
print("[Gyroscope X] Minimum Value = ", np.min(ste_gyrx))
print("[Gyroscope Y] Minimum Value = ", np.min(ste_gyry))
print("[Gyroscope Z] Minimum Value = ", np.min(ste_gyrz))
print("")
print("[Gyroscope X] Maximum Value = ", np.max(ste_gyrx))
print("[Gyroscope Y] Maximum Value = ", np.max(ste_gyry))
print("[Gyroscope Z] Maximum Value = ", np.max(ste_gyrz))
print("")
print("[Gyroscope X] Standard Deviation = ", np.std(ste_gyrx))
print("[Gyroscope Y] Standard Deviation = ", np.std(ste_gyry))
print("[Gyroscope Z] Standard Deviation = ", np.std(ste_gyrz))

#Heart Rate Variability
print("")
print("[HRM] Mean Value = ", np.mean(ste_hrm))
print("[HRM] Minimum Value = ", np.min(ste_hrm))
print("[HRM] Maximum Value = ", np.max(ste_hrm))
print("[HRM] Standard Deviation = ", np.std(ste_hrm))

#CASE 2: Sleeping (something I haven't done for a loooong time)
data = np.load('sleep_03.npz')
acc = data['acc']
gyr = data['gyr']
hrm = data['hrm']
data.close()

# Acceleration
accx = acc[:,0]
accy = acc[:,1]
accz = acc[:,2]
ste_accx = ste(accx, window_hamming)
ste_accy = ste(accy, window_hamming)
ste_accz = ste(accz, window_hamming)
#Gyroscope
gyrx = gyr[:,0]
gyry = gyr[:,1]
gyrz = gyr[:,2]
ste_gyrx = ste(gyrx, window_hamming)
ste_gyry = ste(gyry, window_hamming)
ste_gyrz = ste(gyrz, window_hamming)
#Heart Rate Variation
ste_hrm = ste(hrm, window_hamming)

print('')
print("CASE 2: Sleeping")
#Acceleration
print("[Acceleration X] Mean Value = ", np.mean(ste_accx))
print("[Acceleration Y] Mean Value = ", np.mean(ste_accy))
print("[Acceleration Z] Mean Value = ", np.mean(ste_accz))
print("")
print("[Acceleration X] Minimum Value = ", np.min(ste_accx))
print("[Acceleration Y] Minimum Value = ", np.min(ste_accy))
print("[Acceleration Z] Minimum Value = ", np.min(ste_accz))
print("")
print("[Acceleration X] Maximum Value = ", np.max(ste_accx))
print("[Acceleration Y] Maximum Value = ", np.max(ste_accy))
print("[Acceleration Z] Maximum Value = ", np.max(ste_accz))
print("")
print("[Acceleration X] Standard Deviation = ", np.std(ste_accx))
print("[Acceleration Y] Standard Deviation = ", np.std(ste_accy))
print("[Acceleration Z] Standard Deviation = ", np.std(ste_accz))

#Gyroscope
print("")
print("[Gyroscope X] Mean Value = ", np.mean(ste_gyrx))
print("[Gyroscope Y] Mean Value = ", np.mean(ste_gyry))
print("[Gyroscope Z] Mean Value = ", np.mean(ste_gyrz))
print("")
print("[Gyroscope X] Minimum Value = ", np.min(ste_gyrx))
print("[Gyroscope Y] Minimum Value = ", np.min(ste_gyry))
print("[Gyroscope Z] Minimum Value = ", np.min(ste_gyrz))
print("")
print("[Gyroscope X] Maximum Value = ", np.max(ste_gyrx))
print("[Gyroscope Y] Maximum Value = ", np.max(ste_gyry))
print("[Gyroscope Z] Maximum Value = ", np.max(ste_gyrz))
print("")
print("[Gyroscope X] Standard Deviation = ", np.std(ste_gyrx))
print("[Gyroscope Y] Standard Deviation = ", np.std(ste_gyry))
print("[Gyroscope Z] Standard Deviation = ", np.std(ste_gyrz))

#Heart Rate Variability
print("")
print("[HRM] Mean Value = ", np.mean(ste_hrm))
print("[HRM] Minimum Value = ", np.min(ste_hrm))
print("[HRM] Maximum Value = ", np.max(ste_hrm))
print("[HRM] Standard Deviation = ", np.std(ste_hrm))






