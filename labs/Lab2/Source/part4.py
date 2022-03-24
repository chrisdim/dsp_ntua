# Digital Signal Processing - Lab 2 - Part 4
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165
# Denoising of motion signals with Butterworth and Wiener filters

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import librosa
from scipy import signal
plt.rcParams.update({'figure.max_open_warning': 0})

plt.close('all')
counter =0

# 4.1 STFTs of accy and gyry

data = np.load('sleep_01.npz')
#data.files
acc = data['acc']
gyr = data['gyr']
hrm = data['hrm']
data.close()

accy = acc[:,1]
gyry = gyr[:,1]
fs1 = 20
Ts1 = 1/fs1

# STFT Parameters
window = 20
overlap = 10
one = int (window / Ts1)
two = int (overlap / Ts1)

# STFT creation
G = librosa.stft(accy, n_fft = one, hop_length = two)
#print(G.shape)

# Plot STFT of Accelartion (Y-axis)
t = np.linspace(0, np.size(accy)*Ts1, 60)
f = np.linspace(0, fs1/2, 201)
counter = counter+1
plt.figure(counter)
plt.pcolormesh(t,f,np.log10(np.abs(G)))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('log-STFT of Acceleration (Y-axis)')

G = librosa.stft(gyry, n_fft = one, hop_length = two)
#print(G.shape)

# Plot STFT of Angular Velocity (Y-axis)
t = np.linspace(0, np.size(gyry)*Ts1, 60)
f = np.linspace(0, fs1/2, 201)
counter = counter+1
plt.figure(counter)
plt.pcolormesh(t,f,20*np.log10(np.abs(G)))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('log-STFT of Angular Velocity (Y-axis)')
plt.show()

# 4.2 Butterworth Filter
sos = sp.signal.butter(6,2,'lowpass', fs=fs1, output = 'sos')
butter_accy = sp.signal.sosfilt(sos,accy)

# Butterworth Filter Frequency Response
t = np.arange(0,np.size(butter_accy)*Ts1, Ts1) #time index
counter = counter+1
plt.figure(counter)
w, h = signal.sosfreqz(sos)
plt.plot(w*fs1/(2*np.pi), (abs(h)), label = 'Order 6')
plt.title('Butterworth Filter Frequency Response')
plt.xlabel('Frequency [Hertz]')
plt.ylabel('Amplitude [V]')
plt.axvline(2, color='magenta', linestyle = '--') # cutoff frequency
plt.legend()
plt.show()

counter=counter+1
plt.figure(counter)
plt.plot(t,butter_accy)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (m/s^2)')
plt.title('Acceleration (Y-axis) after Butterworth Filtering')
plt.show()

counter=counter+1
plt.figure(counter)
butter_gyry = sp.signal.sosfilt(sos,gyry)
t = np.arange(0,np.size(butter_gyry)*Ts1, Ts1) #time index
plt.plot(t,butter_gyry)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (deg/s)')
plt.title('Angular Speed (Y-axis) after Butterworth Filtering')
plt.show()

# STFT Parameters
window = 20
overlap = 10
one = int (window / Ts1)
two = int (overlap / Ts1)

# STFT creation
G = librosa.stft(butter_accy, n_fft = one, hop_length = two)

# Plot STFT of Accelartion (Y-axis) after Butterworth Filtering
t = np.linspace(0, np.size(butter_accy)*Ts1, 60)
f = np.linspace(0, fs1/2, 201)
counter = counter+1
plt.figure(counter)
plt.pcolormesh(t,f,np.log10(np.abs(G)))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('log-STFT of Butterworth Filtered Acceleration (Y-axis)')
plt.show()

G = librosa.stft(butter_gyry, n_fft = one, hop_length = two)
# Plot STFT of Angular Velocity (Y-axis) after Butterworth Filtering
t = np.linspace(0, np.size(butter_gyry)*Ts1, 60)
f = np.linspace(0, fs1/2, 201)
counter = counter+1
plt.figure(counter)
plt.pcolormesh(t,f,20*np.log10(np.abs(G)))
plt.xlabel('Time (sec)')
plt.ylabel('Frequency(Hz)')
plt.title('log-STFT of Butterworth Filtered Angular Velocity (Y-axis)')
plt.show()

# 4.3 Denoising with Wiener Filter
def power_spec(x):
    DFT = np.fft.fft(x)
    P = (np.abs(DFT)**2)/np.size(x)
    return P

number = 0
for sig in [accy, gyry]:
    number=number+1
    #insert 8 samples to have exactly 12000 samples
    for i in range(8):
        sig = np.insert(sig,np.size(sig),sig[np.size(sig)-1])
        
    # Find a segment of no movement [around 300-550 sec]
    nomove = sig[int(300*fs1):int(300*fs1)+400]
    Pnoise = power_spec(nomove) # 400 samples
    L = 400 # window length
    rectwin = signal.boxcar(L)
    total_win = np.size(sig)//L
    wiener_sig = np.zeros(np.size(sig),dtype=complex)
    for i in range(total_win):
        v = sig[i*L:(i+1)*L]*rectwin
        Px = power_spec(v)
        Pd = Px - Pnoise
        for j in range(np.size(Pd)):
            if Pd[j]<0:
                Pd[j]=0
        Hwiener = Pd/(Pd+Pnoise)
        DFT = np.fft.fft(v)
        Output = DFT*Hwiener
        wiener_sig[i*L:(i+1)*L] = np.fft.ifft(Output)
    
    # Plot signal after Wiener Filtering
    counter=counter+1
    plt.figure(counter)
    t = np.arange(0,np.size(wiener_sig)*Ts1, Ts1) #time index
    plt.plot(t,sig, label='Original Signal')
    
    # Discard Imaginary part --> unimportant
    plt.plot(t,wiener_sig.real, label ='Signal after Wiener Filtering')
    plt.xlabel('Time (sec)')
    ylabels = ['Amplitude (m/s^2)','Amplitude (deg/s)']
    plt.ylabel(ylabels[number-1])
    titles = ['Acceleration (Y-axis) after Wiener Filtering','Gyroscope (Y-axis) after Wiener Filtering']
    plt.title(titles[number-1])
    plt.legend()
    plt.show()

    # STFT Parameters
    window = 20
    overlap = 10
    one = int (window / Ts1)
    two = int (overlap / Ts1)
    
    # STFT creation
    G = librosa.stft(np.abs(wiener_sig), n_fft = one, hop_length = two)
    # Plot STFT after Wiener Filtering
    t = np.linspace(0, np.size(sig)*Ts1, 61)
    f = np.linspace(0, fs1/2, 201)
    counter = counter+1
    plt.figure(counter)
    G = G + 0.0001 #so that log0 does not occur
    plt.pcolormesh(t,f,np.log10(np.abs(G)))
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency(Hz)')
    titles = ['log-STFT of Butterworth Filtered Acceleration (Y-axis)','log-STFT of Butterworth Filtered Gyroscope (Y-axis)']
    plt.title(titles[number-1])
    plt.show()
        
        
# 4.4 Butter and Wiener Filtering Signals of our choice
data = np.load('step_00.npz')
#data.files
acc = data['acc']
gyr = data['gyr']
hrm_step = data['hrm']
data.close()
accz_step = acc[:,2]
gyrz_step = gyr[:,2]

data = np.load('sleep_00.npz')
#data.files
acc = data['acc']
gyr = data['gyr']
hrm_sleep = data['hrm']
data.close()
accz_sleep = acc[:,2]
gyrz_sleep = gyr[:,2]


#insert 8 samples to have exactly 12000 samples
for i in range(8):
    accz_step = np.insert(accz_step,np.size(accz_step),accz_step[np.size(accz_step)-1])
    accz_sleep = np.insert(accz_sleep,np.size(accz_sleep),accz_sleep[np.size(accz_sleep)-1])
    gyrz_step = np.insert(gyrz_step,np.size(gyrz_step),gyrz_step[np.size(gyrz_step)-1])
    gyrz_sleep = np.insert(gyrz_sleep,np.size(gyrz_sleep),gyrz_sleep[np.size(gyrz_sleep)-1])


fs1 = 20 #sampling freq
Ts1 = 1/fs1
t = np.arange(0,np.size(accz_step)*Ts1, Ts1) #time index

counter = counter+1
plt.figure(counter)
plt.suptitle('Sleep and Step Signals of our choice')
plt.subplot(411)
plt.plot(t,accz_step,'b', label = 'Accz - "step_00.npz"')
plt.xlabel('Time [sec]')
plt.ylabel('[m/sec^2]')
plt.legend()

plt.subplot(412)
plt.plot(t,accz_sleep,'m', label = 'Accz - "sleep_00.npz"')
plt.xlabel('Time [sec]')
plt.ylabel('[m/sec^2]')
plt.legend()

plt.subplot(413)
plt.plot(t,gyrz_step,'b', label = 'Gyrz - "step_00.npz"')
plt.xlabel('Time [sec]')
plt.ylabel('[deg/sec]')
plt.legend()

plt.subplot(414)
plt.plot(t,gyrz_sleep,'m', label = 'Gyrz - "sleep_00.npz"')
plt.xlabel('Time [sec]')
plt.ylabel('[deg/sec]')
plt.legend()

times = 0
for sig in [accz_step,accz_sleep,gyrz_step,gyrz_sleep]:
    times=times+1
        
    #BUTTERWORTH FILTERING
    sos = sp.signal.butter(6,2,'lowpass', fs=fs1, output = 'sos')
    butter_sig = sp.signal.sosfilt(sos,sig)
    
    counter=counter+1
    plt.figure(counter)
    t = np.arange(0,np.size(sig)*Ts1, Ts1) #time index
    plt.plot(t,sig,'b',label='Original Signal')
    plt.plot(t,butter_sig, 'y',label='After Butterworth Filtering')
    plt.xlabel('Time (sec)')
    ylabels = ['Amplitude (m/s^2)','Amplitude (m/s^2)','Amplitude (deg/sec)','Amplitude (deg/sec)']
    plt.ylabel(ylabels[times-1])
    titles = ['Acceleration (Z-axis) - "step_00.npz"','Acceleration (Z-axis) - "sleep_00.npz"','Gyroscope (Z-axis) - "step_00.npz"','Gyroscope (Z-axis) - "sleep_00.npz"']
    plt.title(titles[times-1])
    plt.legend()
    plt.show()
    
    # STFT Parameters
    window = 20
    overlap = 10
    one = int (window / Ts1)
    two = int (overlap / Ts1)
    
    # STFT creation
    G = librosa.stft(butter_sig, n_fft = one, hop_length = two)
    # Plot STFT of Accelartion (Y-axis) after Butterworth Filtering
    t = np.linspace(0, np.size(butter_sig)*Ts1, 61)
    f = np.linspace(0, fs1/2, 201)
    counter = counter+1
    plt.figure(counter)
    plt.pcolormesh(t,f,np.log10(np.abs(G)))
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency(Hz)')
    titles = ['log-STFT of Butterworth Filtered Acceleration (Z-axis) - "step_00.npz"','log-STFT of Butterworth Filtered Acceleration (Z-axis) "sleep_00.npz"','log-STFT of Butterworth Filtered Gyroscope (Z-axis) - "step_00.npz"','log-STFT of Butterworth Filtered Gyroscope (Z-axis) "sleep_00.npz"']
    plt.title(titles[times-1])
    plt.show()
    
    #WIENER FILTERING
    # Find a segment of no movement [around 506 sec]
    nomove = sig[int(506*fs1):int(506*fs1)+400]
    Pnoise = power_spec(nomove) # 400 samples
    L = 400 # window length
    rectwin = signal.boxcar(L)
    total_win = np.size(sig)//L
    wiener_sig = np.zeros(np.size(sig),dtype=complex)
    for i in range(total_win):
        v = sig[i*L:(i+1)*L]*rectwin
        Px = power_spec(v)
        Pd = Px - Pnoise
        for j in range(np.size(Pd)):
            if Pd[j]<0:
                Pd[j]=0
        Hwiener = Pd/(Pd+Pnoise)
        DFT = np.fft.fft(v)
        Output = DFT*Hwiener
        wiener_sig[i*L:(i+1)*L] = np.fft.ifft(Output)
    
    # Plot signal after Wiener Filtering
    counter=counter+1
    plt.figure(counter)
    t = np.arange(0,np.size(wiener_sig)*Ts1, Ts1) #time index
    plt.plot(t,sig, label='Original Signal')
    plt.plot(t,wiener_sig.real, label ='Signal after Wiener Filtering')
    plt.xlabel('Time (sec)')
    ylabels = ['Amplitude (m/s^2)','Amplitude (m/s^2)','Amplitude (deg/s)','Amplitude (deg/s)']
    plt.ylabel(ylabels[times-1])
    titles = ['Acceleration (Z-axis) after Wiener Filtering - "step_00.npz"','Acceleration (Z-axis) after Wiener Filtering - "sleep_00.npz"','Gyroscope (Z-axis) after Wiener Filtering - "step_00.npz"','Gyroscope (Z-axis) after Wiener Filtering - "sleep_00.npz"']
    plt.title(titles[times-1])
    plt.legend()
    plt.show()

    # STFT creation
    G = librosa.stft(np.abs(wiener_sig), n_fft = one, hop_length = two)
    # Plot STFT after Wiener Filtering
    t = np.linspace(0, np.size(sig)*Ts1, 61)
    f = np.linspace(0, fs1/2, 201)
    counter = counter+1
    plt.figure(counter)
    G = G + 0.0001 #so that log0 does not occur
    plt.pcolormesh(t,f,np.log10(np.abs(G)))
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency(Hz)')
    titles = ['log-STFT of Wiener Filtered Acceleration (Z-axis) "step_00.npz"','log-STFT of Wiener Filtered Acceleration (Z-axis) "sleep_00.npz"','log-STFT of Wiener Filtered Gyroscope (Z-axis) "step_00.npz"','log-STFT of Wiener Filtered Gyroscope (Z-axis) "sleep_00.npz"']
    plt.title(titles[times-1])
    plt.show()
        
    
    
    
    
