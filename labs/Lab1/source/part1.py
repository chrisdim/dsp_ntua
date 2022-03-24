# Digital Signal Processing - Lab 1 - Part 1
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
import sounddevice as sd
import soundfile as sf

plt.close('all')

# Part 1

#1.1 Defining 10 telephone tones
n = np.arange(0,1000,1) #number of samples

Frow = np.array([0.5346, 0.5906, 0.6535, 0.7217])
Fcolumn = np.array([0.9273, 1.0247, 1.1328])

def digit_tone(row, column):
    digit = np.sin(Frow[row]*n) + np.sin(Fcolumn[column]*n)
    return digit

digit1 = digit_tone(0,0)
digit2 = digit_tone(0,1)
digit3 = digit_tone(0,2)
digit4 = digit_tone(1,0)
digit5 = digit_tone(1,1)
digit6 = digit_tone(1,2)
digit7 = digit_tone(2,0)
digit8 = digit_tone(2,1)
digit9 = digit_tone(2,2)
digit0 = digit_tone(3,1)

counter=0
counter = counter+1
plt.figure(counter)
plt.plot(n,digit5, label ='d5[n] = sin(0.5906n)+sin(1.0247n)')
plt.xlabel('Discrete Time n')
plt.ylabel('d5[n]')
plt.title('Tone of Digit 5')
plt.legend()
plt.show()

#1.2 Computing DFT of digits d4[n] and d6[n]
counter = counter+1
plt.figure(counter)
DFT4 = np.fft.fft(digit4)
f = np.fft.fftfreq(1000, 1/1000)
plt.plot(f,np.abs(DFT4),'b', label='|F{sin(0.5906n)+sin(0.9273n)}|')
plt.xlabel('Sample frequency k')
plt.ylabel('|D4(k)|')
plt.title('DFT of tone d4[n]')
plt.legend()
plt.show()

counter = counter+1
plt.figure(counter)
DFT6 = np.fft.fft(digit6)
f = np.fft.fftfreq(1000, 1/1000)
plt.plot(f,np.abs(DFT6),'b', label='|F{sin(0.5906)+sin(1.1328)}|')
plt.xlabel('Sample frequency k')
plt.ylabel('|D6(k)|')
plt.title('DFT of tone d6[n]')
plt.legend()
plt.show()

#1.3 Synthesizing tone sequence
#AM1 = 03117037
#AM2 = 03117165
#Digit_Seq = AM1+AM2 = 06234202
digits = 8
samples = 1000*digits + 100*(digits-1) #including 100 zeros between tones
n = np.arange(0,samples,1) #number of samples for 8 digits
tone_seq = np.zeros((samples))

tone_seq[0:1000] = digit0
tone_seq[1100:2100] = digit6
tone_seq[2200:3200] = digit2
tone_seq[3300:4300] = digit3
tone_seq[4400:5400] = digit4
tone_seq[5500:6500] = digit2
tone_seq[6600:7600] = digit0
tone_seq[7700:8700] = digit2

#Plot tone sequence in time
counter = counter+1
plt.figure(counter)
n = np.arange(0,8700,1)
plt.plot(n,tone_seq,'b', label ='Tone Sequence: 06234202')
plt.xlabel('Discrete Time n')
plt.ylabel('Tone_Sequence[n]')
plt.title('Tone Sequence of 8 tones')
plt.legend()
plt.show()

#Plor DFT of tone sequence
counter = counter+1
plt.figure(counter)
y = np.fft.fft(tone_seq)
f = np.linspace(-np.pi, np.pi,8700)
f = np.fft.fftshift(f)
plt.plot(f,np.abs(y),'b', label ='Tone Sequence: 06234202')
plt.xlabel('Radial frequency ω')
plt.ylabel('|F(ω)|')
plt.title('DFT of tone sequence')
plt.legend()
plt.show()

# Record tone sequence
sf.write('tone_sequence.wav', tone_seq, 3000)
sd.play(tone_seq,3000)

#1.4 Windowed FFT

def windowed_dft(i,j):
    n = 4096 # for more smooth result
    #Rectangular Window:
    window_rect = sp.signal.get_window("boxcar", 1000, 'true') #equivalent to no window at all
    #Hamming Window:
    window_hamming = sp.signal.get_window("hamming", 1000, 'true')
    
    t = np.arange(0, 1, step=1/1000.)
    
    #Slide in time the rectangular window signal:
    windowed = np.zeros((int(np.size(tone_seq))))
    windowed[i:j] = window_rect
    s = tone_seq*windowed
    s = s[i:j]
    
    #Rectangular Windowed DFT:
    w = np.fft.rfft(s, n=n)
    freqs = np.fft.rfftfreq(n, d=t[1] - t[0]) # DFT sample frequencies
    
    #Slide in time the Hamming window signal:
    windowed = np.zeros((int(np.size(tone_seq))))
    windowed[i:j] = window_hamming
    s = tone_seq*windowed
    s = s[i:j]
    
    #Hamming Windowed DFT:
    v = np.fft.rfft(s, n=n)

    plot = plt.subplot(2, 1, 1)
    plt.plot(freqs, 20*np.log10(np.abs(w)), 'b', label='Rectangular Window')
    plt.xlabel('k = 1000ω/2π sample frequencies')
    plt.ylabel('DFT (logarithmic)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(freqs, 20*np.log10(np.abs(v)), 'b', label='Hamming Window')
    plt.xlabel('k = 1000ω/2π sample frequencies')
    plt.ylabel('DFT (logarithmic)')
    plt.legend()
    return plot

#Plot windowed DFT for all 8 digits of tone_sequence:
    
counter = counter+1
plt.figure(counter)
f4 = windowed_dft(0,1000)
plt.suptitle('DFT of samples 0-1000 (Tone of Digit 0)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(1100,2100)
plt.suptitle('DFT of samples 1100-2100 (Tone of Digit 6)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(2200,3200)
plt.suptitle('DFT of samples 2200-3200 (Tone of Digit 2)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(3300,4300)
plt.suptitle('DFT of samples 3300-4300 (Tone of Digit 3)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(4400,5400)
plt.suptitle('DFT of samples 4400-5400 (Tone of Digit 4)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(5500,6500)
plt.suptitle('DFT of samples 5500-6500 (Tone of Digit 2)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(6600,7600)
plt.suptitle('DFT of samples 6600-7600 (Tone of Digit 0)')
plt.show()

counter = counter+1
plt.figure(counter)
f4 = windowed_dft(7700,8700)
plt.suptitle('DFT of samples 7700-8700 (Tone of Digit 2)')
plt.show()

#1.5 DFT sample frequency - value of k

def freq_to_k(frequency, N_samples):
    k = np.round((frequency*N_samples)/(2*np.pi))
    return k

def k_to_freq(k, N_samples):
    frequency = (2*np.pi*k)/N_samples
    return frequency

N = 1000 #number of samples
k_row0 = freq_to_k(Frow[0],N)
k_row1 = freq_to_k(Frow[1],N)
k_row2 = freq_to_k(Frow[2],N)
k_row3 = freq_to_k(Frow[3],N)

k_column0 = freq_to_k(Fcolumn[0],N)
k_column1 = freq_to_k(Fcolumn[1],N)
k_column2 = freq_to_k(Fcolumn[2],N)

k_rows = np.array([k_row0, k_row1, k_row2, k_row3])
k_columns = np.array([k_column0, k_column1, k_column2])
print("Values k of rows = ",k_rows)
print("Values k of columns = ", k_columns)

#1.6 Decode Function

#Function that finds the second greatest element of an array
def secondmax(arr,arr_size):   
    # There should be atleast 
    # two elements       
    first = second = -2147483648
    for i in range(arr_size): 
      
        # If current element is 
                # smaller than first 
        # then update both 
                # first and second  
        if (arr[i] > first): 
          
            second = first 
            first = arr[i] 
            
        # If arr[i] is in 
        # between first and  
        # second then update second  
        elif (arr[i] > second and arr[i] != first): 
            second = arr[i] 
      
    if (second == -2147483648): 
        print("There is no second largest element") 
    else: 
        return second

#Function that calculates the number of zeros between consecutive tones:
def zeros_between_tones(sequence,tones):
    zerosbetween = np.zeros((tones+2))
    for i in range(tones+1):    
        samples = sequence[1000*i+int(zerosbetween[i]): 1000*(i+1)+int(zerosbetween[i])]
        zerosbetween[i] = np.size(samples) - np.size(np.nonzero(samples))
    return zerosbetween[0:tones]

def ttdecode(sequence):
    
    #Analyze input tone sequence into separate tones of 1000 samples:
    number_of_tones = round(np.count_nonzero(sequence)/1000)
    tone = np.zeros((number_of_tones,1000))
    zerosbetween = zeros_between_tones(sequence,number_of_tones)
    for i in range(number_of_tones):
        tone[i,:] = sequence[(1000*i+int(np.sum(zerosbetween[0:i]))):(1000*i+int(np.sum(zerosbetween[0:i]))+1000)]
        
    #Calculate Windowed DFT of each tone:
    n = 1000 # for more smooth result
    t = np.arange(0, 1, step=1/1000.)
    freqs = np.fft.rfftfreq(n, d=t[1] - t[0])
    
    #Hamming Window:
    window_hamming = sp.signal.get_window("hamming", 1000, 'true')
    
    #Calculate Energy = |X[k]|^2 of tones:
    energy = np.zeros((number_of_tones,2049))
    max_energy = np.zeros((number_of_tones)) #max energy value
    secondmax_energy = np.zeros((number_of_tones)) #second max energy value
    k_ofmax = np.zeros((number_of_tones,2))
    k_ofmax2 = np.zeros((number_of_tones,2))
    
    
    for i in range(number_of_tones):
        s = tone[i,:]
        #Hamming Windowed DFT:
        DFT = np.fft.rfft(s * window_hamming, n=n)
        energy = np.square(np.abs(DFT))
        max_energy = np.max(energy)
        secondmax_energy = secondmax(energy,np.size(energy))
        k_ofmax[i] = np.where(energy == max_energy)
        k_ofmax2[i] = np.where(energy == secondmax_energy)

    #Define positions in which we have maximum energy for each tone
    positions = np.zeros((number_of_tones,2))
    positions[:,0] = k_ofmax[:,0]
    positions[:,1] = k_ofmax2[:,1]
    
    #Transform positions into k values
    kvalues = np.zeros((number_of_tones,2))
    for i in range(number_of_tones):
        kvalues[i,0] = freqs[int(positions[i,0])]
        kvalues[i,1] = freqs[int(positions[i,1])]
    
    #Accepted k values according to given table
    accepted_k = np.concatenate((k_rows, k_columns), axis=None)
    
    #Function that finds k values closest to the accepted
    def closest(array, K): 
         idx = (np.abs(array - K)).argmin() 
         return array[idx]
    
    #Create array of accepted k values according to table given
    for i in range(number_of_tones):
        kvalues[i,0] = closest(accepted_k, kvalues[i,0])
        kvalues[i,1] = closest(accepted_k, kvalues[i,1])
        
    #Function that accords k values to tone digits
    digitseq = np.zeros(number_of_tones)
    def ktodigit(i, j):
        if (i==k_row0 and j==k_column0)or(j==k_row0 and i==k_column0):
            return '1'
        elif (i==k_row0 and j==k_column1)or(j==k_row0 and i==k_column1):
            return '2'
        elif (i==k_row0 and j==k_column2)or(j==k_row0 and i==k_column2):
            return '3'
        elif (i==k_row1 and j==k_column0)or(j==k_row1 and i==k_column0):
            return '4'
        elif (i==k_row1 and j==k_column1)or(j==k_row1 and i==k_column1):
           return '5'
        elif (i==k_row1 and j==k_column2)or(j==k_row1 and i==k_column2):
            return '6'
        elif (i==k_row2 and j==k_column0)or(j==k_row2 and i==k_column0):
            return '7'
        elif (i==k_row2 and j==k_column1)or(j==k_row2 and i==k_column1):
            return '8'
        elif (i==k_row2 and j==k_column2)or(j==k_row2 and i==k_column2):
            return '9'
        elif (i==k_row3 and j==k_column1)or(j==k_row3 and i==k_column1):
            return '0'
        
        
    for i in range(number_of_tones):
        digitseq[i] = ktodigit(kvalues[i,0],kvalues[i,1])
    
    return digitseq

tonesequence = ttdecode(tone_seq)
print('AM1+AM2 =', tonesequence)

#1.7 Load easySig.npy and hardSig.npy
easysig = np.load('easySig.npy')
easysequence = ttdecode(easysig)
print('Easy Tone Sequence: ', easysequence)

hardsig = np.load('hardSig.npy')
hardsequence = ttdecode(hardsig)
print('Hard Tone Sequence: ', hardsequence)