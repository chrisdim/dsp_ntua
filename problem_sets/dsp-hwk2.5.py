# Digital Signal Processing - hwk2 - Part 2.5
# Christos Dimopoulos - 03117037

# Exercise 2.5: Digital Filter Design
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal

plt.close('all')
counter =0

fs = 0.001
omega = np.arange(0,2*np.pi,fs)
Hcont = 5/(np.sqrt(omega**4-6*omega**2+25))
counter+=1
plt.figure(counter)
plt.plot(omega,abs(Hcont))
plt.xlabel('Ω = ω/Τ [rad/sec]')
plt.ylabel('Magnitude |H(jΩ)|')
plt.title('Magnitude Response of Analogue RLC Filter')
plt.show()

Hdiscrete = -1.25j*((1/(1-np.exp(-1+2j-omega*1j))-(1/(1-np.exp(-1-2j-omega*1j)))))
counter=counter+1
plt.figure(counter)
plt.plot(omega,abs(Hdiscrete), label='Impulse Invariance Method')
plt.xlabel('ω [rad/sec]')
plt.ylabel('Magnitude |Η(ω)|')
plt.title('Magnitude Response of Digital Filter')
plt.legend()
plt.ylim(bottom=0)
plt.show()
