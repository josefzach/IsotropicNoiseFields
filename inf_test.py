# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 23:19:52 2021

@author: Rafael Karrer
"""

from isotropic_noise_fields import SphericallyIsotropicNoiseField_1DArray
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# set parameters
fft_length = 256
sample_frequency = 44100
sensor_distance = 0.01
speed_of_sound = 340
signal_length = 10
number_of_sources = 50

# compute theoretical magnitude squared coherence
arg = 2*np.pi*np.arange(0,fft_length/2+1)/fft_length*sample_frequency*sensor_distance/speed_of_sound
arg[0] = 1
Cth = np.abs(np.sin(arg)/(arg))**2
Cth[0] = 1

# Spherically isotropic noise field for 1D array
sinf_1d = SphericallyIsotropicNoiseField_1DArray(sample_frequency=sample_frequency, 
                                                 signal_length=signal_length, 
                                                 number_of_sources=number_of_sources, 
                                                 sensor_distance=sensor_distance, 
                                                 speed_of_sound=speed_of_sound)
x1,x2 = sinf_1d.generate()

# Magnitude squared coherence estimate
f, C = signal.coherence(x1,x2, sample_frequency, nperseg=fft_length)

# Plot theoretical curve versus estimated result
fig, ax = plt.subplots()
ax.plot(f, Cth.T, label='Theoretical MSC')
ax.plot(f, C.T, label='Estimated MSC')
ax.legend(loc='upper right', shadow=True)
plt.ylim([0,1])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude Squared Coherence')
plt.show()
