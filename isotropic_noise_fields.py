# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 23:15:00 2021

@author: Rafael Karrer
"""

import numpy as np

# Spherically isotropic noise field for a 1D sensor array
class SphericallyIsotropicNoiseField_1DArray:
    def __init__(self, sample_frequency=22050, 
                 signal_length=1, 
                 number_of_sources=50, 
                 sensor_distance=0.1, 
                 speed_of_sound=340):
        
        # Initialize parameters
        self.speed_of_sound = speed_of_sound
        self.sample_frequency = sample_frequency
        self.signal_length = signal_length
        self.number_of_sources = number_of_sources
        self.sensor_distance = sensor_distance
        
        # Compute number of time-domain samples 
        self.number_of_samples = int(self.sample_frequency * self.signal_length)
        
        # Compute minimum required FFT length
        self.fft_length = 2**int(np.ceil(np.log2(self.number_of_samples / 2)))
        
        # Compute frquency vector
        self.frequency_vector = self.sample_frequency / 2 * np.array([np.arange(self.fft_length+1)])/(self.fft_length+1)
        
        # Compute cosine of azimuth angle
        self.cos_azimuth = 2 * np.array([np.arange(number_of_sources)])/number_of_sources - 1
        #self.azimuth = np.arccos(2 * np.array([np.arange(number_of_sources)])/number_of_sources - 1)
        
        # Compute retardation matrix
        self.retardation_matrix = np.exp(-1j * self.cos_azimuth.T * self.frequency_vector * 2 * np.pi * self.sensor_distance / self.speed_of_sound)
        
    def generate(self):
        # Generate first sensor signal in frequency domain
        X1 = np.random.randn(self.number_of_sources,self.fft_length+1) + 1j*np.random.randn(self.number_of_sources,self.fft_length+1)
        
        # Apply retardation matrix to derive second sensor signal
        X2 = np.multiply(X1, self.retardation_matrix)
        
        # Collapse and sum sensor signals over all angles
        X1 = np.array([np.sum(X1,0)])
        X2 = np.array([np.sum(X2,0)])
        
        # Extend to full mirrored FFT spectrum 
        X1 = np.concatenate((np.sqrt(self.fft_length)*np.real(X1[:,0:1]),
                             np.sqrt(self.fft_length/2)*X1[:,1:-1],
                             np.sqrt(self.fft_length)*np.real(X1[:,-1:-2:-1]),
                             np.sqrt(self.fft_length/2)*np.conj(np.fliplr(X1[:,1:-1]))),axis=1)
        X2 = np.concatenate((np.sqrt(self.fft_length)*np.real(X2[:,0:1]),
                             np.sqrt(self.fft_length/2)*X2[:,1:-1],
                             np.sqrt(self.fft_length)*np.real(X2[:,-1:-2:-1]),
                             np.sqrt(self.fft_length/2)*np.conj(np.fliplr(X2[:,1:-1]))),axis=1)
        
        # Transform to time domain
        x1 = np.real(np.fft.ifft(X1))
        x2 = np.real(np.fft.ifft(X2))
        
        # Crop to specified length
        x1 = x1[:,0:self.number_of_samples]
        x2 = x2[:,0:self.number_of_samples]
        
        return x1,x2

# Spherically isotropic noise field for a 3D sensor array
class SphericallyIsotropicNoiseField_3DArray:
    def __init__(self):
        raise NotImplementedError

# Cylindrically isotropic noise field for a 1D sensor array
class CylindricallyIsotropicNoiseField_1DArray:
    def __init__(self):
        raise NotImplementedError
        
# Cylindrically isotropic noise field for a 3D sensor array
class CylindricallyIsotropicNoiseField_3DArray:
    def __init__(self):
        raise NotImplementedError