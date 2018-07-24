import collections
import os
import glob
import pandas as pd
import numpy as np
import scipy
from numpy.lib import stride_tricks

from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
from scipy import signal
from scipy import stats, signal
from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scipy.signal import butter, lfilter, freqz


from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import pylab

import librosa
import librosa.display

from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import math


'''
This class enables interactive plots which zoom out and zoom in on mouse scroll
'''
class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion


'''
Compute the covariance between sequence 1 and sequence 2. 
'''
def covariance(seq1, seq2):
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)

    cov = np.cov(seq1, seq1)
    # if plot_true == True:
    #     plt.plot(cov)
    return cov  

'''
Compute the covariance between sequence 1 and sequence 2. 
Args: 
- array1, array2 are the sequence of whom you want to compute the correlation
- Mode
    - 'full':
            This returns the convolution at each point of overlap, with an output shape of (N+M-1,). 
            At the end-points of the convolution, the signals do not overlap completely, and boundary effects may be seen.
    - 'same':
            Mode ‘same’ returns output of length max(M, N). Boundary effects are still visible.
    - 'valid':
             Mode ‘valid’ returns output of length max(M, N) - min(M, N) + 1. The convolution product 
            is only given for points where the signals overlap completely. Values outside the signal boundary have no effect. 
'''
def correlation(array1, array2, mode, plot_true):

    cor = signal.correlate2d(array1, array2, boundary='symm', mode=mode)      
    if plot_true == True:
        plt.plot(corr)

    return cor          

'''
Computes and plots the Mel Frequency Cepstral Coefficients and plots them. 
Args:
- y: The audio signal array
- sr: The sampling rate
- n_mfcc: number of coefficients to be returned
- plotFlag: This is a boolean argument which when set True displays the plots  
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
- filename:  Name of the audio file you are which is under test. This will be the name of the plot files which are being saved. 
'''

def mfcc(y,sr, n_mfcc,plotFlag,save_flag,filename):
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    write('../test_audio/fut.wav', sr, y)       #write file under test

    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(filename + ':MFCC: First ' + str(len(y)) + ' iterations' + ' with no. of coeffs = ' + str(n_mfcc))
    plt.tight_layout()
        
    if save_flag:
        pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'MFCC.png')    
    if plotFlag:
        plt.show()


'''
Computes and plots the Mel Spectogram. 
Args:
- y: The audio signal array
- sr: The sampling rate
- n_mels: number of MFCCs
- max_freq: The maximum frequency upto which you want to compute the spectrum  
- plotFlag: This is a boolean argument which when set True displays the plots 
- flag_hp  This is a boolean argument which when set True does seperates the harmonic and percussive components and processing is done on the individual components
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
- filename:  Name of the audio file you are which is under test. This will be the name of the plot files which are being saved. 
'''
def melSpectrogram(y, sr, n_mels, max_freq, plotFlag,flag_hp, save_flag, filename):
    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)
        figure()
        S = librosa.feature.melspectrogram(y=y_harm, sr=sr, n_mels=n_mels, fmax=max_freq)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='mel', fmax=max_freq,x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('../results/' + filename + '_' + str(len(y)) + 'i_' + 'melspectrogram.png')
        plt.tight_layout()

        if save_flag:
            pylab.figure(figsize=(70, 70))
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'melspectrogram.png')

        figure()
        S = librosa.feature.melspectrogram(y=y_perc, sr=sr, n_mels=n_mels, fmax=max_freq)
        librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='mel', fmax=max_freq,x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('../results/' + filename + '_' + str(len(y)) + 'i_' + 'melspectrogram.png')
        plt.tight_layout()

        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'melspectrogram.png')

        if plotFlag:
            plt.show()                

    else:     
        figure()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=max_freq)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='mel', fmax=max_freq,x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('../results/' + filename + '_' + str(len(y)) + 'i_' + 'melspectrogram.png')
        plt.tight_layout()

        if save_flag:
            #pylab.figure(figsize=(70, 70))
            #pylab.savefig('test.eps', format='eps', dpi=900) # This does, too
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'melspectrogram.png')

        if plotFlag:
            plt.show()    

    


'''
Computes and plots the Spectogram of the audio signal 
- y: The audio signal array
- sr: The sampling rate
- n_mels: number of MFCCs
- hop_length: The hop_length for the stft operation  
- plotFlag: This is a boolean argument which when set True displays the plots 
- flag_hp  This is a boolean argument which when set True does seperates the harmonic and percussive components and processing is done on the individual components
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
- filename:  Name of the audio file you are which is under test. This will be the name of the plot files which are being saved. 
'''
def spectrogram(y, hop_length, sr, plotFlag,flag_hp,save_flag, filename):


    write('../test_audio/fut.wav', sr, y)      #write file under test
    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)
        write('../test_audio/fut__hamonic_comp.wav', sr, y_harm)
        write('../test_audio/fut_percussive_comp.wav', sr, y_perc)

        D_harm = librosa.stft(y_harm, hop_length=hop_length)
        D_perc = librosa.stft(y_perc, hop_length=hop_length)

    
        plt.subplot(211)    
        librosa.display.specshow(librosa.amplitude_to_db(D_harm,
                                                       ref=np.max),
                               y_axis='log', x_axis='time')
        plt.title('Harmonic')    
        #plt.title('Turkish March:Power spectrogram of harmonic component: First ' + str(len(y)) + ' iterations' + ' with hopsize = ' + str(hop_length))
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()            

        plt.subplot(212)
        librosa.display.specshow(librosa.amplitude_to_db(D_perc,
                                                       ref=np.max),
                               y_axis='log', x_axis='time')
        plt.title('Percussion')
        #plt.title('Turkish March:Power spectrogram of percussive component: First ' + str(len(y)) + ' iterations' + ' with hopsize = ' + str(hop_length))
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'harm_perc_spectogram.png')
        if plotFlag:
            plt.show()

    else:        
        D = librosa.stft(y, hop_length=hop_length)
        #D_left = librosa.stft(y, center=False)

        #D_short = librosa.stft(y, hop_length=64)

        librosa.display.specshow(librosa.amplitude_to_db(D,
                                                       ref=np.max),
                               y_axis='log', x_axis='time')
        plt.title(filename + ':Power spectrogram: First ' + str(len(y)) + ' iterations' + ' with hopsize = ' + str(hop_length))
        plt.colorbar(format='%+2.0f dB')
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'spectogram.png')
        plt.tight_layout()
        if plotFlag:             
            plt.show()

'''
Computes the pitch of the audio file
Args:
- y: The audio signal array
- sr: The sampling rate
'''
def getPitch(y,sr):
    sp = np.fft.fft(y)
    
    freq = np.fft.fftfreq(y.shape[-1])
    half = int(y.shape[-1]/2)
    freq = freq[0:half] 
    freqHz = freq * sr    
    pitch = 69 + 12*np.log2(freqHz/440.0) 
    print("Using the MIDI standard to map frequency to pitch")
    # plt.plot(freqHz, pitch)
    # plt.show()
    return pitch

'''
Computes frequencies contained in the audio signal 
Args:
- y: The audio signal array
- sr: The sampling rate
'''
def getFreq(y,sr):
    sp = np.fft.fft(y)
    freq = np.fft.fftfreq(y.shape[-1])
    half = int(y.shape[-1]/2)
    freq = freq[0:half] 
    freqHz = freq * sr    
    return freqHz, np.max(freqHz), np.min(freqHz)

'''
Computes the RMS energy of the audio signal 
Args:
- y: The audio signal array
- sr: The sampling rate
'''
def rmsEnergy(y):
    y = np.square(y)
    E = np.sum(y)
    rms_E = np.sqrt(E)
    rms_E = float(rms_E)/float(len(y))
    return rms_E 

'''
Computes the centroid of the spectrum of the audio signal 
Args:
- y: The audio signal array
- sr: The sampling rate
'''
def spectral_centroid(y,sr):
    sp = np.fft.fft(y)
    
    freq = np.fft.fftfreq(y.shape[-1])
    mag = np.abs(sp)
    half = int(y.shape[-1]/2)
    mag = mag[0:half]
    freq = freq[0:half]     
    
    freqHz = freq * sr

    centroid = float(np.sum(np.multiply(mag,freqHz)))/float(np.sum(mag))
    return centroid

'''
Computes the number of zero crossings in the audio signal 
Args:
- y: The audio signal array
- sr: The sampling rate
'''
def zero_crossing(y,sr):
    l = []
    for i in range(len(y)-1):
        if y[i]*y[i+1] < 0:
            l.append(i)
    zero_crossing_rate = float(len(l))/float(len(y))
    return zero_crossing_rate, l         

'''
Plot the audio signal in time domain 
Args:
- y: The audio signal array
- sr: The sampling rate
- plotFlag: This is a boolean argument which when set True displays the plots 
- flag_hp  This is a boolean argument which when set True does seperates the harmonic and percussive components and processing is done on the individual components
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
- filename:  Name of the audio file you are which is under test. This will be the name of the plot files which are being saved. 
'''

def plotTimeSeries(y,sr, flag_hp, plotFlag, save_flag, filename):
    # y_8k = librosa.resample(y, sr, sr/downsampleF)
    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)
        write('../test_audio/fut__hamonic_comp.wav', sr, y_harm)
        write('../test_audio/fut_percussive_comp.wav', sr, y_perc)
        librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
        librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
        plt.title('Harmonic + Percussive')
        plt.tight_layout()
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'HPtimeseries.png')
        if plotFlag:
            plt.show()

    else:    
        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=True)
        write('../test_audio/fut.wav', sr, y)
        ax.set_title('Time Series plot of music data')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('time')
        ax.plot(y)
        scale = 1.1
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax, base_scale = scale)
        figPan = zp.pan_factory(ax)
        ax.legend()
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'timeseries.png')
        if plotFlag:
            plt.show()


'''
Plot the spectrum of the audio signal 
Args:
- y: The audio signal array
- sr: The sampling rate
- plotFlag: This is a boolean argument which when set True displays the plots 
- flag_hp  This is a boolean argument which when set True does seperates the harmonic and percussive components and processing is done on the individual components
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
- filename:  Name of the audio file you are which is under test. This will be the name of the plot files which are being saved. 
'''
def plotSpectrum(y,sr,flag_hp, plotFlag, save_flag, filename):

    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)
        write('../test_audio/fut__hamonic_comp.wav', sr, y_harm)
        write('../test_audio/fut_percussive_comp.wav', sr, y_perc)

        sp = np.fft.fft(y_harm)
        
        freq = np.fft.fftfreq(y_harm.shape[-1])
        mag = np.abs(sp)
        half = int(y.shape[-1]/2)
        mag = mag[0:half]
        freq = freq[0:half] 
        freqHz = freq * sr

        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=True)

        ax.set_title('Harmony Spectrum of ' + filename )
        ax.set_xlabel('Maginitude')
        ax.set_ylabel('Frequency [in hertz]')
        ax.plot(freqHz, mag)
        scale = 1.1
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax, base_scale = scale)
        figPan = zp.pan_factory(ax)
        ax.legend()
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'Hspectrum.png')

        sp = np.fft.fft(y_perc)
        
        freq = np.fft.fftfreq(y_perc.shape[-1])
        mag = np.abs(sp)
        half = int(y.shape[-1]/2)
        mag = mag[0:half ]
        freq = freq[0:half] 
        freqHz = freq * sr

        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=True)

        ax.set_title('Percussion Spectrum of ' + filename)
        ax.set_xlabel('Maginitude')
        ax.set_ylabel('Frequency [in hertz]')
        ax.plot(freqHz, mag)
        scale = 1.1
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax, base_scale = scale)
        figPan = zp.pan_factory(ax)
        ax.legend()
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'Pspectrum.png')
        if plotFlag:
            plt.show()        

    else: 
        sp = np.fft.fft(y)

        freq = np.fft.fftfreq(y.shape[-1])
        mag = np.abs(sp)
        half = int(y.shape[-1]/2)
        mag = mag[0:half ]
        freq = freq[0:half] 
        freqHz = freq * sr

        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=True)

        ax.set_title('Spectrum of ' + filename)
        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Frequency [in hertz]')
        ax.plot(freqHz, mag)
        scale = 1.1
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax, base_scale = scale)
        figPan = zp.pan_factory(ax)
        ax.legend()
        if save_flag:
            pylab.savefig('../results/' + filename + '_' + str(len(y)) + 'i_' + 'spectrum.png')
        if plotFlag:
            plt.show()


'''
returns the numerator and denominator polynomials of the IIR filter 
Args:
- cutoff: The cutoff frequency 
- fs: Sampling Frequency 
- order: order of the filter 
'''
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

'''
Returns the filtered signal by applying butterworth low pass filter 
Args:
- y: raw signal which you want to filter  
- cutoff: The cutoff frequency 
- fs: Sampling Frequency 
- order: order of the filter 
'''

def butter_lowpass_filter(y, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, y)

    return y

'''
Useful comments: 

-The Nyquist frequency is half the sampling rate.

-You are working with regularly sampled data, so you want a digital filter, not an analog filter. 
This means you should not use analog=True in the call to butter, and you should use scipy.signal.freqz (not freqs) to generate the frequency response.

-One goal of those short utility functions is to allow you to leave all your frequencies expressed in Hz.
As long as you express your frequencies with consistent units, the scaling in the utility functions takes care of the normalization for you.
'''


'''
Returns the filtered signal by applying butterworth low pass filter. Plots the output and the input signal to enable analysis 
Args:
- y: raw signal which you want to filter  
- cutoff: The cutoff frequency 
- fs: Sampling Frequency 
- order: order of the filter 
- plotFlag: This is a boolean argument which when set True displays the plots 
- freq_resp_plot : This is a boolean argument which when set True displays the plots in frequency domain  
'''
def lpf(y, sr, order, fs, cutoff, freq_resp_plot, plotFlag):

    if freq_resp_plot:    
        # Get the filter coefficients so we can check its frequency response.
        b, a = butter_lowpass(cutoff, fs, order)
        # Plot the frequency response.
        w, h = freqz(b, a, worN=8000)
        # plt.subplot(2, 1, 1)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        pylab.savefig('freq_response.png')



    y_filtered = butter_lowpass_filter(y, cutoff, fs, order)
    figure()
    # plt.subplot(2, 1, 2)
    plt.plot(y, 'b-', label='data')
    figure()
    plt.plot(y_filtered, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    pylab.savefig('fitered.png')
    if plotFlag:
        plt.show()

    return y_filtered    


'''
DISCLAIMER: Not working correctly.  
Perform cepstral analysis of the audio signal
Args:
- y: The audio signal array
- sr: The sampling rate
- plotFlag: This is a boolean argument which when set True displays the plots 
'''
def cepstral_analysis(y, sr, plotFlag):

    logfft = np.log(fft(y))
    freq = np.fft.fftfreq(y.shape[-1])

    ceps = ifft(logfft)
    if plotFlag:
        figure()
        plt.plot(freq*sr, np.abs(logfft))
        pylab.savefig('logfft.png')
        figure()
        plt.plot(ceps)
        pylab.savefig('ceps.png')
    
    return ceps

'''
compute and plot the fft of an image
- filepath: filepath of the image 
- shift_zeroF:  This is a boolean argument which when set True shifts the zero-frequency component to the center of the spectrum.
- plotFlag: This is a boolean argument which when set True displays the plots 
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
'''
def img_fft(filepath, shift_zeroF, plotFlag, save_flag):
    img = np.asarray(Image.open(filepath))    # Reads the images in gray scale
    f = np.fft.fft2(img)            # Computes the fourier transform
    if shift_zeroF:
        f = np.fft.fftshift(f)   
    
    mag_spectrum = np.log(np.abs(f))
    phase = np.angle(img_fft)

    
    plt.subplot(121),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(mag_spectrum)
    plt.title('FFT magnitude'), plt.xticks([]), plt.yticks([])

    plt.show()

    return f # mag_spectrum, phase

'''
compute and plot the inverse fft of the fft of an image
- img_fft: fft of an image 
- shift_zeroF:  This is a boolean argument which when set True shifts the zero-frequency component to the center of the spectrum.
- plotFlag: This is a boolean argument which when set True displays the plots 
- save_flag: This is a boolean argument which when set True saves the plots on your filesystem
'''

def img_ifft(img_fft, shift_zeroF, plotFlag, save_flag):
    if shift_zeroF:
        img_back = np.fft.ifftshift(fshift)             #The inverse of fftshift
    else:
        img_back = np.fft.ifft2(f)               #This function computes the inverse of the 2-dimensional discrete Fourier Transform  
    img_back = np.log(np.abs(img_back))

    return img_back



'''
Computes and plots the tempogram of an audio signal  
Args:
- y: The audio signal array
- sr: The sampling rate
- hop_length: hop_length for stft operation 
- plotFlag: This is a boolean argument which when set True displays the plots 
- flag_hp  This is a boolean argument which when set True does seperates the harmonic and percussive components and processing is done on the individual components
'''

def tempogram(y, sr, hop_length, flag_hp, plotFlag):

    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)
        write('../test_audio/fut__hamonic_comp.wav', sr, y_harm)
        write('../test_audio/fut_percussive_comp.wav', sr, y_perc)        

        y = y_harm
        onset_envp = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)       # compute the onset envelope of the music signal 
        tempogram = librosa.feature.tempogram(onset_envelope=onset_envp, sr=sr, hop_length=hop_length)   # compute the tempogram
        
        global_autoCorr = librosa.autocorrelate(onset_envp, max_size=tempogram.shape[0])
        global_autoCorr = librosa.util.normalize(global_autoCorr)
        # Estimate the global tempo for display purposes
        
        tempo = librosa.beat.tempo(onset_envelope=onset_envp, sr=sr,
                                   hop_length=hop_length)[0]

        # figure()
        # plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        # plt.legend(frameon=True, framealpha=0.75)
        # x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
        #                 num=tempogram.shape[0])
        # plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
        # plt.plot(x, global_autoCorr, '--', alpha=0.75, label='Global autocorrelation')
        # plt.xlabel('Lag (seconds)')
        # plt.axis('tight')
        # plt.legend(frameon=True)
        figure()

        # PLOT IN BPM [Beats Per Second]
        freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
        plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                     label='Mean local autocorrelation', basex=2)
        plt.semilogx(freqs[1:], global_autoCorr[1:], '--', alpha=0.75,
                     label='Global autocorrelation', basex=2)
        plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
                    label='Estimated tempo={:g}'.format(tempo))
        plt.legend(frameon=True)
        plt.title("harmonic")
        plt.xlabel('BPM')
        plt.axis('tight')
        plt.grid()
        plt.tight_layout()

        y = y_perc
        onset_envp = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)       # compute the onset envelope of the music signal 
        tempogram = librosa.feature.tempogram(onset_envelope=onset_envp, sr=sr, hop_length=hop_length)   # compute the tempogram
        
        global_autoCorr = librosa.autocorrelate(onset_envp, max_size=tempogram.shape[0])
        global_autoCorr = librosa.util.normalize(global_autoCorr)
        # Estimate the global tempo for display purposes
        
        tempo = librosa.beat.tempo(onset_envelope=onset_envp, sr=sr,
                                   hop_length=hop_length)[0]

        # figure()
        # plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        # plt.legend(frameon=True, framealpha=0.75)
        # x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
        #                 num=tempogram.shape[0])
        # plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
        # plt.plot(x, global_autoCorr, '--', alpha=0.75, label='Global autocorrelation')
        # plt.xlabel('Lag (seconds)')
        # plt.axis('tight')
        # plt.legend(frameon=True)
        figure()

        # PLOT IN BPM [Beats Per Second]
        freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
        plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                     label='Mean local autocorrelation', basex=2)
        plt.semilogx(freqs[1:], global_autoCorr[1:], '--', alpha=0.75,
                     label='Global autocorrelation', basex=2)
        plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
                    label='Estimated tempo={:g}'.format(tempo))
        plt.legend(frameon=True)
        plt.xlabel('BPM')
        plt.axis('tight')
        plt.title("percussive")
        plt.grid()
        plt.tight_layout()      





    else:         
        onset_envp = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)       # compute the onset envelope of the music signal 
        tempogram = librosa.feature.tempogram(onset_envelope=onset_envp, sr=sr, hop_length=hop_length)   # compute the tempogram
        
        global_autoCorr = librosa.autocorrelate(onset_envp, max_size=tempogram.shape[0])
        global_autoCorr = librosa.util.normalize(global_autoCorr)
        # Estimate the global tempo for display purposes
        
        tempo = librosa.beat.tempo(onset_envelope=onset_envp, sr=sr,
                                   hop_length=hop_length)[0]

        print tempo
        figure()
        plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        plt.legend(frameon=True, framealpha=0.75)
        x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                        num=tempogram.shape[0])
        plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
        plt.plot(x, global_autoCorr, '--', alpha=0.75, label='Global autocorrelation')
        plt.xlabel('Lag (seconds)')
        plt.axis('tight')
        plt.legend(frameon=True)
        figure()

        # PLOT IN BPM [Beats Per Second]
        freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
        plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                     label='Mean local autocorrelation', basex=2)
        plt.semilogx(freqs[1:], global_autoCorr[1:], '--', alpha=0.75,
                     label='Global autocorrelation', basex=2)
        plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
                    label='Estimated tempo={:g}'.format(tempo))
        plt.legend(frameon=True)
        plt.xlabel('BPM')
        plt.axis('tight')
        plt.grid()
        plt.tight_layout()


    if plotFlag:   
        plt.show()
