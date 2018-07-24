# Cepstrum Analysis

I am using an audio file which has the fundamental frequency of 440Hz and is sampled at 44100Hz.  

**Time Series plot of the audio under test**

#### Time Series plot of the entire file
![alt text](../images/ceps_timeplot.png "Time Series plot of the entire file")

#### Time Series plot of the first 2000 frames
![alt text](../images/ceps_plottime_2000.png "Time Series plot of the first 2000 frames")

#### Fourier Transform of the audio file
![alt text](../images/ceps_spectrum.png "Fourier Transform of the audio file")

#### STFT of the audio file
![alt text](../images/stft_cepstrum.png "STFT of the audio file")



#### Cepstrum function: 

```python

def cepstral_analysis(y, sr, plotFlag):

    logfft = np.log(fft(y))
    ceps = ifft(logfft)
    
    return ceps
```
#### logfft
![alt text](../images/logfft.png "logfft")

#### Cepstrum
![alt text](../images/ceps.png "Cepstrum")


I am filtering the ```ceps``` using a butterworth filter with the following specifications. 
```python
order = 6
fs = sr       # sample rate, Hz
cutoff = 500  # desired cutoff frequency of the filter, Hz
```

#### Frequency Response of the Spectrum
![alt text](../images/freq_response.png "Frequency Response of the Spectrum")

#### This is the output of the filtering on the cepstrum: 

Filtered cepstrum
![alt text](../images/fitered.png "Filtered cepstrum")




