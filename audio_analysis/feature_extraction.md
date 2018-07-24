# Music Feature Extraction. 

### Music features we know:

- rhythm 
- similarity
- genre
- melody
- harmony 
- dynamics

- loudness
- timbre
- pitch
- spectrum 
- duration 
- energy 
- frequency
- Tone color 

**Techniques for music feature extraction:**


- **melspectrogram:** Compute a Mel-scaled power spectrogram
- **mfcc:** Mel-frequency cepstral coefficients
- **chorma-stft:** Compute a chromagram from a waveform or power spectrogram
- **spectral_contrast:** Compute spectral contrast
- **tonnetz:** Computes the tonal centroid features (tonnetz)


## Spectrogram

- Speech signal represented as a sequence of spectral vectors. The spectral vector is a distribution of frequencies. It indicates how strongly is a particular frequency present in the given time frame. Each band of time frame has a spectral vector. A spectral vector is computed by performing FFT on the audio signal in the time frame. 


## Cepstral Analysis

**Hilbert Transform on Music Signals**

The Hilbert transform calculates the "analytic" signal, i.e. it calculates a matching imaginary part to a real signal by shifting the phase by 90 degrees in the frequency domain. It's reputation of calculating the "envelope" comes mainly from communication technology. It works very well with a narrow band signals, like amplitude or frequency or phase modulated signals. It's based on the simple fact that the analytic signal of a sine wave is a complex exponentional, the magnitude of which is 1. So the hilbert transform (or to be precise calculating the magnitude of the analytic signal) will eliminate the carrier quite well and leaves you with the modulation signal, or "envelope".

For broad band music, this is mostly not the case and the analytic signal isn't particularly helpful. A cymbal crash is basically a noise burst. The hilbert transform of this is just another noise burst and so is the magnitude of the analytic signal. You can't eliminate a carrier, if there isn't one to start with.

IF you are looking for something that roughly gives you "perceived loudness as a function of time", there many different ways to do this from simple lossy peak detectors all the way to sophisticated perceptual algorithms including cochlea models, binaural & temporal masking, nerve excitation models etc. It really depends on your application.

