## Computation of Tempo Information

Description of how tempo information can be extracted both from music and EEG signals. 
- we transform a signal into a tempogram _T_ which is a time-tempo representation of a signal 
- A tempogram reveals periodicities in a given signal, similar to a spectrogram. 


### Onset Detection 
- Finding start times of  perceptually relevant acoustic events in music signal  [_Note that we use perceptually important events and not just beats_]
- Onset is the time position where a note is played  

	**Method for onset detection in audio signals:**
		**Energy Method** 
			- Amplitude squaring 
			- Windowing (energy envelope)
			- Differentiation (capturing changes)
			- Half wave recitification (only energy increases are relevant for note onsets)
			- Peak Picking 
-The above described energy method is relevant only for percussive music. Many instruments like strings have weak note onsets. Also no increase may be observed in case of complex sound mixture. Hence we need more refined methods which capture _changes of spectral content_ , _changes of pitch_ , _changes of harmony_ .  		
		**Spectral Method**
			- Spectogram (Frequency Vs time)  
			- Logarithmic Compression [Y = log(1 + C.X) ]
			- Differentiation [1st order temporal difference, only positive changes in spectral content are captured]
			- Accumulation [framewise accumulation of all +ive intensity changes]
			- Normalization
			- Peak Picking    


Note : 
- Aspects concerning pitch, harmony, or timbre are captured by spectogram. Allows to detect local energy changes in certain frequency ranges.   		
- Logarithimic Compression accounts for dynamic range compression, enhancement of low-intensity values and high frequency components
- In peak picking, Usage of local thresholding techniques since in general many spurious peaks exist. 


## Tempogram for EEG signals  

Steps : 
- Aggregate the 64 EEG channels into one signal.
*Note* : Note that there is a lot of redundancy in these channels. This redundancy can be exploited to improve the SNR.
Use the channel aggregation filter. It was learned as part of a convolutional neural network (CNN) during a previous experiment attempting to recognize the stimuli from the EEG recordings. A technique called “similarity-constraint encoding” (SCE) was applied that is motivated by earlier work on learning similarity measures from relative similarity constraints. [Reference](https://www.audiolabs-erlangen.de/content/05-fau/professor/00-mueller/03-publications/2016_StoberPM_BeatEEG_ISMIR.pdf)

- From the aggregated EEG signal, we then compute a novelty curve. Here, opposed to the novelty computation for the audio signal, we assume that the beat periodicities we want to measure are already present in the time-domain EEG signal.We therefore interpret the EEG signal as a kind of novelty curve.

- As pre-processing, we normalize the signal by subtracting a moving average curve. This ensures that the signal is centered around zero and low frequent components of the signal are attenuated. 

- The resulting signal is then used as a novelty curve to compute an EEG tempogram that reveals how dominant different tempi are at a given time point in the EEG signal. 


## Tempo Histograms

Extract a single tempo value from the audio and EEG tempograms. 
Steps: 
- Aggregate the time-tempo information over the time by computing a *tempo histogram*. A value H(τ)in the tempo histogram indicates how present a certain tempo τ is within the entire - signal. 
- Analyse the highest peak in the tempo histograms of EEG and audio files. Both the peaks should be nearly same. 

