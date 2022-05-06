reuniao 29/04
- 22 de julho, terminar a implementacao.
- texto todo pronto ate 19 de agosto, pois sao 10 dias antes de acabar o periodo 
- 29 de agosto acaba o periodo
- pensar em uma aplicacao simples, para nao ficar apenas na parte teorica
- reestruturar o titulo, capitulos e subseccoes 
- voltar a andar com o texto, para nao deixar de ultima hora



### main folder:

[] moddecomp (WIP)
Decomposes a signal into a collection of low-frequency modulator
envelopes and their corresponding high-frequency carriers.
Inputs: X, Fs, subband frequency boundaries, downsampling factor
Outputs: M - An array of modulator signals, C - An array of complex-exponential carrier signals.

[] modfilter
Apply the modulation filtering over an input signal.
Inputs: X, Fs, subband frequency boundaries, downsampling factor
Outputs: Y, M, C

[] modop_shell
A template function for designing your own modulation 
analysis/modification/synthesis routines.

[] modspecgram
Legacy version of the modulation spectrum. use modspectrum().

[] modspectrum
Plots the joint-frequency modulation spectrum of a signal.

[] modsynth
Recombines modulator and carrier signals to form an audio signal.

[] tutorial1_modulationFrequency

[] tutorial2_speechAnalysis

[] tutorial3_functions

[] tutorial4_modulationFiltering

### demod:

[x] carrier2if
- tests [x]
Extracts the instantaneous frequency track(s) from
 the phase of complex-exponential carrier signal(s).

[] detectpitch (WIP)
Detects the fundamental frequency of a signal,
assuming a harmonic signal model.

[x] if2carrier
- tests [x]
Converts instantaneous frequency track(s) into
complex-exponential carrier signal(s).

[x] moddecompcog
- tests []
Coherently demodulates subband signals using carriers 
based on time-varying spectral center-of-gravity.

[] moddecompharm
 Coherently demodulates a signal based on a pitch 
estimate and an assumption of harmonic carriers.

[x] moddecomphilb
- tests []
 Incoherently demodulates subband signals using magnitude Hilbert envelopes.


[] modrecon
Reconstructs subband signals from modulator/carrier pairs.


[] modreconharm
 Reconstructs a signal from modulators and harmonic carriers.

[] viewcarriers
 Overlays carrier frequencies with a spectrogram of
the original audio signal for comparison.

### filter:

[x] designfilter
- tests [x]
 Designs a narrowband multirate FIR filter.

[x] filterfreqz
- tests []
 Plots the frequency response of a multirate filter.

[x] narrowbandfilter
- tests []
Performs a multirate filter operation.

### filterbank:

[x] cutoffs2fbdesign 
- tests [x]
Generates filterbank design parameters from a list
 of subband cutoff frequencies.

[x] designfilterbank
- tests [x]
Designs a filterbank with arbitrary subband
spacing and bandwidths.

[] designfilterbankgui
Runs a GUI for designing a filterbank with 
equispaced subbands and near-perfect synthesis.

[x] designfilterbankstft
- tests []
Designs a filterbank with equispaced subbands
based on the short-time Fourier transform

[] filterbankfreqz
Plots the frequency responses of the subbands in a
filterbank design.

[] filterbanksynth
Recombine subband signals to form an audio signal.

[x] filtersubbands
- tests [x]
Use a filterbank design to extract subband
signals from an audio signal.
