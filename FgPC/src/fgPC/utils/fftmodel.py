r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    The class FFTModel is used to calculate the FFT of a time series
                as well as to find the highest peaks in the FFT.

"""
import numpy as np

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

class FFTModel():

    def __init__(self, threshold = 0.01, distance = None):
        self.threshold = threshold
        self.distance = distance


    def calculateFFT(self, data, t):
        r"""
        Method calculates the FFT of the time series

        Parameters
        ----------
        data : array
            array with the data corresponding to the time points
        t : array
            array with the time points

        Returns
        ----------
        List of:
        amplitudesOrig : array
            array with the original amplitudes
        amplitudes : array
            array with the norm of amplitudes
        freqs : array
            array with the frequencies
        """

        amplitudesOrig = 2*rfft(data)/data.shape[0]
        freqs = rfftfreq(data.shape[0], (t[-1]-t[0])/data.shape[0])*2*np.pi

        amplitudes = np.abs(amplitudesOrig)
        amplitudes[0] = amplitudes[0]/2

        return [amplitudesOrig, amplitudes, freqs]
    
    def performFFT(self, data, t, highFreq = 10, phaseshift = None):
        r"""
        Method calculate the FFT of the data as well as finds
        the highest peaks in the FFT

        Parameters
        ----------
        data : array
            array with the data corresponding to the time points
        t : array
            array with the time points
        highFreq : float optional
            highest frequency to be considered
        
        Returns
        ----------
        fftData : list
            list with arrays of the frequencies and amplitudes
        peakVal : list
            list with arrays of the frequencies and amplitudes of the peaks
            First is the frequency
            Seccond the peak amplitudes sorted by constant, cos and sin
            Third the peak amplitudes in absolute values
        """

        dataTrunc = data
        tTrunc = t

        fftData = self.getAmplitudeFrequency(dataTrunc, tTrunc, highFreq)

        peakInd, _ = find_peaks(fftData[1],
                                height=self.threshold * np.max(fftData[1]),
                                distance=self.distance)
        
        peakInd = np.append(0, peakInd)
        peakVal = [fftData[0][peakInd], fftData[2][peakInd], fftData[1][peakInd]]

        # make phase shift
        # magnitudes = np.abs(peakVal[1])
        # phases = np.angle(peakVal[1])
        # if phaseshift is None:
        #     phaseshift = phases[1]
        # adjusted_phases = phases -phaseshift
        # adjusted_amplitudes = magnitudes * np.exp(1j* adjusted_phases)
        # peakVal[1] = adjusted_amplitudes

        peakVal[1] = self.getAmplitude(peakVal[1])

        return fftData, peakVal, phaseshift
    
    # def truncate_signal(self,signal,t):
    #     r"""
    #     Method that truncates the signal to a length that is a multiple of the period

    #     Parameters
    #     ----------
    #     signal : array
    #         array with the signal
    #     t : array
    #         array with the time points

    #     Returns
    #     ----------
    #     truncated_signal : array
    #         array with the truncated signal
    #     truncated_t : array
    #         array with the truncated time points
    #     """

    #     autocorr = np.correlate(signal, signal, mode='full')
    #     peaks, _ = find_peaks(autocorr)
    #     peak_distances = np.diff(peaks)
        
    #     period = int(np.median(peak_distances))
        
    #     num_periods = len(signal) // period
    #     truncated_length = num_periods * period
    #     truncated_signal = signal[:truncated_length]
    #     truncated_t = t[:truncated_length]
        
    #     return truncated_signal, truncated_t


    def getAmplitudeFrequency(self, data, t, highFreq = 10):
        r"""
        Method that performs an FFT and returns the Amplitude Frequency response

        Parameters
        ----------
        data : array
            array with the data corresponding to the time points
        t : array
            array with the time points

        Returns
        ----------
        freqs : array
            array with the frequencies
        amplitudes : array
            array with the amplitudes
        """
        
        amplitudesXOrig, amplitudes, freqs = self.calculateFFT(data, t)

        freqs = freqs[freqs < highFreq]
        amplitudes = amplitudes[0:len(freqs)]
        return [freqs, amplitudes, amplitudesXOrig] # divide by 1000 to get Hz not mHz

    # def saveAmplitude(self,data, fileName):
    #     r"""
    #     Method that saves the amplitude and frequencies in a file

    #     Parameters
    #     ----------
    #     data : array
    #         array with the data
    #     fileName : string
    #         name of the file
    #     """

    #     fileStr = fileName + ".npy"
    #     np.save(fileStr, data)

    def getAmplitude(self,complAmpl):
        r"""
        Method that returns the amplitude from the complex amplitude
        in sine and cosine form
        
        Parameters
        ----------
        complAmpl : array
            array with the complex amplitude
            
        Returns
        ----------
        amplitude : array
            array with the amplitudes
            first the constant, then cosine and sine
        """

        constAmp = np.real(complAmpl[0])/2
        cosAmp = np.real(complAmpl[1:])
        sinAmp = -np.imag(complAmpl[1:])

        amplitude = np.concatenate(([constAmp], cosAmp, sinAmp))

        return amplitude
    
    # def calculateSignalFromFFT(self, amplitudes, freqs, t):
    #     r'''method that calculates the signal from the FFT
        
    #     Parameters
    #     ----------
    #     amplitudes : array
    #         array with the amplitudes in constant, cos and sin
    #     freqs : array
    #         array with the frequencies
    #     t : array
    #         array with the time points
        
    #     Returns
    #     ----------
    #     signal : array
    #         array with the signal
    #     signal_v : array
    #         array with the signal velocity
    #     '''

    #     h = freqs.shape[0]-1

    #     cosAmp = amplitudes[1:h+1]
    #     sinAmp = amplitudes[h+1:]

    #     signal = np.zeros(t.shape[0])
    #     signal_v = np.zeros(t.shape[0])

    #     signal += amplitudes[0]
        
    #     for i in range(h):
    #         signal += cosAmp[i]*np.cos(freqs[i+1]*t) + \
    #                   sinAmp[i]*np.sin(freqs[i+1]*t)
    #         signal_v += -freqs[i+1]*cosAmp[i]*np.sin(freqs[i+1]*t) + \
    #                     freqs[i+1]*sinAmp[i]*np.cos(freqs[i+1]*t)
    #     return signal, signal_v