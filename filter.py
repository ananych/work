from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order):
    """Objective: generate butterworth bandpass filter coefficient.
        @param lowcut: The lower frequency .
        @param highcut: the upper frequency
        @param fs: sampling rate
        @param order: The order of the filter
        @return b,a coefficient of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(highcut, fs, order):
    """Objective: generate butterworth lowpass filter coefficient.
        @param highcut: the upper frequency
        @param fs: sampling rate
        @param order: The order of the filter
        @return b,a coefficient of the filter
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a


def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    """Objective: This function computes the coefficients of a finite impulse response filter.
            @param ntaps: Length of the filter (number of coefficients)
            @param lowcut: The lower frequency .
            @param highcut: the upper frequency
            @param fs: sampling rate
            @param window: Desired window to use. default hamming
            @return coefficient of the filter
    """
    from scipy.signal import firwin
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps


def butter_bandpass_filter(data, cutoff_freq, fs, order):
    """Objective: butterworth bandpass filter.
        @param data: The signal pass into filter
        @param lowcut: The lower frequency .
        @param highcut: the upper frequency
        @param fs: sampling rate
        @param order: The order of the filter
        @return the signal after filtering
    """
    b, a = butter_bandpass(cutoff_freq[0], cutoff_freq[1], fs, order=order)
    y = lfilter(b, a, data)
    return y