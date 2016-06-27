from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y

def butter_low_pass_filter(data, sample_rate, low_cut):
    low = lowcut / (0.5 * sample_rate)
    b, a = butter(order, low, btype='lowpass')
    y = lfilter(b, a, data)
    return y

def butter_high_pass_filter(data, sample_rate, high_cut):
    high = highcut / (0.5 * sample_rate)
    b, a = butter(order, high, btype='highpass')
    y = lfilter(b, a, data)
    return y
