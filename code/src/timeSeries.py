
import numpy as np

def autocorr(x, lags):
    xcorr = np.correlate(x-x.mean(), x-x.mean(), 'full')
    xcorr = xcorr[xcorr.size//2:]/xcorr.max()
    answer = xcorr[:lags+1]
    return answer

