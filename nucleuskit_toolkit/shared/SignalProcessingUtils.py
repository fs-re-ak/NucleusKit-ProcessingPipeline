import math
import numpy as np


# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def findNearestIdx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if (idx > 0 and (idx == len(array)) or (math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]))):
        return idx-1
    else:
        return idx


def interpolateNan(xs, ys, valuedIdx):

    if valuedIdx[0]!=0:
        print("first index need to be valued")
        return ys

    for i in range(valuedIdx.shape[0]-1):

        if((xs[valuedIdx[i+1]] - xs[valuedIdx[i]]) > 0):
            slope = (ys[valuedIdx[i+1]] - ys[valuedIdx[i]])/(xs[valuedIdx[i+1]] - xs[valuedIdx[i]])
        else:
            slope = 0
        zero = ys[valuedIdx[i]]

        for j in range(valuedIdx[i]+1, valuedIdx[i+1]):
            ys[j] = zero+slope*(xs[j]-xs[valuedIdx[i]])

    for i in range(valuedIdx[-1],ys.shape[0]):
        ys[i] = ys[valuedIdx[-1]]
    return ys



# Old implementation used in pop7Emotions previously
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx


def smooth(x, window_len=11, window='hanning'):
    """smooth the recording_data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len < 3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len/2):-int(window_len/2)]


def movingAverage(a, n=3, step=1) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    return ret[::step]