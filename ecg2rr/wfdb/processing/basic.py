import numpy as np
from scipy import signal


def resample_ann(resampled_t, ann_sample):
    """
    Compute the new annotation indices.
    Parameters
    ----------
    resampled_t : ndarray
        Array of signal locations as returned by scipy.signal.resample.
    ann_sample : ndarray
        Array of annotation locations.
    Returns
    -------
    ndarray
        Array of resampled annotation locations.
    """
    tmp = np.zeros(len(resampled_t), dtype='int16')
    j = 0
    break_loop = 0
    tprec = resampled_t[j]
    for i, v in enumerate(ann_sample):
        break_loop = 0
        while True:
            d = False
            if v < tprec:
                j -= 1
                tprec = resampled_t[j]

            if j+1 == len(resampled_t):
                tmp[j] += 1
                break

            tnow = resampled_t[j+1]
            if tprec <= v and v <= tnow:
                if v-tprec < tnow-v:
                    tmp[j] += 1
                else:
                    tmp[j+1] += 1
                d = True
            j += 1
            tprec = tnow
            break_loop += 1
            if (break_loop > 1000):
                tmp[j] += 1
                break
            if d:
                break

    idx = np.where(tmp>0)[0].astype('int64')
    res = []
    for i in idx:
        for j in range(tmp[i]):
            res.append(i)
    assert len(res) == len(ann_sample)

    return np.asarray(res, dtype='int64')

def normalize_bound(sig, lb=0, ub=1):
    """
    Normalize a signal between the lower and upper bound.
    Parameters
    ----------
    sig : ndarray
        Original signal to be normalized.
    lb : int, float, optional
        Lower bound.
    ub : int, float, optional
        Upper bound.
    Returns
    -------
    ndarray
        Normalized signal.
    """
    mid = ub - (ub - lb) / 2
    min_v = np.min(sig)
    max_v = np.max(sig)
    mid_v =  max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return sig * coef - (mid_v * coef) + mid

def smooth(sig, window_size):
    """
    Apply a uniform moving average filter to a signal.
    Parameters
    ----------
    sig : ndarray
        The signal to smooth.
    window_size : int
        The width of the moving average filter.
    Returns
    -------
    ndarray
        The convolved input signal with the desired box waveform.
    """
    box = np.ones(window_size)/window_size
    return np.convolve(sig, box, mode='same')


def get_filter_gain(b, a, f_gain, fs):
    """
    Given filter coefficients, return the gain at a particular
    frequency.
    Parameters
    ----------
    b : list
        List of linear filter b coefficients.
    a : list
        List of linear filter a coefficients.
    f_gain : int, float, optional
        The frequency at which to calculate the gain.
    fs : int, float, optional
        The sampling frequency of the system.
    
    Returns
    -------
    gain : int, float
        The passband gain at the desired frequency.
    """
    # Save the passband gain
    w, h = signal.freqz(b, a)
    w_gain = f_gain * 2 * np.pi / fs

    ind = np.where(w >= w_gain)[0][0]
    gain = abs(h[ind])

    return gain
