import numpy as np
from scipy.signal import find_peaks

def detect_jump(left_foot, right_foot, prominences = 0.2, width = 100):
    """
    > It finds the peaks in the left and right foot data, and then finds the peaks that are close to
    each other in time and have a large prominence
    
    Args:
      left_foot: the left foot's height data
      right_foot: the right foot data
      prominences: the minimum height of the peak, relative to the surrounding troughs.
      width: the width of the window to look for the minimum value in. Defaults to 100
    """
    lf_height = np.asarray(left_foot[:50]).mean()
    rf_height = np.asarray(right_foot[:50]).mean()
    left_foot = np.asarray(left_foot- lf_height)
    right_foot = np.asarray(right_foot- rf_height)

    l_peaks, lprop = find_peaks(left_foot, distance=80, height=0.05, prominence=0.05)
    r_peaks, rprop = find_peaks(right_foot, distance=80, height=0.05, prominence=0.05)
    jumps = []
    j = 0
    for i, lp in enumerate(l_peaks):
        if j >= len(r_peaks):
            break

        while r_peaks[j] < lp and abs(r_peaks[j] - lp) >= 5:
            if j+1 < len(r_peaks):
                j += 1
            else:
                break

        # two peaks at the same time(< 0.05s), distance < 0.05m, prominences > 0.2m
        if abs(r_peaks[j] - lp) < 5 :  
            peaks_dist = abs(lprop['peak_heights'][i] - rprop['peak_heights'][j])
            # peak_prominences =  (lprop['prominences'][i] + rprop['prominences'][j])/2
            wl = max(0, lp - width)
            wr = min(lp+width, left_foot.shape[0])
            peak_prominences = (lprop['peak_heights'][i] - left_foot[wl:wr].min() + rprop['peak_heights'][j] - right_foot[wl:wr].min())/2

            if peaks_dist < 0.05 and peak_prominences > prominences:
                jumps.append(lp)
    return l_peaks, r_peaks, jumps