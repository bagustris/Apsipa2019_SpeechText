# run this to unuse cuda
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# script to save feature
import numpy as np
import pickle 
import audiosegment 
from helper import *
from features import *

def calculate_features(frames, freq, options):
    window_sec = 0.2
    window_n = int(freq * window_sec)

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        
        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
        
framerate = 16000
data_path = '/media/bagus/data01/dataset/IEMOCAP_full_release/'
with open(data_path + 'data_collected.pickle', 'rb') as handle:
    data2 = pickle.load(handle)


## outlier removal
# delete noisy speech
#speech = np.delete(speeches, (1061, 1430, 1500, 1552, 1566, 1574, 1575, 1576, 1862, 1863, 1864, 1865, 1868, 1869,
#                              1875, 1878, 1880, 1883, 1884, 1886, 1888, 1890, 1892, 1893, 1930, 1931, 1932, 1969,
#                              1970, 1971, 1975, 1976, 1977, 1979, 1980, 1981, 1984, 1985, 1986, 1987, 1988, 1989, 
#                              1990, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2002, 2003, 2076, 2106, 2110,
#                              2177, 2178, 2179, 2180, 2206, 2241, 2242, 2243, 2245, 2246, 2253, 2254, 2262, 2263, 
#                              2357, 2358, 2359, 2362, 2368, 2373, 2374, 2418, 2523, 2525, 2526, 2534, 2539, 2542,
#                              2549, 2552, 2553, 2554, 2555, 2556, 2561, 2562, 2563, 2564, 2578, 2670, 2671, 2672, 
#                              2692, 2694, 2695, 2728, 2733, 2889, 2890, 3034, 3304, 3511, 3524, 3525, 3528, 3655, 
#                              3802, 3864, 3930, 4038, 4049, 4051, 4061, 4193, 4241, 4301, 4302, 4307, 4569, 4570), 0)

# doing silence removal, only threshold under 0.01 wokrks
voiced_feat = []
duration = 0.1
threshold = 0.001

for i in range(len(speech)):
    x_head = data2[i]['signal']
    #x_head = speech[i]
    seg = audiosegment.from_numpy_array(speech[i], framerate)
    seg = seg.filter_silence(duration_s=duration, threshold_percentage=threshold)
    st_features = calculate_features(seg.to_numpy_array(), framerate, None)
    #st_features = calculate_features(x_head, framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
    voiced_feat.append(st_features.T)
    print(i)

voiced_feat = np.array(voiced_feat)
voiced_feat.shape
np.save('voiced_feat_file_01_007.npy', voiced_feat)
