import audiosegment
import pickle
import numpy as np
from tensorflow import set_random_seed

set_random_seed(2)
np.random.seed(1)

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, Embedding, Conv1D, Conv2D, Dropout, MaxPooling2D, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from features import *
from helper import *
from attention_helper import *

#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import StratifiedKFold, cross_val_score

framerate = 16000
data_path = '/media/bagus/data01/dataset/IEMOCAP_full_release/'
with open(data_path + 'data_collected.pickle', 'rb') as handle:
    data2 = pickle.load(handle)

# next load features
voiced_feat = np.load('voiced_feat_file_001_001.npy')

# Check label
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])

Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])
        
Y = label_binarize(Y,emotions_used)
# print shape, 4 emotion each column, 4936 label
Y.shape

emo = np.delete(Y, (1061, 1430, 1500, 1552, 1566, 1574, 1575, 1576, 1862, 1863, 1864, 1865, 1868, 1869,
                              1875, 1878, 1880, 1883, 1884, 1886, 1888, 1890, 1892, 1893, 1930, 1931, 1932, 1969,
                              1970, 1971, 1975, 1976, 1977, 1979, 1980, 1981, 1984, 1985, 1986, 1987, 1988, 1989, 
                              1990, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2002, 2003, 2076, 2106, 2110,
                              2177, 2178, 2179, 2180, 2206, 2241, 2242, 2243, 2245, 2246, 2253, 2254, 2262, 2263, 
                              2357, 2358, 2359, 2362, 2368, 2373, 2374, 2418, 2523, 2525, 2526, 2534, 2539, 2542,
                              2549, 2552, 2553, 2554, 2555, 2556, 2561, 2562, 2563, 2564, 2578, 2670, 2671, 2672, 
                              2692, 2694, 2695, 2728, 2733, 2889, 2890, 3034, 3304, 3511, 3524, 3525, 3528, 3655, 
                              3802, 3864, 3930, 4038, 4049, 4051, 4061, 4193, 4241, 4301, 4302, 4307, 4569, 4570), 0)

# model3 attention based
def attention_model(optimizer='rmsprop'):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(100, 34)))
    model.add(Bidirectional(AttentionDecoder(128,128)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model1 = attention_model()
model1.summary()

# train data
hist = model1.fit(voiced_feat, Y, batch_size=32, epochs=30, verbose=1, shuffle=True,
                 validation_split=0.2)

acc1 = hist.history['val_acc']
print(max(acc1))

#import matplotlib.pyplot as plt
#plt.plot(acc1)
#plt.savefig('accuracy_006_001_split03_epoch_50.pdf')
#seed = 7 
#np.random.seed(seed)

#model = KerasClassifier(build_fn=speech_model1, epochs=100, batch_size=16, verbose=0)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, voiced_feat, Y, cv=kfold)
#print(result.mean())
