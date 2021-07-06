# ter using word vector
# this code is not part of APSIPA paper
# need to download word2vector first


import numpy as np
import os
import sys

import wave
import copy
import math

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, merge, Embedding, Convolution1D, Dropout, Bidirectional
from attention_helper import *

from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from gensim.models import KeyedVectors

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = '/media/bagustris/bagus/dataset/IEMOCAP_full_release/'
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
# framerate = 16000

import pickle
data = data_path + 'data_collected_full.pickle'
with open(data, 'rb') as handle:
    data2 = pickle.load(handle)

text = []

for ses_mod in data2:
    text.append(ses_mod['transcription'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

MAX_SEQUENCE_LENGTH = max([len(t) for t in token_tr_X])
x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

# read word vector, takes long time, better to save as npy
path = '/media/bagustris/atmaja/github/models/'
w2v_model = 'GoogleNews-vectors-negative300.bin'
embeddings_model = KeyedVectors.load_word2vec_format(
                    path + w2v_model, binary=True)

word_index = tokenizer.word_index
num_words = len(word_index)

# dimension = 300
nb_words = len(word_index) +1
embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
    if word in embeddings_model.index2word:
        embedding_matrix[i] = embeddings_model[word]

# read labels
Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])

Y = label_binarize(Y,emotions_used)

print(Y.shape)

# build model with Attention model
def attention_model(optimizer='rmsprop'):
    model = Sequential()
    model.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
    model.add(LSTM(128, return_sequences=True))
    model.add(AttentionDecoder(128, 128))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model4 = attention_model()
model4.summary()

# train model
model4.fit(x_train_text, Y, batch_size=16, epochs=25,
            verbose=1, validation_split=0.2)

# print training accuracy
acc4=max(hist4.history['val_acc'])
print(acc4)



