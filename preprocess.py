"""
This python script pre-processes the Tzanetaki's genre dataset
and generates the [X,y] labels required for any classification
purposes


"""

from __future__ import print_function
import numpy as np
import librosa.core as lc
import glob
import os
import pickle
import h5py

# Get the directory for the data location
# Replace the data_loc with the relative location
# to Genres dataset
data_loc = './data/genres/'
file_names = glob.glob(data_loc + '*/*.au')
file_names.sort()


assert len(file_names) == 1000, "ERROR: Couldn't read files properly. Is your data_loc correct?"

song_labels_dic = { 'blues': 0, 'classical': 1, 'country': 2,
                    'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                    'pop': 7, 'reggae': 8, 'rock': 9}

# Setup some vars
sampleRate = 22050
n_fft = 1024

X = []
genres_list = list(song_labels_dic.keys())
genres_list.sort()
genre_flag = 0

if not os.path.exists('./ckpt'):
    os.makedirs('./ckpt')

    for file in file_names:
        song, _ = lc.load(file)
        song_dft = np.abs(lc.stft(song, n_fft = n_fft))
        X.append(song_dft)
        if len(X) == 100:
            print('Writing: ' + genres_list[genre_flag] + '.pkl file...')
            with open('./ckpt/' + genres_list[genre_flag] + '.pkl', 'wb') as f:
                pickle.dump(X, f)
            X = []
            genre_flag = genre_flag + 1

import sys
sys.modules[__name__].__dict__.clear()

import glob
import pickle
import h5py
import numpy as np

X, y = np.zeros((2586938,513), dtype=np.float32), np.zeros((2586938,), dtype=np.float32)
files = glob.glob('./ckpt/*.pkl')
files.sort()
X_index = 0
for i in range(len(files)):
    print("Running Op on " + files[i])
    with open(files[i], 'rb') as f:
        genre = pickle.load(f)
    song_counter = 1
    for song in genre:
        print(files[i] + " : " + str(song_counter))
        song_counter = song_counter+1
        song = song.T.astype(np.float32)
        for sample in song:
            X[X_index, :] = sample
            y[X_index] = i
            X_index = X_index + 1

assert X.shape[0] == y.shape[0], "ERROR: X, y dimension mismatch"


print('Saving data.h5 ...')
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X', data = X)
h5f.create_dataset('y', data = y)
h5f.close()