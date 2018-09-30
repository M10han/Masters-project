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
import time


song_labels_dic = { 'blues': 0, 'classical': 1, 'country': 2,
                        'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                        'pop': 7, 'reggae': 8, 'rock': 9}


def compute_features(data_loc='../data/genres/'):
    file_names = glob.glob(data_loc + '*/*.au')
    file_names.sort()


    assert len(file_names) == 1000, "ERROR: Couldn't read files properly. Is your data_loc correct?"

    

    # Setup some vars
    n_fft = 1024

    X = []
    genres_list = list(song_labels_dic.keys())
    genres_list.sort()
    genre_flag = 0

    if not os.path.exists('../ckpt'):
        os.makedirs('../ckpt')

        for file in file_names:
            song, _ = lc.load(file)
            song_dft = np.abs(lc.stft(song, n_fft = n_fft))
            X.append(song_dft)
            if len(X) == 100:
                print('Writing: ' + genres_list[genre_flag] + '.pkl file...')
                with open('../ckpt/' + genres_list[genre_flag] + '.pkl', 'wb') as f:
                    pickle.dump(X, f)
                X = []
                genre_flag = genre_flag + 1
    
    return True

def get_features():
    start = time.time()
    X, y, song_index = np.zeros((2586938,513), dtype=np.float32), np.zeros((2586938,), dtype=np.float32), np.zeros((2586938,), dtype=np.uint16)

    files = glob.glob('../ckpt/*.pkl')
    files.sort()
    X_index = 0
    song_index_counter = 0
    for i in range(len(files)):
        print("Running Op on " + files[i])
        with open(files[i], 'rb') as f:
            genre = pickle.load(f)
        song_counter = 1
        for song in genre:
            song_index_counter = song_index_counter + 1
            print(files[i] + " : " + str(song_counter))
            song_counter = song_counter+1
            song = song.T.astype(np.float32)
            for sample in song:
                X[X_index, :] = sample
                y[X_index] = i
                song_index[X_index] = song_index_counter
                X_index = X_index + 1
    print(song_index_counter)
    assert song_index_counter == 1000, "ERROR: Index mismatch"
    assert X.shape[0] == y.shape[0], "ERROR: X, y dimension mismatch"
    print('Time taken: ',time.time()-start, ' seconds.')
    return X, y, song_index

def split_data(X, y, song_index):
    np.random.seed(1234)  # for reproducibility
    # Training: 50%, Validation: 20%, Test: 30%
    song_indices = np.asarray([x for x in range(1,101)])
    train_indices, val_indices, test_indices = [], [], []
    for start_index in range(0,1000,100):
        np.random.shuffle(song_indices)
        new_song_indices = song_indices + start_index
        for i in range(50):
            train_indices.append(new_song_indices[i])
        for i in range(50,70):
            val_indices.append(new_song_indices[i])
        for i in range(70,100):
            test_indices.append(new_song_indices[i])

    print(song_index,song_index.shape)
    # print(train_indices,test_indices,val_indices, len(train_indices)+len(test_indices)+len(val_indices))
    X_indices = []
    for train_index in train_indices:
        X_indices = X_indices + list(np.where(song_index==train_index)[0])
    X_train, y_train, song_index_train = X[X_indices], y[X_indices], song_index[X_indices]

    X_indices = []
    for test_index in test_indices:
        X_indices = X_indices + list(np.where(song_index==test_index)[0])
    X_test, y_test, song_index_test = X[X_indices], y[X_indices], song_index[X_indices]
    
    X_indices = []
    for val_index in val_indices:
        X_indices = X_indices + list(np.where(song_index==val_index)[0])
    X_val, y_val, song_index_val = X[X_indices], y[X_indices], song_index[X_indices]


    return {'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val,
            'song_index_train': song_index_train,
            'song_index_test': song_index_test,
            'song_index_val': song_index_val}

if __name__=='__main__':
    compute_features()