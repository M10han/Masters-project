from __future__ import print_function
import numpy as np
import glob
import pickle
import time

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

import sys
sys.path.append("./models")
sys.path.append("./base")

import tensorflow as tf
from models.dbn import DBN



song_labels_dic = { 'blues': 0, 'classical': 1, 'country': 2,
                    'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                    'pop': 7, 'reggae': 8, 'rock': 9 }

print('Loading data...')

start = time.time()
X, y = np.zeros((2586938,513), dtype=np.float32), np.zeros((2586938,10), dtype=np.float32)
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
            y[X_index, i] = 1
            X_index = X_index + 1

assert X.shape[0] == y.shape[0], "ERROR: X, y dimension mismatch"
print('Time taken: ',time.time()-start, ' seconds.')
# Scaling
X = X/X.max()

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)
del X
del y
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.285, random_state=0)

# Training
print('Training...')
start = time.time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
classifier = DBN(output_act_func='softmax', hidden_act_func='relu', loss_fuc='cross_entropy', use_for='classification', dbn_lr=0.001, dbn_epochs=474, dbn_struct=[513, 50, 50, 50, 10], rbm_v_type='bin', rbm_epochs=10, batch_size=256, cd_k=1, rbm_lr=0.001)
classifier.build_model()
classifier.train_model(X_train, Y_train, sess)
print('[TRAINING] Time took: ', time.time()-start, ' seconds.')
# Save the model
#saver = tf.train.Saver({"classifier":classifier})
#save_path = saver.save(sess, './checkpoint/model.ckpt')


# Restore it
#classifier.initializer.run()
#classifier = saver.restore(sess, './checkpoint/model.ckpt')

# Test
Y_pred = list()
print("[Test data...")
for i in range(Y_test.shape[0]):
    print(">>>Test fault {}:".format(i))
    Y_pred.append(classifier.test_model(X_test[i].reshape((1,513)), Y_test[i].reshape((1,10)), sess))

Y_pred = np.asarray(Y_pred,dtype=np.float32)
Y_pred = np.reshape(Y_pred,(Y_pred.shape[0],Y_pred.shape[2]))
 
print('Done.\nAccuracy: %f' % accuracy_score(np.argmax(Y_test,axis=1), np.argmax(Y_pred,axis=1)))

