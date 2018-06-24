from __future__ import print_function
import argparse
from features import *
from models.dbn import DBN
import tensorflow as tf

def set_args():
    parser = argparse.ArgumentParser('Arguments for running DBN on songs database')
    parser.add_argument('--compute-features', type=bool, default=False, 
                        help='Argument to set pre-computed features or compute from scratch')
    

    args = parser.parse_args()
    return args

def train(X_train, Y_train):
    # Training
    print('Training...')
    start = time.time()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    classifier = DBN(output_act_func='softmax', 
                     hidden_act_func='relu', 
                     loss_fuc='cross_entropy', 
                     use_for='classification', 
                     dbn_lr=0.001, 
                     dbn_epochs=474, 
                     dbn_struct=[513, 50, 50, 50, 10], 
                     rbm_v_type='bin', 
                     rbm_epochs=10,
                     batch_size=256, 
                     cd_k=3, 
                     rbm_lr=0.0001)
    classifier.build_model()
    classifier.train_model(X_train, Y_train, sess)
    print('[TRAINING] Time took: ', time.time()-start, ' seconds.')

def test(X_test,Y_test):
    Y_pred = list()
    print("[Test data...")
    for i in range(Y_test.shape[0]):
        print(">>>Test fault {}:".format(i))
        Y_pred.append(classifier.test_model(X_test[i].reshape((1,513)), Y_test[i].reshape((1,10)), sess))

    Y_pred = np.asarray(Y_pred,dtype=np.float32)
    Y_pred = np.reshape(Y_pred,(Y_pred.shape[0],Y_pred.shape[2]))
    
    print('Done.\nAccuracy: %f' % accuracy_score(np.argmax(Y_test,axis=1), np.argmax(Y_pred,axis=1)))

def main():
    args = set_args()
    if args.compute_features:
        # Call pre-processing functions
        status = compute_features()
        if not status: return

    X, y, song_index = get_features()
    X = (X - X.min()) / (X.max() - X.min())
    data = split_data(X, y, song_index)
    del X
    del y
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    train(X_train,y_train)
    test(X_val,y_val)
    




if __name__=='__main__':
    main()