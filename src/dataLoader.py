import six.moves.cPickle as pickle
import gzip

import numpy

import theano
import theano.tensor as T

from features import *

def load_data(dataset,only_train=False):
    if dataset == 'mnist.pkl.gz':
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path

        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            from six.moves import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, dataset)

        print('... loading data')

        # Load the dataset
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

    elif dataset=='tzanetakis':
        # Do Tzanetakis instructions here
        X, y, song_index = get_features()
        X = (X - X.min())/(X.max()-X.min())
        
        data = split_data(X, y, song_index)
        del X
        del y
        del song_index
        train_idx = [i for i in range (data['X_train'].shape[0])]
        np.random.shuffle(train_idx)
        data['X_train'] = data['X_train'][train_idx]
        data['y_train'] = data['y_train'][train_idx]

        #testing
        test_idx = [i for i in range (data['X_test'].shape[0])]
        np.random.shuffle(test_idx)
        data['X_test'] = data['X_test'][test_idx]
        data['y_test'] = data['y_test'][test_idx]

        #val
        val_idx = [i for i in range (data['X_val'].shape[0])]
        np.random.shuffle(val_idx)
        data['X_val'] = data['X_val'][val_idx]
        data['y_val'] = data['y_val'][val_idx]
    

        train_set = (data['X_train'], data['y_train'])
        test_set = (data['X_test'],data['y_test'])
        valid_set = (data['X_val'],data['y_val'])
        del data


    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')
    if not only_train:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set) 
    else:
        test_set_x, test_set_y = None, None
        valid_set_x, valid_set_y = shared_dataset(valid_set)        
    
    train_set_x, train_set_y = shared_dataset(train_set)
    del train_set
    del valid_set
    del test_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
