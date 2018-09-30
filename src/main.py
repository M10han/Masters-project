from __future__ import print_function, division
import sys
import timeit

from dataLoader import *
from dbn import DBN

from six.moves import cPickle

import keras
from keras.models import Sequential
from keras.layers import Dense

from features import *


def test_DBN():
    # datasets = load_single_data(valid=True, test=True)
    # song_index_test = datasets[3][2]
    # datasets = [datasets[1],datasets[2],(None,None)]
    X, y, song_index = get_features()
    X = (X - X.min()) / (X.max() - X.min())
    data = split_data(X, y, song_index)

    del X
    del y
    del song_index

    train_set = (data['X_train'], data['y_train'])
    test_set = (data['X_test'], data['y_test'])
    valid_set = (data['X_val'], data['y_val'])
    song_index_set = (data['song_index_train'], data['song_index_val'], data['song_index_test'])
    del data

    f = open('../checkpoint/finetune.save','rb')
    dbn = cPickle.load(f)
    f.close()
    model = Sequential()
    model.add(Dense(50, activation='sigmoid', input_shape=(513,), weights=(dbn.sigmoid_layers[0].W.get_value(),
                                                                           dbn.sigmoid_layers[0].b.get_value())))
    model.add(Dense(50, activation='sigmoid', weights=(dbn.sigmoid_layers[1].W.get_value(),
                                                       dbn.sigmoid_layers[1].b.get_value())))
    model.add(Dense(50, activation='sigmoid', weights=(dbn.sigmoid_layers[2].W.get_value(),
                                                       dbn.sigmoid_layers[2].b.get_value())))
    # model.add(Dense(10, activation='softmax', weights=(dbn.logLayer.W.get_value(),
    #                                                    dbn.logLayer.b.get_value())))

    print("Model Initialized successfully!!!")

    results = model.predict(train_set[0], batch_size=200)
    print(results.shape)
    # results_classes = results.argmax(axis=-1)
    # song_labels = np.unique(song_index_set[0])
    # accuracies = []
    # for song_label in song_labels:
    #     current_song_metrics = results_classes[song_index_set[0]==song_label]
    #     actual_song_metrics = train_set[1][song_index_set[0]==song_label]
    #     current_song_class = np.argmax(np.bincount(current_song_metrics))
    #     accuracies.append(current_song_class==actual_song_metrics[0])
    # print("Training Accuracy = ", np.sum(accuracies) / len(accuracies))
    #
    # results = model.predict(valid_set[0], batch_size=200)
    # results_classes = results.argmax(axis=-1)
    # song_labels = np.unique(song_index_set[1])
    # accuracies = []
    # for song_label in song_labels:
    #     current_song_metrics = results_classes[song_index_set[1] == song_label]
    #     actual_song_metrics = valid_set[1][song_index_set[1] == song_label]
    #     current_song_class = np.argmax(np.bincount(current_song_metrics))
    #     accuracies.append(current_song_class==actual_song_metrics[0])
    # print("Validation Accuracy = ", np.sum(accuracies) / len(accuracies))
    #
    # results = model.predict(test_set[0], batch_size=200)
    # results_classes = results.argmax(axis=-1)
    # song_labels = np.unique(song_index_set[2])
    # accuracies = []
    # for song_label in song_labels:
    #     current_song_metrics = results_classes[song_index_set[2] == song_label]
    #     actual_song_metrics = test_set[1][song_index_set[2] == song_label]
    #     current_song_class = np.argmax(np.bincount(current_song_metrics))
    #     accuracies.append(current_song_class==actual_song_metrics[0])
    # print("Test Accuracy = ", np.sum(accuracies) / len(accuracies))

    # Training Accuracy = 0.97
    # Validation Accuracy = 0.735
    # Test Accuracy = 0.7466666666666667




def train_DBN(finetune_lr=0.1, pretraining_epochs=10,
             pretrain_lr=0.001, k=1, training_epochs=474,
             dataset='tzanetakis', batch_size=100, pretrain=False):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    

    datasets = load_data(dataset, only_train=True)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    del datasets

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=513,
              hidden_layers_sizes=[50, 50, 50],
              n_outs=10)

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    if pretrain:
        print('... pre-training the model')
        start_time = timeit.default_timer()
        # Pre-train layer-wise
        for i in range(dbn.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
                print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
                print(numpy.mean(c, dtype='float64'))

        end_time = timeit.default_timer()
        # end-snippet-2
        print('The pretraining code for file ' + os.path.split(__file__)[1] +
            ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

        f = open('../checkpoint/pretrain.save', 'wb')
        cPickle.dump(dbn, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    ########################
    # FINETUNING THE MODEL #
    ########################
    f = open('../checkpoint/pretrain.save','rb')
    dbn = cPickle.load(f)
    f.close()

    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    del train_set_x
    del train_set_y
    del valid_set_x
    del valid_set_y

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetuning the model')
    # early-stopping parameters

    # look as this many examples regardless
    patience = 4 * n_train_batches

    # wait this much longer when a new best is found
    patience_increase = 2.

    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    # go through this many minibatches before checking the network on
    # the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # # test it on the test set
                    # test_losses = test_model()
                    # test_score = numpy.mean(test_losses, dtype='float64')
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #        'best model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))

            # if patience <= iter:
            #     done_looping = True
            #     break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           ) % (best_validation_loss * 100., best_iter + 1))#'with test performance %f %%'#, test_score * 100.))
    print('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

    f = open('../checkpoint/finetune.save', 'wb')
    cPickle.dump(dbn, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print("Models saved successfully!")


if __name__ == '__main__':
    # train_DBN()
    test_DBN()