'''
Reference:
https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import product

import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.losses import categorical_crossentropy       # choose categorical crossentropy as loss function
from keras.regularizers import l2                       # use L2 regularizer
from keras.optimizers import SGD                        # use SGD as optimizer

from call_back_func import TestBatchCallback
from dir_info import data_dir, result_dir
from config import batch_size, epochs, num_batch_record

mlp_result_dir = os.path.join(result_dir, 'mlp')
if not os.path.exists(mlp_result_dir):
    os.makedirs(mlp_result_dir)

# some parameters related to model training but not hyper-parameters
num_classes = 10

# TODO - load mnist data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
mnist_data = np.load(os.path.join(data_dir, 'mnist.npz'))
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']
x_test = mnist_data['x_test']
y_test = mnist_data['y_test']

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# TODO - uniformization
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# TODO - 切割训练集和验证集
x_val = x_train[50000:, :]
y_val = y_train[50000:, :]
x_train = x_train[:50000, :]
y_train = y_train[:50000, :]

# batch_size = 128
# epochs = 2

# Used to record some information
train_loss_record_dict ={}
train_acc_record_dict ={}
val_loss_record_dict ={}
val_acc_record_dict ={}
test_loss_record_dict ={}
test_acc_record_dict ={}

# TODO - defind hyper-parameters to be tuned
# learning_rate = 0.001
# l2_lambda = 1e-6
# dense_hidden_units = 512    # network parameters
# learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# l2_lambda = [1e-4, 1e-5, 1e-6, 1e-7]
# dense_hidden_units = [128, 256, 512]    # network parameters
learning_rate = [0.1, 0.01, 0.005, 0.001]
l2_lambda = [1e-4, 1e-5, 1e-6]
dense_hidden_units = [128, 256, 512]    # network parameters

for learning_rate, l2_lambda, dense_hidden_units in product(learning_rate, l2_lambda, dense_hidden_units):
    model_id = '%s_%s_%s' % (learning_rate, l2_lambda, dense_hidden_units)

    # TODO - define models
    l2_regularizer = l2(l=l2_lambda)

    model = Sequential()

    # TODO - add L2 Regularizer in fully connected layers
    model.add(Dense(dense_hidden_units,
                    kernel_regularizer=l2_regularizer,
                    activation='relu',
                    input_shape=(784,)))
    # model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=learning_rate),
                  metrics=['accuracy'])

    # test_callback = TestEpochCallback(x_test, y_test)
    test_callback = TestBatchCallback(x_val, y_val, x_test, y_test)
    # TODO - traning
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[test_callback])

    # TODO - Record information
    model_performance_dict = OrderedDict(
        train_losses=test_callback.train_losses,
        val_losses=test_callback.val_losses,
        test_losses=test_callback.test_losses,
        train_acc=test_callback.train_acc,
        val_acc=test_callback.val_acc,
        test_acc=test_callback.test_acc
    )

    model_performance_df = pd.DataFrame(model_performance_dict)
    model_performance_df.index = list(range(1, model_performance_df.shape[0] + 1))
    model_performance_df.index = model_performance_df.index * num_batch_record
    model_performance_df.index.name = 'batch'
    model_performance_df.to_excel(os.path.join(mlp_result_dir, 'mnist_mlp_performance_%s.xls' % (model_id)))

    train_loss_record_dict[(learning_rate, l2_lambda, dense_hidden_units)] = test_callback.train_losses
    train_acc_record_dict[(learning_rate, l2_lambda, dense_hidden_units)] = test_callback.train_acc
    val_loss_record_dict[(learning_rate, l2_lambda, dense_hidden_units)] = test_callback.val_losses
    val_acc_record_dict[(learning_rate, l2_lambda, dense_hidden_units)] = test_callback.val_acc
    test_loss_record_dict[(learning_rate, l2_lambda, dense_hidden_units)] = test_callback.test_losses
    test_acc_record_dict[(learning_rate, l2_lambda, dense_hidden_units)] = test_callback.test_acc
    print('model %s done!\n\n' % (model_id))

train_loss_record_df = pd.DataFrame(train_loss_record_dict).transpose()
train_loss_record_df.columns = train_loss_record_df.columns * num_batch_record
train_loss_record_df.index.names = ['lr', 'l2_lambda', 'dense_hidden_units']
train_loss_record_df = train_loss_record_df.reset_index()
train_loss_record_df.to_excel(os.path.join(mlp_result_dir, 'mlp_train_loss_record_df.xls'), index=None)
train_loss_record_df.to_hdf(os.path.join(mlp_result_dir, 'mlp_train_loss_record_df.h5'), key='mlp_train_loss_record_df')

train_acc_record_df = pd.DataFrame(train_acc_record_dict).transpose()
train_acc_record_df.columns = train_acc_record_df.columns * num_batch_record
train_acc_record_df.index.names = ['lr', 'l2_lambda', 'dense_hidden_units']
train_acc_record_df = train_acc_record_df.reset_index()
train_acc_record_df.to_excel(os.path.join(mlp_result_dir, 'mlp_train_acc_record_df.xls'), index=None)
train_acc_record_df.to_hdf(os.path.join(mlp_result_dir, 'mlp_train_acc_record_df.h5'), key='mlp_train_acc_record_df')

val_loss_record_df = pd.DataFrame(val_loss_record_dict).transpose()
val_loss_record_df.columns = val_loss_record_df.columns * num_batch_record
val_loss_record_df.index.names = ['lr', 'l2_lambda', 'dense_hidden_units']
val_loss_record_df = val_loss_record_df.reset_index()
val_loss_record_df.to_excel(os.path.join(mlp_result_dir, 'mlp_val_loss_record_df.xls'), index=None)
val_loss_record_df.to_hdf(os.path.join(mlp_result_dir, 'mlp_val_loss_record_df.h5'), key='mlp_val_loss_record_df')

val_acc_record_df = pd.DataFrame(val_acc_record_dict).transpose()
val_acc_record_df.columns = val_acc_record_df.columns * num_batch_record
val_acc_record_df.index.names = ['lr', 'l2_lambda', 'dense_hidden_units']
val_acc_record_df = val_acc_record_df.reset_index()
val_acc_record_df.to_excel(os.path.join(mlp_result_dir, 'mlp_val_acc_record_df.xls'), index=None)
val_acc_record_df.to_hdf(os.path.join(mlp_result_dir, 'mlp_val_acc_record_df.h5'), key='mlp_val_acc_record_df')

test_loss_record_df = pd.DataFrame(test_loss_record_dict).transpose()
test_loss_record_df.columns = test_loss_record_df.columns * num_batch_record
test_loss_record_df.index.names = ['lr', 'l2_lambda', 'dense_hidden_units']
test_loss_record_df = test_loss_record_df.reset_index()
test_loss_record_df.to_excel(os.path.join(mlp_result_dir, 'mlp_test_loss_record_df.xls'), index=None)
test_loss_record_df.to_hdf(os.path.join(mlp_result_dir, 'mlp_test_loss_record_df.h5'), key='mlp_test_loss_record_df')

test_acc_record_df = pd.DataFrame(test_acc_record_dict).transpose()
test_acc_record_df.columns = test_acc_record_df.columns * num_batch_record
test_acc_record_df.index.names = ['lr', 'l2_lambda', 'dense_hidden_units']
test_acc_record_df = test_acc_record_df.reset_index()
test_acc_record_df.to_excel(os.path.join(mlp_result_dir, 'mlp_test_acc_record_df.xls'), index=None)
test_acc_record_df.to_hdf(os.path.join(mlp_result_dir, 'mlp_test_acc_record_df.h5'), key='mlp_test_acc_record_df')

print('all done!')

