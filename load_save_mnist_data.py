# '''
# Download, Save and Load MNIST dataset
# '''
# import os
# import numpy as np
# import tensorflow as tf
#
# from dir_info import data_dir
#
# def obtain_save_mnist_data():
#     # TODO - Load training and eval data
#     mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#     train_data = mnist.train.images  # Returns np.array
#     train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#     eval_data = mnist.test.images  # Returns np.array
#     eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#     # TODO - Save data
#     np.save(os.path.join(data_dir, 'train_features.npy'), train_data)
#     np.save(os.path.join(data_dir, 'train_labels.npy'), train_labels)
#
#     np.save(os.path.join(data_dir, 'eval_features.npy'), eval_data)
#     np.save(os.path.join(data_dir, 'eval_labels.npy'), eval_labels)
#
# def load_mnist_data():
#     # TODO - Save data
#     train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
#     train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
#
#     eval_features = np.load(os.path.join(data_dir, 'eval_features.npy'))
#     eval_labels = np.load(os.path.join(data_dir, 'eval_labels.npy'))
#
#     return train_features, train_labels, eval_features, eval_labels
#
# if __name__ == '__main__':
#     obtain_save_mnist_data()
