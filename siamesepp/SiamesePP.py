#!usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import time
import h5py

import cv2
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Lambda, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model

# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng

def initialize_weights(shape, dtype=None):
    return K.variable(np.random.normal(loc = 0.0, scale = 1e-2, size = shape), dtype)


def initialize_bias(shape, dtype=None):
    return K.variable(np.random.normal(loc = 0.5, scale = 1e-2, size = shape), dtype)


def load_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_median = hf['signal_median'][:]
        signal_skew = hf['signal_skew'][:]
        signal_kurt = hf['signal_kurt'][:]
        signal_diff = hf['signal_diff'][:]
        signal_lens = hf['signal_lens'][:]
        label = hf['label'][:]

    return bases, signal_means, signal_stds, signal_median, signal_skew, \
        signal_kurt, signal_diff, signal_lens, label


def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net



def train_model(train_file, batch_size=1, kmer=17, epochs=100):

    embedding_flag = ""

    ## preprocess data
    bases, signal_means, signal_stds, signal_median, signal_skew, \
        signal_kurt, signal_diff, signal_lens, label = load_data(train_file)


    embedding_size = 5
    embedding_flag += "_one-hot_embedded"
    embedded_bases = tf.one_hot(bases, embedding_size)


    ## prepare inputs for NNs
    input_train = tf.concat([embedded_bases,
                                    tf.reshape(signal_means, [-1, kmer, 1]),
                                    tf.reshape(signal_stds, [-1, kmer, 1]),
                                    tf.reshape(signal_median, [-1, kmer, 1]),
                                    tf.reshape(signal_skew, [-1, kmer, 1]),
                                    tf.reshape(signal_kurt, [-1, kmer, 1]),
                                    tf.reshape(signal_diff, [-1, kmer, 1]),
                                    tf.reshape(signal_lens, [-1, kmer, 1])],
                                    axis=2)

    import pdb;pdb.set_trace()
    model = get_siamese_model((105, 105, 1))
    optimizer = Adam(lr = 0.00006)
    model.compile(loss="binary_crossentropy",optimizer=optimizer)

    model.fit(input_train, label, batch_size=batch_size, epochs=epochs)

    return None


def main():
    path = '/workspace/projects/nanopore/stockholm/ecoli/sequence_features/test_more_features/all_features/val_seq.h5'
    features = train_model(path)
    

    


if __name__ == '__main__':
    main()