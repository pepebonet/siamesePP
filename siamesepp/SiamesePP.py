#!usr/bin/env python3
import os
import sys
import time
import h5py
import click
import pickle
import datetime
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Lambda, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng

def initialize_weights(shape, dtype=None):
    return K.variable(np.random.normal(loc = 0.0, scale = 1e-2, size = shape), dtype)


def initialize_bias(shape, dtype=None):
    return K.variable(np.random.normal(loc = 0.5, scale = 1e-2, size = shape), dtype)


def load_data(file, pair):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer_{}'.format(pair)][:]
        # signal_means = hf['signal_means_{}'.format(pair)][:]
        # signal_stds = hf['signal_stds_{}'.format(pair)][:]
        # signal_median = hf['signal_median_{}'.format(pair)][:]
        # signal_skew = hf['signal_skew_{}'.format(pair)][:]
        # signal_kurt = hf['signal_kurt_{}'.format(pair)][:]
        # signal_diff = hf['signal_diff_{}'.format(pair)][:]
        signal_lens = hf['signal_lens_{}'.format(pair)][:]
        label = hf['label_{}'.format(pair)][:]

    return bases, signal_lens, label
    # return bases, signal_means, signal_stds, signal_median, signal_skew, \
        # signal_kurt, signal_diff, signal_lens, label


def get_alternative_siamese(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(256, 5, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D(name='seq_pooling_layer'))
    model.add(tf.keras.layers.Dropout(0.2))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net


def get_pairs(g1, g2): 
    pairs = [np.zeros((g1.shape[0], g1.shape[1], g1.shape[2])) for i in range(2)]
    for i in range(g1.shape[0]):
        pairs[0][i,:,:] = g1[i]#.reshape(g1.shape[1], g1.shape[2], 1)
        pairs[1][i,:,:] = g2[i]#.reshape(g1.shape[1], g1.shape[2], 1)
    return pairs
    


def concat_tensors(bases, v2, kmer):
    return tf.concat([bases, 
                                tf.reshape(v2, [-1, kmer, 1])],
                                # tf.reshape(v3, [-1, kmer, 1]),
                                # tf.reshape(v4, [-1, kmer, 1]),
                                # tf.reshape(v5, [-1, kmer, 1]),
                                # tf.reshape(v6, [-1, kmer, 1]),
                                # tf.reshape(v7, [-1, kmer, 1]),
                                # tf.reshape(v8, [-1, kmer, 1])],
                                axis=2)


def train_model(train_file, val_file, log_dir, model_dir, 
    batch_size, kmer_sequence, epochs):

    embedding_flag = ""

    ## preprocess data
    bases_x, signal_means_x, \
        label_x = load_data(train_file, 'x')
    bases_y, signal_means_y,  \
        label_y = load_data(train_file, 'y')
    v1_x, v2_x, vy_x  = load_data(val_file, 'x')
    v1_y, v2_y, vy_y  = load_data(val_file, 'y')
    # bases_x, signal_means_x, signal_stds_x, signal_median_x, signal_skew_x, \
    #     signal_kurt_x, signal_diff_x, signal_lens_x, label_x = load_data(train_file, 'x')
    # bases_y, signal_means_y, signal_stds_y, signal_median_y, signal_skew_y, \
    #     signal_kurt_y, signal_diff_y, signal_lens_y, label_y = load_data(train_file, 'y')
    # v1_x, v2_x, v3_x, v4_x, v5_x, v6_x, v7_x, v8_x, vy_x  = load_data(val_file, 'x')
    # v1_y, v2_y, v3_y, v4_y, v5_y, v6_y, v7_y, v8_y, vy_y  = load_data(val_file, 'y')

    embedding_size = 5
    embedding_flag += "_one-hot_embedded"
    embedded_bases = tf.one_hot(bases_x, embedding_size)
    val_bases = tf.one_hot(v1_x, embedding_size)

    ## prepare inputs for NNs
    input_train_x = concat_tensors(
        embedded_bases, signal_means_x, 
        kmer_sequence
    )
    input_train_y = concat_tensors(
        embedded_bases, signal_means_y, 
        kmer_sequence
    )
    input_val_x = concat_tensors(
        val_bases, v2_x, kmer_sequence
    )
    input_val_y = concat_tensors(
        val_bases, v2_y, kmer_sequence
    )
    # input_train_x = concat_tensors(
    #     embedded_bases, signal_means_x, signal_stds_x, signal_median_x, 
    #     signal_skew_x, signal_kurt_x, signal_diff_x, signal_lens_x, kmer_sequence
    # )
    # input_train_y = concat_tensors(
    #     embedded_bases, signal_means_y, signal_stds_y, signal_median_y, 
    #     signal_skew_y, signal_kurt_y, signal_diff_y, signal_lens_y, kmer_sequence
    # )
    # input_val_x = concat_tensors(
    #     val_bases, v2_x, v3_x, v4_x, v5_x, v6_x, v7_x, v8_x, kmer_sequence
    # )
    # input_val_y = concat_tensors(
    #     val_bases, v2_y, v3_y, v4_y, v5_y, v6_y, v7_y, v8_y, kmer_sequence
    # )

    pairs_train = get_pairs(input_train_x, input_train_y)
    pairs_val = get_pairs(input_val_x, input_val_y)

    
    model = get_alternative_siamese((kmer_sequence, embedding_size + 1))
    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_lstm")

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    model.fit(pairs_train, label_x, batch_size=batch_size, epochs=epochs,
                                                callbacks = [tensorboard_callback],
                                                validation_data = (pairs_val, vy_x))
    model.save(model_dir + "siamesePP_model")

    return None


#TODO load the pairs properly and learn how that would serve as input to the model (look at example)
@click.command(short_help='Script to separate files per position')
@click.option(
    '-tf', '--train_file', default='',
    help='path to training set'
)
@click.option(
    '-vf', '--val_file', default='',
    help='path to validation set'
)
@click.option(
    '-ks', '--kmer_sequence', default=17,
    help='kmer length for sequence training'
)
@click.option(
    '-ep', '--epochs', default=5,
    help='Number of epochs for training'
)
@click.option(
    '-md', '--model_dir', default='models/',
    help='directory to trained model'
)
@click.option(
    '-ld', '--log_dir', default='logs/',
    help='training log directory'
)
@click.option(
    '-bs', '--batch_size', default=512,
    help='Batch size for training both models'
)
def main(train_file, val_file, log_dir, model_dir, 
    batch_size, kmer_sequence, epochs):

    train_model(
        train_file, val_file, log_dir, model_dir, 
        batch_size, kmer_sequence, epochs
    )
    

if __name__ == '__main__':
    main()