#!usr/bin/env python3
import os
import click
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense

import siamesepp.utils as ut
import siamesepp.model as md


def train_model(train_file, val_file, log_dir, model_dir, 
    batch_size, kmer_sequence, epochs):

    pairs_train, labels_train = ut.get_pairs(train_file, kmer_sequence)
    pairs_val, labels_val = ut.get_pairs(val_file, kmer_sequence)

    embedding_size = 5
    model = md.get_alternative_siamese((kmer_sequence, embedding_size + 4))
    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_lstm")

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    model.fit(pairs_train, labels_train, batch_size=batch_size, epochs=epochs,
                                            callbacks = [tensorboard_callback],
                                            validation_data = (pairs_val, labels_val))
    model.save(model_dir + "siamesePP_model")
    model.save_weights(os.path.join(model_dir, 'weights.h5'))
    
    return None