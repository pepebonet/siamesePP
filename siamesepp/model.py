#!usr/bin/env python3
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense


class ConvBlock(Model):

    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = Conv1D(filters, kernel_size, padding="same",
                            kernel_initializer="lecun_normal")
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LocalBlock(Model):

    def __init__(self, filters, kernel_size):
        super(LocalBlock, self).__init__()
        self.localconv = LocallyConnected1D(filters, kernel_size,
                                            kernel_initializer="lecun_normal")
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.localconv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvLocalBlock(Model):

    def __init__(self, filters, kernel_size):
        super(ConvLocalBlock, self).__init__()
        self.conv = Conv1D(filters, kernel_size, padding="same",
                            kernel_initializer="lecun_normal")
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.localconv = LocallyConnected1D(filters, kernel_size)
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.localconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class SequenceCNN(Model):

    def __init__(self, cnn_block, block_num, filters, kernel_size):
        super(SequenceCNN, self).__init__()
        self.block_num = block_num
        if cnn_block == 'conv':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvBlock(filters, kernel_size))
        elif cnn_block == 'local':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, LocalBlock(filters, kernel_size))
        else:
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvLocalBlock(filters, kernel_size))
        self.pool = GlobalAveragePooling1D(name='seq_pooling_layer')
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel=False):
        x = inputs
        for i in range(self.block_num):
            x = getattr(self, "block%i" % i)(x)
        x = self.pool(x)
        if submodel:
            return x
        x = self.dense1(x)
        return self.dense2(x)


def initialize_bias(shape, dtype=None):
    return K.variable(np.random.normal(loc = 0.5, scale = 1e-2, size = shape), dtype)


def get_alternative_siamese(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # model = SequenceCNN('conv', 6, 256, 4)
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
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net