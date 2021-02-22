#!usr/bin/env python3
import os
import datetime
import tensorflow as tf

import siamesepp.utils as ut
import siamesepp.model as md

embedding_size = 5

def train_model(train_file, val_file, log_dir, model_dir, 
    batch_size, kmer_sequence, epochs):

    pairs_train, labels_train = ut.get_pairs(train_file, kmer_sequence)
    pairs_val, labels_val = ut.get_pairs(val_file, kmer_sequence)
        
    model = md.get_alternative_siamese((kmer_sequence, embedding_size + 4))

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq")
    model_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq_model")

    model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])
    print(model.summary())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)

    callback_list = [
                    tensorboard_callback,
                    tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                        monitor='val_accuracy',
                                                        mode='max',
                                                        save_best_only=True,
                                                        save_weights_only= False)
                    ]
    model.fit(pairs_train, labels_train, batch_size=batch_size, epochs=epochs,
                                            callbacks = [tensorboard_callback],
                                            validation_data = (pairs_val, labels_val))
    model.save(model_dir)
    model.save_weights(os.path.join(model_dir, 'weights.h5'))
    
    return None