#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import siamesepp.utils as ut
import siamesepp.model as md


def acc_test_single(data, labels, model, kmer, score_av='binary'):
    test_loss, test_acc = model.evaluate(data, tf.convert_to_tensor(labels))

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )
    
    return [test_acc, precision, recall, f_score], pred, inferred


def infer_mods(data, model):

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    return pred, inferred


def call_mods(data_type, test_file, model_file, kmer, output):
    data, labels = ut.get_pairs(test_file, kmer)
    model = load_model(model_file)

    if data_type == 'sup':
        acc, pred, inferred = acc_test_single(data, labels, model, kmer)
        ut.save_output(acc, output, 'accuracy_measurements.txt')
    else:
        pred, inferred = infer_mods(data, model)
    
    ut.save_probs(pred, inferred, labels, output)
