#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import siamesepp.utils as ut
import siamesepp.train as tr


def load_model(model_file, kmer, feat):
    model = tr.get_alternative_siamese((kmer, feat))
    model.load_weights(os.path.join(model_file, "weights.h5"))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    return model


def acc_test_single(data, labels, model_file, kmer, feat=12, score_av='binary'):
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


def call_mods(data_type, test_file, model_file, kmer, output, feat=12):
    data, labels = ut.get_pairs(test_file, kmer, data_type)
    model = load_model(model_file, kmer, feat)

    if data_type == 'sup':
        acc, pred, inferred = acc_test_single(data, labels, model, kmer)
        save_output(acc, output, 'accuracy_measurements.txt')
    else:
        pred, inferred = infer_mods(data, model)
    save_probs(pred, inferred, labels, output)


def save_probs(probs, inferred, labels, output):
    out_probs = os.path.join(output, 'test_pred_prob.txt')
    probs_to_save = pd.DataFrame(columns=['labels', 'probs', 'inferred'])
    probs_to_save['labels'] = labels
    probs_to_save['probs'] = probs
    probs_to_save['inferred'] = inferred
    probs_to_save.to_csv(out_probs, sep='\t', index=None)


def save_output(acc, output, label):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, label), index=False, sep='\t')


@click.command(short_help='Script test a set')
@click.option(
    '-dt', '--data-type', required=True,
    type=click.Choice(['sup', 'unsup']),
    help='Supervised '
)
@click.option(
    '-tf', '--test_file', default='',
    help='path to test set'
)
@click.option(
    '-ks', '--kmer_sequence', default=17,
    help='kmer length for sequence training'
)
@click.option(
    '-md', '--model_dir', default='models/',
    help='directory to trained model'
)
@click.option(
    '-o', '--output', default=''
)
def main(data_type, test_file, model_dir, kmer_sequence, output):

    call_mods(data_type, test_file, model_dir, kmer_sequence, output)
    

if __name__ == '__main__':
    main()