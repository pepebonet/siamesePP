#!/usr/bin/env python3
import os
import h5py
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split

from tqdm import tqdm


names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens', 
            'methyl_label', 'flag']
base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def kmer2code(kmer_bytes):
    return [base2code_dna[x] for x in kmer_bytes]


def get_data(treated):
    return pd.read_csv(treated, sep='\t', header=None, names=names_all)


def get_equals_set(df):
    c = Counter(df['pos_in_strand'])
    to_analyze = [el for el in c.elements() if c[el] >= 2]
    
    arr1 = np.empty([0,], dtype='int32'); arr2 = np.empty([0,], dtype='int32')
    for el in tqdm(list(set(to_analyze))):
        pos = np.argwhere(df['pos_in_strand'].values == el).flatten()  
        arr1 = np.append(arr1, np.asarray(pos[:int(len(pos) / 2)]))
        arr2 = np.append(arr2, np.asarray(pos[int(len(pos) / 2):]))

    return pd.merge(df.iloc[arr1, :], df.iloc[arr2, :], on=['pos_in_strand'], how='inner')


def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def preprocess_sequence(df, output, data_type, file, pair):
    df = df.dropna()
    kmer = df['kmer_{}'.format(pair)].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means_{}'.format(pair)].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds_{}'.format(pair)].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median_{}'.format(pair)].values]
    base_skew = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_skew_{}'.format(pair)].values]
    base_kurt = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_kurt_{}'.format(pair)].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff_{}'.format(pair)].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens_{}'.format(pair)].values]
    if data_type != 'unsup':
        label = df['label']

    file_name = os.path.join(output, '{}_seq.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer_{}".format(pair),  data=np.stack(kmer))
        hf.create_dataset("signal_means_{}".format(pair),  data=np.stack(base_mean))
        hf.create_dataset("signal_stds_{}".format(pair),  data=np.stack(base_std))
        hf.create_dataset("signal_median_{}".format(pair),  data=np.stack(base_median))
        hf.create_dataset("signal_skew_{}".format(pair),  data=np.stack(base_skew))
        hf.create_dataset("signal_kurt_{}".format(pair),  data=np.stack(base_kurt))
        hf.create_dataset("signal_diff_{}".format(pair),  data=np.stack(base_diff))
        hf.create_dataset("signal_lens_{}".format(pair),  data=np.stack(base_signal_len))
        if data_type != 'unsup':
            hf.create_dataset("label_{}".format(pair),  data=label)

    return None


@click.command(short_help='Script to separate files per position')
@click.option(
    '-dt', '--data-type', required=True,
    type=click.Choice(['sup', 'unsup']),
    help='Supervised '
)
@click.option(
    '-ud', '--unsupervised-data', help='unsupervised data file'
)
@click.option(
    '-t', '--treated', help='treated file'
)
@click.option(
    '-un', '--untreated', help='untreated file'
)
@click.option(
    '-o', '--output', default=''
)
def main(data_type, unsupervised_data, treated, untreated, output):
    if data_type == 'sup':

        treated = get_data(treated)
        untreated = get_data(untreated)
        
        treat_untreat = pd.merge(treated, untreated, on=['pos_in_strand'], how='inner')
        treat_untreat['label'] = 0

        treat_treat = get_equals_set(treated)
        treat_treat['label'] = 1
        
        untreat_untreat = get_equals_set(untreated)
        untreat_untreat['label'] = 1

        df = pd.concat([treat_untreat, treat_treat, untreat_untreat])
        data = get_training_test_val(df)

        for el in data:
            preprocess_sequence(el[0], output, el[1], 'x')
            preprocess_sequence(el[0], output, el[1], 'y') 
    
    else:
        df = get_data(unsupervised_data)
        df_pairs = get_equals_set(df)

        preprocess_sequence(df_pairs, output, data_type, 'mms', 'x')
        preprocess_sequence(df_pairs, output, data_type, 'mms', 'y') 


if __name__ == '__main__':
    main()