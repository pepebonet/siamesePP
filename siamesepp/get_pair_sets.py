#!/usr/bin/env python3
import os
import h5py
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split

import siamesepp.utils as ut

#TODO change to only seq
names_all = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_diff', 'qual', 'mis', 'ins', 'del', 'methyl_label']

names_seq = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_diff', 'methyl_label']


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------

def get_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=names_seq)
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['strand']

    return df


def get_equals_set(df):
    df = df.sort_values(by=['id']).reset_index(drop=True)
    count = Counter(df['id'])
    to_analyze = [el for el in count.elements() if count[el] >= 2]
    
    ids_df = np.sort(df['id'].values.ravel())

    ids_all = np.sort(np.asarray(to_analyze).ravel())
    indices = np.searchsorted(ids_df, list(set(ids_all)), side='right')

    arr1 = np.empty([0,], dtype='int32')
    arr2 = np.empty([0,], dtype='int32')

    for el in tqdm(zip(list(set(ids_all)), indices)):
        num_of_pos = count[el[0]]
        pos = np.arange(el[1] - num_of_pos, el[1])
        arr1 = np.append(arr1, pos[:int(len(pos) / 2)])
        arr2 = np.append(arr2, pos[int(len(pos) / 2):])

    merged = pd.merge(
        df.iloc[arr1, :], df.iloc[arr2, :], on=['id'], how='inner'
    )
    merged['label'] = 1

    return merged


def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def do_supervised(treated, untreated, data_type, output, split_file):
    treated = get_data(treated)
    untreated = get_data(untreated)

    treat_untreat = pd.merge(treated, untreated, on=['id'], how='inner')
    treat_untreat['label'] = 0
    import pdb;pdb.set_trace()
    print(treat_untreat.shape[0])
    treat_treat = get_equals_set(treated)
    
    untreat_untreat = get_equals_set(untreated)
    import pdb;pdb.set_trace()
    df = pd.concat([treat_untreat, treat_treat, untreat_untreat])
    print('Total number of features: {}'.format(df.shape[0]))

    if df.empty:
        raise Exception('No features could be extracted...')

    if split_file:
        data = get_training_test_val(df)
    else:
        data = [(df, 'test')]
    
    for el in data:
        ut.preprocess_sequence(el[0], output, data_type, el[1], 'x')
        ut.preprocess_sequence(el[0], output, data_type, el[1], 'y') 


def do_unsupervised(treated, untreated, data_type, output):
    treated = get_data(treated)
    untreated = get_data(untreated)

    treated['id'] = treated['chrom'] + '_' + treated['pos'].astype(str)
    untreated['id'] = untreated['chrom'] + '_' + untreated['pos'].astype(str)

    df_pairs = pd.merge(treated, untreated, on='id', how='inner')

    ut.preprocess_sequence(df_pairs, output, data_type, 'mms_untreated', 'x')
    ut.preprocess_sequence(df_pairs, output, data_type, 'mms_untreated', 'y') 
