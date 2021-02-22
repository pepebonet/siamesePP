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

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------

def kmer2code(kmer_bytes):
    return [base2code_dna[x] for x in kmer_bytes]


def get_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=names_all)
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['strand']

    return df


def get_equals_set(df):
    c = Counter(df['id'])
    to_analyze = [el for el in c.elements() if c[el] >= 2]

    if to_analyze:
        arr1 = np.empty([0,], dtype='int32')
        arr2 = np.empty([0,], dtype='int32')

        for el in tqdm(list(set(to_analyze))):

            pos = np.argwhere(df['id'].values == el).flatten()  
            arr1 = np.append(arr1, np.asarray(pos[:int(len(pos) / 2)]))
            arr2 = np.append(arr2, np.asarray(pos[int(len(pos) / 2):]))

        merged = pd.merge(df.iloc[arr1, :], df.iloc[arr2, :], on=['id'], how='inner')
        emrged['label'] = 1

        return merged
    
    else:
        return pd.DataFrame()


def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def do_supervised(treated, untreated, data_type, output):
    treated = get_data(treated)
    untreated = get_data(untreated)

    treat_untreat = pd.merge(treated, untreated, on=['id'], how='inner')
    treat_untreat['label'] = 0

    treat_treat = get_equals_set(treated)
    
    untreat_untreat = get_equals_set(untreated)

    df = pd.concat([treat_untreat, treat_treat, untreat_untreat])
    import pdb; pdb.set_trace()
    if df.empty:
        raise Exception('No features could be extracted...')

    data = get_training_test_val(df)

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
