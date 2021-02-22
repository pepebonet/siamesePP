import os
import h5py
import numpy as np
import tensorflow as tf

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

# ------------------------------------------------------------------------------
# TRAIN AND CALL MODIFICATIONS
# ------------------------------------------------------------------------------

def load_seq_data(file, pair):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer_{}'.format(pair)][:]
        signal_means = hf['signal_means_{}'.format(pair)][:]
        signal_stds = hf['signal_stds_{}'.format(pair)][:]
        signal_medians = hf['signal_median_{}'.format(pair)][:]
        signal_range = hf['signal_diff_{}'.format(pair)][:]
        label = hf['methyl_label_{}'.format(pair)][:]
        chrom = hf['chrom_{}'.format(pair)][:]
        readname = hf['readname_{}'.format(pair)][:]
        pos = hf['pos_{}'.format(pair)][:]
        strand = hf['strand_{}'.format(pair)][:]
        pos_in_strand = hf['pos_in_strand_{}'.format(pair)][:]

    return bases, signal_means, signal_stds, signal_medians, \
        signal_range, label, chrom, readname, pos, strand, pos_in_strand


def get_feat_pairs(g1, g2): 
    pairs = [np.zeros((g1.shape[0], g1.shape[1], g1.shape[2])) for i in range(2)]
    
    for i in range(g1.shape[0]):
        pairs[0][i,:,:] = g1[i]
        pairs[1][i,:,:] = g2[i]
    
    return pairs


def concat_tensors_seq(bases, signal_means, signal_stds, signal_medians,
                        signal_range, kmer):
    return tf.concat([bases,
                                tf.reshape(signal_means, [-1, kmer, 1]),
                                tf.reshape(signal_stds, [-1, kmer, 1]),
                                tf.reshape(signal_medians, [-1, kmer, 1]),
                                tf.reshape(signal_range, [-1, kmer, 1])],
                                axis=2)


def get_pairs(feat_file, kmer):

    ## preprocess data
    bases_x, signal_means_x, signal_stds_x, signal_medians_x, \
            signal_range_x, label_x, chrom_x, readname_x, pos_x, strand_x, \
                pos_in_strand_x  = load_seq_data(feat_file, 'x')

    bases_y, signal_means_y, signal_stds_y, signal_medians_y, \
            signal_range_y, label_y, chrom_y, readname_y, pos_y, strand_y, \
                pos_in_strand_y  = load_seq_data(feat_file, 'y')

    embedding_size = 5
    bases = tf.one_hot(bases_x, embedding_size)

    ## prepare inputs for NNs
    input_x = concat_tensors_seq(bases, signal_means_x, signal_stds_x, 
        signal_medians_x, signal_range_x, kmer)
    
    input_y = concat_tensors_seq(bases, signal_means_y, signal_stds_y, 
        signal_medians_y, signal_range_y, kmer)

    pairs = get_feat_pairs(input_x, input_y)

    return pairs, label_x


# ------------------------------------------------------------------------------
#  PREPROCESS
# ------------------------------------------------------------------------------

def kmer2code(kmer_bytes):
    return [base2code_dna[x] for x in kmer_bytes]


def preprocess_sequence(df, output, label_file, file, pair):

    kmer = df['kmer_{}'.format(pair)].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means_{}'.format(pair)].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds_{}'.format(pair)].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median_{}'.format(pair)].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff_{}'.format(pair)].values]
    label = df['methyl_label_{}'.format(pair)]
    chrom = df['chrom_{}'.format(pair)].values.astype('S')
    strand = df['strand_{}'.format(pair)].values.astype('S')
    readname = df['readname_{}'.format(pair)].values.astype('S')
    pos = df['pos_{}'.format(pair)].values
    pos_in_strand = df['pos_in_strand_{}'.format(pair)].values

    file_name = os.path.join(
        output, '{}'.format(label_file), '{}_{}.h5'.format(file, label_file)
    )
    if not os.path.isdir(os.path.dirname(file_name)):
        file_name = os.path.join(output, '{}_{}.h5'.format(file, label_file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset('kmer_{}'.format(pair),  data=np.stack(kmer))
        hf.create_dataset('signal_means_{}'.format(pair),  data=np.stack(base_mean))
        hf.create_dataset('signal_stds_{}'.format(pair),  data=np.stack(base_std))
        hf.create_dataset('signal_median_{}'.format(pair),  data=np.stack(base_median))
        hf.create_dataset('signal_diff_{}'.format(pair),  data=np.stack(base_diff))
        hf.create_dataset('methyl_label_{}'.format(pair),  data=label)
        hf.create_dataset('chrom_{}'.format(pair),  data=chrom, chunks=True, maxshape=(None,), dtype='S10')
        hf.create_dataset('strand_{}'.format(pair),  data=strand, chunks=True, maxshape=(None,), dtype='S1')
        hf.create_dataset('readname_{}'.format(pair),  data=readname, chunks=True, maxshape=(None,), dtype='S200')
        hf.create_dataset('pos_{}'.format(pair),  data=pos, chunks=True, maxshape=(None,))
        hf.create_dataset('pos_in_strand_{}'.format(pair),  data=pos_in_strand, chunks=True, maxshape=(None,))

    return None
