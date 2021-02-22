import h5py
import numpy as np
import tensorflow as tf

def load_data(file, pair, data_type):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer_{}'.format(pair)][:]
        signal_means = hf['signal_means_{}'.format(pair)][:]
        signal_stds = hf['signal_stds_{}'.format(pair)][:]
        signal_median = hf['signal_median_{}'.format(pair)][:]
        signal_skew = hf['signal_skew_{}'.format(pair)][:]
        signal_kurt = hf['signal_kurt_{}'.format(pair)][:]
        signal_diff = hf['signal_diff_{}'.format(pair)][:]
        signal_lens = hf['signal_lens_{}'.format(pair)][:]
        if data_type == 'sup':
            label = hf['label_{}'.format(pair)][:]
        else: 
            label = np.zeros(signal_means.shape[0])

    return bases, signal_means, signal_stds, signal_median, signal_skew, \
        signal_kurt, signal_diff, signal_lens, label


def get_feat_pairs(g1, g2): 
    pairs = [np.zeros((g1.shape[0], g1.shape[1], g1.shape[2])) for i in range(2)]
    for i in range(g1.shape[0]):
        pairs[0][i,:,:] = g1[i]
        pairs[1][i,:,:] = g2[i]
    return pairs


def concat_tensors(bases, v2, v3, v4, v5, v6, v7, v8, kmer):
    return tf.concat([bases, 
                                tf.reshape(v2, [-1, kmer, 1]),
                                tf.reshape(v3, [-1, kmer, 1]),
                                tf.reshape(v4, [-1, kmer, 1]),
                                tf.reshape(v5, [-1, kmer, 1]),
                                tf.reshape(v6, [-1, kmer, 1]),
                                tf.reshape(v7, [-1, kmer, 1]),
                                tf.reshape(v8, [-1, kmer, 1])],
                                axis=2)


def get_pairs(feat_file, kmer_sequence, data_type='sup'):
    embedding_flag = ""

    ## preprocess data
    v1_x, v2_x, v3_x, v4_x, v5_x, v6_x, v7_x, v8_x, vy_x  = load_data(feat_file, 'x', data_type)
    v1_y, v2_y, v3_y, v4_y, v5_y, v6_y, v7_y, v8_y, vy_y  = load_data(feat_file, 'y', data_type)

    embedding_size = 5
    embedding_flag += "_one-hot_embedded"
    bases = tf.one_hot(v1_x, embedding_size)

    ## prepare inputs for NNs
    input_x = concat_tensors(
        bases, v2_x, v3_x, v4_x, v5_x, v6_x, v7_x, v8_x, kmer_sequence
    )
    input_y = concat_tensors(
        bases, v2_y, v3_y, v4_y, v5_y, v6_y, v7_y, v8_y, kmer_sequence
    )

    pairs = get_feat_pairs(input_x, input_y)

    return pairs, vy_x


def preprocess_sequence(df, output, label_file, file):

    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median'].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff'].values]
    label = df['methyl_label']
    chrom = df['chrom'].values.astype('S')
    strand = df['strand'].values.astype('S')
    readname = df['readname'].values.astype('S')
    pos = df['pos'].values
    pos_in_strand = df['pos_in_strand'].values

    file_name = os.path.join(
        output, '{}'.format(label_file), '{}_{}.h5'.format(file, label_file)
    )
    if not os.path.isdir(os.path.dirname(file_name)):
        file_name = os.path.join(output, '{}_{}.h5'.format(file, label_file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_median",  data=np.stack(base_median))
        hf.create_dataset("signal_diff",  data=np.stack(base_diff))
        hf.create_dataset("methyl_label",  data=label)
        hf.create_dataset('chrom',  data=chrom, chunks=True, maxshape=(None,), dtype='S10')
        hf.create_dataset('strand',  data=strand, chunks=True, maxshape=(None,), dtype='S1')
        hf.create_dataset('readname',  data=readname, chunks=True, maxshape=(None,), dtype='S200')
        hf.create_dataset('pos',  data=pos, chunks=True, maxshape=(None,))
        hf.create_dataset('pos_in_strand',  data=pos_in_strand, chunks=True, maxshape=(None,))

    return None