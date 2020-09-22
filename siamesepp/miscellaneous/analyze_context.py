#!/usr/bin/envs python3

import os
import click
import pandas as pd
from collections import Counter

import sys
sys.path.append('../.')
import siamesepp.utils as ut


@click.command(short_help='Analyze output treatments')
@click.option(
    '-p', '--predictions', help='predictions'
)
@click.option(
    '-tf', '--test-file', help='test-file'
)
@click.option(
    '-o', '--output', default=''
)
def main(predictions, test_file, output):
    preds = pd.read_csv(predictions, sep='\t')
    test = ut.load_data(test_file, 'x', 'unsup')

    concat = pd.concat([preds, sequences], axis=1)
    dissimilar = concat[concat['inferred'] == 0]
    
    print(Counter(dissimilar[8]))
    print(Counter(concat[8]))


if __name__ == '__main__':
    main()