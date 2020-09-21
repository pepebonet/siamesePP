#!usr/bin/env python3
import logging
import argparse
from argparse import Namespace

import click

from .train import *
from .get_pair_sets import *
from .call_modifications import *


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# ------------------------------------------------------------------------------
# CLICK
# ------------------------------------------------------------------------------

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--debug', help="Show more progress details", is_flag=True)
def cli(debug):
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logging_level)

    if not debug:
        # Hide bgdata messages
        logging.getLogger('bgdata').setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# CALL MODIFICATIONS
# ------------------------------------------------------------------------------

@cli.command(short_help='Calling modifications on a set')
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
def call_modifications(data_type, test_file, model_dir, kmer_sequence, output):

    call_mods(data_type, test_file, model_dir, kmer_sequence, output)
    


# ------------------------------------------------------------------------------
# GET PAIR SETS
# ------------------------------------------------------------------------------

@cli.command(short_help='Separate files per position and create pair sets')
@click.option(
    '-dt', '--data-type', required=True,
    type=click.Choice(['sup', 'unsup']),
    help='Supervised or unsupervised tests'
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
def get_pairs(data_type, unsupervised_data, treated, untreated, output):
    if data_type == 'sup':
        do_supervised(treated, untreated, data_type, output)
    else:
        do_unsupervised(unsupervised_data, data_type, output)


# ------------------------------------------------------------------------------
# TRAIN SIAMESE NETWORK
# ------------------------------------------------------------------------------

@cli.command(short_help='Train siamese network')
@click.option(
    '-tf', '--train_file', default='',
    help='path to training set'
)
@click.option(
    '-vf', '--val_file', default='',
    help='path to validation set'
)
@click.option(
    '-ks', '--kmer_sequence', default=17,
    help='kmer length for sequence training'
)
@click.option(
    '-ep', '--epochs', default=5,
    help='Number of epochs for training'
)
@click.option(
    '-md', '--model_dir', default='models/',
    help='directory to trained model'
)
@click.option(
    '-ld', '--log_dir', default='logs/',
    help='training log directory'
)
@click.option(
    '-bs', '--batch_size', default=512,
    help='Batch size for training both models'
)
def train_siamese(train_file, val_file, log_dir, model_dir, 
    batch_size, kmer_sequence, epochs):

    train_model(
        train_file, val_file, log_dir, model_dir, 
        batch_size, kmer_sequence, epochs
    )
    

if __name__ == '__main__':
    cli()
