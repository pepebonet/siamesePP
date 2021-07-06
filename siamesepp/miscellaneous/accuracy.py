#!/usr/bin/env python3

import os 
import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support


def plot_ROC_alone(df, fig_out, kn='Linear'):
    fpr_df, tpr_df, thresholds = roc_curve(
        df['labels'].values, df['probs'].values
    )
    roc_auc_df = auc(fpr_df, tpr_df)
    plt.plot (fpr_df, tpr_df, lw=2, label ='df: {}'.format(round(roc_auc_df, 3)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def plot_barplot_deepmp_siamese(df, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#f03b20'], 
        hue_order=['DeepMP', 'SiamesePP'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('SiamesePP', '#f03b20')]:
        custom_lines.append(
                plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
                mew=0, color=el[1], label=el[0])[0] 
            )


    ax.set_ylabel("Performance", fontsize=12)
    ax.set_ylabel("", fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=2, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'accuracies_plot.pdf')
    plt.savefig(out_dir)
    plt.close()


def get_barplot(deepmp_accuracies, siamese_accuracies, output):
    deepmp_acc = pd.read_csv(deepmp_accuracies, sep='\t')
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    siamese_acc = pd.read_csv(siamese_accuracies, sep='\t')
    siamese_acc = siamese_acc.T.reset_index()
    siamese_acc['Model'] = 'SiamesePP'
    
    df_acc = pd.concat([deepmp_acc, siamese_acc])
    
    plot_barplot_deepmp_siamese(df_acc, output)


def plot_ROC_deepmp_siamese(siamese, deepmp, fig_out, kn='Linear'):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_siam, tpr_siam, thresholsiam = roc_curve(
        siamese['labels'].values, siamese['probs'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_siam = auc(fpr_siam, tpr_siam)

    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='SiamesePP AUC: {}'.format(round(roc_auc_siam, 3)))[0] 
    )

    plt.plot (fpr_siam, tpr_siam, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')

    plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()


def plot_precision_recall_siamese(deepmp, siamese, fig_out):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    dmp_prec, dmp_rec, _ = precision_recall_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    siam_prec, siam_rec, _ = precision_recall_curve(
        siamese['labels'].values, siamese['probs'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_siam = auc(siam_rec, siam_prec)

    # plot the precision-recall curves
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='SiamesePP AUC: {}'.format(round(auc_siam, 3)))[0] 
    )

    plt.plot(siam_rec, siam_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')
    

    # axis labels
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()


@click.command(short_help='SVM accuracy output')
@click.option(
    '-do', '--deepmp_output', default='', 
    help='Output table from deepMP'
)
@click.option(
    '-so', '--siamese_output', default='', 
    help='Output table from deepMP sequence module only'
)
@click.option(
    '-da', '--deepmp_accuracies', default='', 
    help='Output accuracies from DeepMP'
)
@click.option(
    '-sa', '--siamese_accuracies', default='', 
    help='Output accuracies from siamesePP'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(deepmp_output, siamese_output, deepmp_accuracies, siamese_accuracies, output):
    out_fig = os.path.join(output, 'AUC_comparison.pdf')
    prc_fig = os.path.join(output, 'PRC_comparison.pdf')

    if siamese_output:
        siamese = pd.read_csv(siamese_output, sep='\t')

    if deepmp_output:
        deepmp = pd.read_csv(deepmp_output, sep='\t')

    get_barplot(deepmp_accuracies, siamese_accuracies, output)
    plot_ROC_deepmp_siamese(siamese, deepmp, out_fig)
    plot_precision_recall_siamese(deepmp, siamese, prc_fig)


if __name__ == "__main__":
    main()