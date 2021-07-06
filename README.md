# SiamesePP
SiamesePP is a deep learning tool to detect DNA modifications from Nanopore sequencing samples. SiamesePP exploits one-shot learning properties (https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) to detect modifications for which the model was not trained on. The proposed model employs a similar architecture and features as DeepMP (https://github.com/pepebonet/DeepMP). 
 
# Contents
- [Installation](#Installation)
- [Usage](#Usage)        

# Installation
## Clone repository
First download siamesePP from the github repository:

        git clone https://github.com/pepebonet/siamesePP.git

## Install dependencies
We highly recommend to use a virtual environment for the installation and employment of siamesePP:

`Create an environment and install siamesePP:`

        conda create --name siamesepp python=3.8
        conda activate siamesepp
        pip install -e .


It may be needed to install all dependencies employed by DeepMP (https://github.com/pepebonet/DeepMP) to fully meet the requirements for siamesePP.


# Usage

This section highlights the main functionalities of siamesePP and the commands to run them. Before starting, it is worth noticing that to run siamesePP a treated and an untreated samples are needed. 

## Feature extraction:
To extract features follow the usage section in DeepMP (Second option --> Extract sequence features). The output consists of the features for the treated and untreated samples. 

## Get pair set

Once the features are extracted, the next step is to generate the pairs that will constitute the input to the model. The pairs of features are generated as follows: 

```
siamesePP get-pairs -dt sup -t treated_features.tsv -un untreated_features.tsv -o output/path
```

- `-dt` is used to specify the type of data employed (supervised or unsupervised).

## Train models

Train model from binary files:

```
siamesePP train-siamese  -tf path/to/train_sup.h5 -vf path/to/val_sup.h5 -md path/to/model -ep 10
```

## Call modifications
```
siamesePP call-modifications -dt sup -tf path/to/test_sup.h5 -md path/to/trained_model -o output/path
```

- `-dt` is used to specify the type of data employed (supervised or unsupervised).
