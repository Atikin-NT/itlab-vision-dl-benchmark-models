# Example of training a GCN using DGL


## Introduction


This repository provides an example of training a graph neural network 
using the DGL framework on the [Cora][cora] dataset.


The Cora dataset consists of 2708 scientific publications classified into 
one of seven classes. The citation network consists of 5429 links. 
Each publication in the dataset is described by a 0/1-valued word vector 
indicating the absence/presence of the corresponding word from 
the dictionary.


**Goal**: predict which class an article belongs to.

## Structure

- `dgl_cora.ipynb` file with a description of the model architecture, 
  its training, as well as an inference.
- `GCN.py` file describing the architecture of the GCN model.
- `APPNP.py` file describing the architecture of the [APPNP][appnp] model.
- `CRD_CLS.py` file describing the architecture of the [CRD_CLS][crd_cls] model.
- `gcn_model.pt` GCN model file in **pt** format.
- `appnp_model.pt` APPNP model file in **pt** format.
- `crd_cls_model.pt` CRD_CLS model file in **pt** format.

## Usage

This repository is used to provide files to the [dl-benchmark][dl-benchmark] repository.
