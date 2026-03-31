# R-GCN Entity Classification

This project implements **Relational Graph Convolutional Networks (R-GCN)** for node classification on RDF knowledge graphs.

## Overview

R-GCN extends standard Graph Convolutional Networks to handle multi-relational data. This implementation supports classification tasks on semantic web datasets.

**Reference**: https://arxiv.org/abs/1703.06103

## Features

* R-GCN model with relational convolutions
* Multiple datasets: AIFB, AM, BGS, MUTAG
* Baseline methods for comparison
* YAML-based experiment configuration
* Sacred integration for experiment tracking

## Project Structure

```
torch_rgcn/     # Core R-GCN implementation
baselines/      # Baseline methods
configs/        # YAML configs
data/           # Datasets
experiments/    # Training scripts
utils/          # Utilities
```

## Installation

### Conda

```
conda env create -f environment.yml
conda activate torch_rgcn_venv
```

### Pip

```
pip install -r requirements.txt
```

## Usage

### Train model

```
python experiments/classify_nodes.py with configs/rgcn/nc-AIFB.yaml
```

### Baselines

Feature-based:

```
python baselines/feat_baseline.py --dataset aifb --classifier logistic
```

Weisfeiler-Lehman:

```
python baselines/wl_baseline.py --dataset aifb --iterations 3
```

## Datasets

* AIFB
* AM
* BGS
* MUTAG

## Notes

* Uses Sacred for experiment tracking
* Supports CPU and GPU
* RDF data in `.nt` format
