# R-GCN Entity Classification

PyTorch implementation of Relational Graph Convolutional Networks (R-GCN) for node classification on RDF datasets.

## Overview

This project implements R-GCN as described in [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) for entity classification tasks. It demonstrates how to classify nodes in knowledge graphs with multiple types of relations.

## Features

- **R-GCN Model**: Full R-GCN implementation with relational convolutions and decomposition strategies
- **Multiple Datasets**: Support for AIFB, AM, BGS, and MUTAG datasets
- **Baseline Methods**: Feature-based and Weisfeiler-Lehman baselines for comparison
- **Configuration-based Training**: YAML configs for easy experiment management
- **Experiment Tracking**: Sacred integration for logging and reproduction

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.3+
- PyTorch Geometric 1.7+

### Setup with Conda

```bash
conda env create -f environment.yml
conda activate torch_rgcn_venv
```

Alternatively, install manually:
```bash
pip install torch==1.7.1 torch-geometric==1.7.0 torch-scatter==2.0.6 torch-sparse==0.6.9
pip install scikit-learn rdflib sacred pyyaml
```

## Usage

### Download Data

```bash
bash get_data.sh
```

### Train R-GCN Model

```bash
python experiments/classify_nodes.py with configs/rgcn/nc-AIFB.yaml
python experiments/classify_nodes.py with configs/rgcn/nc-AM.yaml
python experiments/classify_nodes.py with configs/rgcn/nc-BGS.yaml
python experiments/classify_nodes.py with configs/rgcn/nc-MUTAG.yaml
```

### Run Baselines

**Feature-based Baseline:**
```bash
python baselines/feat_baseline.py --dataset aifb
python baselines/feat_baseline.py --dataset mutag
```

**Weisfeiler-Lehman Baseline:**
```bash
python baselines/wl_baseline.py --dataset aifb --iterations 3
```

## Project Structure

```
├── experiments/          # Main training scripts
├── torch_rgcn/          # Core R-GCN model implementations
├── baselines/           # Baseline methods for comparison
├── configs/             # YAML configuration files for experiments
├── data/                # Dataset storage (AIFB, MUTAG, etc.)
└── utils/               # Data loading and utility functions
```

## Datasets

- **AIFB**: Linked data from AIFB organization
- **AM**: Active Members dataset
- **BGS**: British Geological Survey dataset  
- **MUTAG**: Mutagenesis dataset

## Model Variants

- **Standard R-GCN**: Full relational convolution with basis decomposition
- **Compact R-GCN (c-rgcn)**: Block diagonal decomposition for efficiency
- **Embedding R-GCN**: With initial node embeddings

## Performance

Typical accuracy ranges:
- R-GCN: 85-95% depending on dataset
- Feature Baseline: 40-60% (no graph structure)
- Weisfeiler-Lehman: 60-80%

## References

- [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)

## License

See LICENSE file for details.






































































































- **Embedding R-GCN**: With initial node embeddings- **Compact R-GCN (c-rgcn)**: Block diagonal decomposition for efficiency- **Standard R-GCN**: Full relational convolution with basis decomposition## Model Variants- **MUTAG**: Mutagenesis dataset- **BGS**: British Geological Survey dataset  - **AM**: Active Members dataset- **AIFB**: Linked data from AIFB organization## Datasets```└── utils/               # Data loading and utility functions├── data/                # Dataset storage (AIFB, MUTAG, etc.)├── configs/             # YAML configuration files for experiments├── baselines/           # Baseline methods for comparison├── torch_rgcn/          # Core R-GCN model implementations├── experiments/          # Main training scripts```## Project Structure```python baselines/wl_baseline.py --dataset aifb --iterations 3```bash**Weisfeiler-Lehman Baseline:**```python baselines/feat_baseline.py --dataset mutagpython baselines/feat_baseline.py --dataset aifb```bash**Feature-based Baseline:**### Run Baselines```python experiments/classify_nodes.py with configs/rgcn/nc-MUTAG.yamlpython experiments/classify_nodes.py with configs/rgcn/nc-BGS.yamlpython experiments/classify_nodes.py with configs/rgcn/nc-AM.yamlpython experiments/classify_nodes.py with configs/rgcn/nc-AIFB.yaml```bash### Train R-GCN Model```bash get_data.sh```bash### Download Data## Usage```pip install scikit-learn rdflib sacred pyyamlpip install torch==1.7.1 torch-geometric==1.7.0 torch-scatter==2.0.6 torch-sparse==0.6.9```bashAlternatively, install manually:```conda activate torch_rgcn_venvconda env create -f environment.yml```bash### Setup with Conda- PyTorch Geometric 1.7+- PyTorch 1.3+- Python 3.7+### Requirements## Installation- **Experiment Tracking**: Sacred integration for logging and reproduction- **Configuration-based Training**: YAML configs for easy experiment management- **Baseline Methods**: Feature-based and Weisfeiler-Lehman baselines for comparison- **Multiple Datasets**: Support for AIFB, AM, BGS, and MUTAG datasets- **R-GCN Model**: Full R-GCN implementation with relational convolutions and decomposition strategies## FeaturesThis project implements R-GCN as described in [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) for entity classification tasks. It demonstrates how to classify nodes in knowledge graphs with multiple types of relations.## OverviewPyTorch implementation of Relational Graph Convolutional Networks (R-GCN) for node classification on RDF datasets.
This project implements Relational Graph Convolutional Networks (R-GCN) for node classification tasks on RDF knowledge graphs. It includes both the main RGCN model and baseline methods for comparison.

## Overview

The R-GCN model extends standard Graph Convolutional Networks to handle typed relations in knowledge graphs, making it suitable for multi-relational graph data. This implementation provides node classification on semantic web datasets.

**Reference**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)

## Datasets

Supported datasets for node classification:
- **AIFB**: SWAT semantic web ontology (~176K triples)
- **AM**: Active Machine ontology (~297K triples)  
- **BGS**: British Geological Survey (~100K triples)
- **MUTAG**: Molecular graph dataset (~391K triples)

## Project Structure

```
torch_rgcn/           - Core R-GCN implementation (layers, models, utilities)
baselines/            - Baseline methods (feature-based, Weisfeiler-Lehman)
configs/              - YAML configuration files for experiments
data/                 - Dataset files (RDF triples, labeled samples)
experiments/          - Main training and evaluation scripts
utils/                - Data loading and utility functions
```

## Installation

### Using Conda

```bash
conda env create -f environment.yml
conda activate torch_rgcn_venv
```

### Using pip

```bash
pip install -r requirements.txt
```

**Key Dependencies**: PyTorch 1.3+, RDFlib, scikit-learn, sacred (experiment tracking)

## Usage

### Train R-GCN Model

```bash
python experiments/classify_nodes.py with configs/rgcn/nc-AIFB.yaml
```

### Run Baselines

Feature-based baseline:
```bash
python baselines/feat_baseline.py --dataset aifb --classifier logistic
```

Weisfeiler-Lehman baseline:
```bash
python baselines/wl_baseline.py --dataset aifb --iterations 3
```

## Configuration

Edit YAML files in `configs/` to modify:
- Model architecture (hidden size, num layers)
- Training parameters (learning rate, epochs, batch size)
- Regularization (L2 penalty, dropout)
- Dataset selection

Example:
```yaml
dataset:
  name: aifb
rgcn:
  hidden_size: 16
  num_layers: 2
training:
  epochs: 50
  lr: 0.01
```

## Expected Performance

Results depend on dataset and model configuration:
- Feature baseline: ~40-60% accuracy (no graph structure)
- Weisfeiler-Lehman: ~60-75% accuracy
- R-GCN: ~75-90% accuracy

## Key Files

- `torch_rgcn/models.py` - NodeClassifier and model definitions
- `experiments/classify_nodes.py` - Main training loop
- `utils/data.py` - RDF data loading and preprocessing
- `baselines/feat_baseline.py` - Feature-based baseline
- `baselines/wl_baseline.py` - Graph kernel baseline

## Notes

- Uses Sacred framework for experiment tracking and reproducibility
- CPU and GPU supported
- RDF triples formatted in NTriples format (.nt)
- Train/test split included in datasets

