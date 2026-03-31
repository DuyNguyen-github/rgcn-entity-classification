# Entity Classification Baselines

This directory contains implementations of baseline methods for evaluating node classification performance.

## Baselines Implemented

### 1. Feature-based Baseline (`feat_baseline.py`)

**Concept**: Use only node attributes/features without leveraging graph structure.

**Key Points**:
- Extracts node properties (predicates) from RDF triples
- Creates feature vectors as one-hot encodings of node properties
- Tests multiple classifiers: Logistic Regression, Random Forest, SVM
- Provides lower bound on performance

**Usage**:
```bash
python baselines/feat_baseline.py --dataset aifb --classifier logistic
python baselines/feat_baseline.py --dataset mutag --classifier rf
python baselines/feat_baseline.py --dataset aifb --classifier svm
```

**Expected Performance**:
- Lower than graph-based methods (ignores structure)
- Depends heavily on feature quality
- Typical: 40-60% accuracy

**Algorithm**:
1. Load RDF triples
2. Extract unique relations (properties)
3. For each node, count participation in each relation
4. Create feature matrix (nodes × relations)
5. Scale features (StandardScaler)
6. Train classifier
7. Evaluate on test set

### 2. Weisfeiler-Lehman Baseline (`wl_baseline.py`)

**Concept**: Graph kernel method that captures local and global graph structure.

**Key Points**:
- Classic algorithm combining node features with graph structure
- Performs iterative refinement of node labels
- Creates feature vectors from label histograms
- More principled than features alone, but simpler than neural methods

**Usage**:
```bash
python baselines/wl_baseline.py --dataset aifb --classifier logistic --iterations 3
python baselines/wl_baseline.py --dataset mutag --classifier rf --iterations 2
python baselines/wl_baseline.py --dataset aifb --classifier svm --iterations 5
```

**Expected Performance**:
- Better than features alone (uses graph structure)
- Typically 50-80% accuracy
- Varies from RGCN (different aggregation strategy)
- Competitive baseline for graph methods

**Algorithm**:
1. Load RDF triples as undirected graph
2. Initialize node labels with degrees
3. For h iterations:
   - For each node: Create multiset = [own_label, sorted(neighbor_labels)]
   - Hash multiset to get new label
   - Accumulate all labels seen
4. Create feature vectors from label histograms
5. Train classifier on concatenated features
6. Evaluate

**WL Iterations Impact**:
- More iterations = captures higher-order structure
- Trade-off between performance and time
- Typical: 2-5 iterations

### 3. RGCN (in experiments/classify_nodes.py)

**Concept**: Relational Graph Convolutional Networks - the main method.

**Features**:
- Type-specific neighbor aggregation
- Multiple configurations: rgcn, c-rgcn (compressed), e-rgcn (embedding-based)
- State-of-the-art for relational graph tasks

**Expected Performance**:
- Best performance (uses structure and relations)
- ~85-90% on AIFB, ~60-70% on MUTAG
- More computationally expensive

---

## Comparison Script

Run all methods and compare results:

```bash
# Run all methods on all datasets
python compare_results.py

# Run specific datasets and classifiers
python compare_results.py --datasets aifb mutag --classifiers logistic rf

# Run only baselines (faster)
python compare_results.py --baselines-only

# Skip specific methods
python compare_results.py --rgcn-only  # Only RGCN
```

**Output**:
- Console comparison table
- `comparison_results.json` with detailed metrics

---

## Quick Start (Windows)

Since bash scripts may not work on Windows, use the Python runner:

```bash
# Run all experiments
python run_experiments_windows.py

# Run specific datasets only
python run_experiments_windows.py --datasets aifb mutag

# Skip certain methods (faster)
python run_experiments_windows.py --no-rgcn --no-wl

# Check environment
python run_experiments_windows.py --datasets aifb --no-rgcn --no-wl
```

---

## Quick Start (Linux/Mac)

```bash
# Run all experiments
bash run_experiments.sh

# Run specific datasets
bash run_experiments.sh aifb
bash run_experiments.sh mutag
```

---

## Interpreting Results

### Metrics

- **Accuracy**: Percentage of correctly classified nodes
- **Precision (macro)**: Average precision across all classes
- **Recall (macro)**: Average recall across all classes
- **F1-score (macro)**: Harmonic mean of precision and recall

### Expected Results

| Dataset | Feat | WL | RGCN |
|---------|------|-------|------|
| AIFB    | 45%  | 65-75% | 85-90% |
| MUTAG   | 50%  | 55-65% | 60-70% |

**What this tells us**:
1. **Feat << WL**: Features alone insufficient, structure matters
2. **WL < RGCN**: Relational information critical
3. **RGCN best**: Neural methods leverage data better

### Class-wise Performance

Pay attention to per-class results:
- Are some classes harder to classify?
- Is performance balanced or imbalanced?
- Does RGCN improve all classes equally?

---

## Implementation Details

### Feature Extraction (Feat Baseline)

Features are extracted from RDF triples:
- Each unique predicate becomes a feature
- For each node, count how many times it appears in each predicate
- Normalized by StandardScaler

Example (AIFB):
```
Node: person_123
Features:
  - affiliation_count: 2
  - researchInterest_count: 5
  - publications_count: 12
  ...
```

### Weisfeiler-Lehman Kernel

**WL Iteration Process**:

Iteration 0:
```
Node 1: label = degree(1) = 3
Node 2: label = degree(2) = 2
```

Iteration 1:
```
Node 1: neighbors = {label(1), neighbors_labels}
        = {3, [2, 2, 4]}  # sorted
        = hash("3,2,2,4") = new_label_1

Node 2: neighbors = {label(2), neighbors_labels}
        = {2, [3, 4, 4]}
        = hash("2,3,4,4") = new_label_2
```

Final feature vector: accumulate all labels across all iterations

### Classifier Choices

- **Logistic Regression**: Fast, interpretable, works well
- **Random Forest**: Non-linear, robust, no scaling needed
- **SVM with RBF**: Powerful but slower, needs tuning

---

## Troubleshooting

### Feature Baseline Issues

**Problem**: Very low accuracy (< 30%)
- **Cause**: Features may not be informative
- **Solution**: Check feature statistics, try Random Forest

**Problem**: Out of memory
- **Cause**: Many features × nodes
- **Solution**: Use PCA for dimensionality reduction

### WL Baseline Issues

**Problem**: Slow execution
- **Cause**: Too many iterations on large graph
- **Solution**: Reduce `--iterations` parameter

**Problem**: No improvement over features
- **Cause**: Graph may be sparse or disconnected
- **Solution**: Check data loading, increase iterations

### RGCN Issues

**Problem**: CUDA out of memory
- **Cause**: GPU memory insufficient
- **Solution**: Set `use_cuda: False` in config

**Problem**: Very poor performance
- **Cause**: Hyperparameters not tuned
- **Solution**: Increase `hidden_size`, adjust `learn_rate`

---

## Dataset Information

### AIFB (AI ontology)
- **Nodes**: ~176
- **Relations**: ~150
- **Classes**: 4 (affiliations)
- **Task**: Predict person's affiliation
- **Difficulty**: Relatively easy (well-structured)

### MUTAG (Chemical compounds)
- **Nodes**: ~27K
- **Relations**: 46
- **Classes**: 2 (mutagenic: yes/no)
- **Task**: Predict compound mutagenicity
- **Difficulty**: Harder (larger, sparse)

---

## References

- **RGCN Paper**: Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (2017)
- **WL Kernel**: Weisfeiler & Lehman, "A reduction of a graph to a canonical form and an algebra arising during this reduction" (1968)
- **This Repository**: Thanapalasingam, "Reproducible Relational Graph Convolutional Networks" (2021)

