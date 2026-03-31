"""
Feature-based Baseline for Node Classification

This baseline uses ONLY node features (attributes) without considering graph structure.
It serves as a lower bound - showing what performance is achievable without leveraging
the relational information that RGCN and other graph methods use.

Classifiers tested:
1. Logistic Regression (simple, interpretable)
2. Random Forest (non-linear, handles mixed features)
3. SVM with RBF kernel (powerful non-linear classifier)
"""

import os
import sys
import pickle
import argparse
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import rdflib as rdf
from rdflib import URIRef
import time
import pathlib

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_node_classification_data, locate_file

def extract_node_features(name, n2i):
    """
    Extract node features from the RDF data.
    Features are properties (predicates) connecting to each node.
    
    Args:
        name: Dataset name ('aifb', 'mutag', etc.)
        n2i: Node to index mapping
        
    Returns:
        features: numpy array of shape (num_nodes, num_features)
    """
    # Map dataset name to file paths
    if name.lower() == 'aifb':
        data_file = locate_file('data/aifb/aifb_stripped.nt.gz')
    elif name.lower() == 'mutag':
        data_file = locate_file('data/mutag/mutag_stripped.nt.gz')
    elif name.lower() == 'am':
        data_file = locate_file('data/am/am_stripped.nt.gz')
    elif name.lower() == 'bgs':
        data_file = locate_file('data/bgs/bgs_stripped.nt.gz')
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Parse RDF graph
    graph = rdf.Graph()
    print(f"Loading RDF graph from {data_file}...")
    # graph.parse(data_file, format='nt')
    import gzip

    with gzip.open(data_file, 'rb') as f:
        graph.parse(f, format='nt')
    print(f"Graph loaded: {len(graph)} triples")
    
    # Extract unique relations (properties)
    relations = set()
    for s, p, o in graph.triples((None, None, None)):
        rel_str = str(p)
        relations.add(rel_str)
    
    relations = sorted(list(relations))
    num_relations = len(relations)
    rel_to_idx = {r: i for i, r in enumerate(relations)}
    
    print(f"Found {num_relations} unique relations")
    
    # Create feature vectors: one-hot encoding of properties
    num_nodes = len(n2i)
    features = np.zeros((num_nodes, num_relations), dtype=np.float32)
    
    # For each node, mark which relations it participates in
    relation_count = defaultdict(int)
    for s, p, o in graph.triples((None, None, None)):
        rel_str = str(p)
        rel_idx = rel_to_idx[rel_str]
        relation_count[rel_str] += 1
        
        # Mark that subject uses this relation
        s_str = str(s) if isinstance(s, URIRef) else s.n3()
        if s_str in n2i:
            features[n2i[s_str], rel_idx] += 1
        
        # Mark that object uses this relation (in-degree)
        o_str = str(o) if isinstance(o, URIRef) else o.n3()
        if o_str in n2i:
            features[n2i[o_str], rel_idx] += 1
    
    print(f"\nRelation statistics:")
    for rel, count in sorted(relation_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {rel}: {count} occurrences")
    
    return features, relations

def run_feat_baseline(dataset_name, classifier_name='logistic'):
    """
    Run feature-based baseline for node classification.
    
    Args:
        dataset_name: 'aifb' or 'mutag'
        classifier_name: 'logistic', 'rf', or 'svm'
    """
    print(f"\n{'='*60}")
    print(f"Feature-based Baseline: {dataset_name.upper()}")
    print(f"Classifier: {classifier_name}")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading {dataset_name} data...")
    triples, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(
        dataset_name, use_test_set=True, prune=False)
    
    print(f"  Loaded {len(n2i)} nodes, {len(r2i)} relations")
    print(f"  Training samples: {len(train)}")
    print(f"  Test samples: {len(test)}")
    
    # Extract node features
    print(f"\nExtracting node features...")
    t_start = time.time()
    X, relations = extract_node_features(dataset_name, n2i)
    print(f"Feature extraction took {time.time() - t_start:.2f}s")
    print(f"Feature shape: {X.shape}")
    
    # Prepare training and test data
    train_idx = np.array([n2i[node] for node, _ in train.items()])
    train_labels = np.array([label for _, label in train.items()])
    
    test_idx = np.array([n2i[node] for node, _ in test.items()])
    test_labels = np.array([label for _, label in test.items()])
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    print(f"\nData prepared:")
    print(f"  X_train shape: {X_train.shape}, labels: {len(np.unique(train_labels))}")
    print(f"  X_test shape: {X_test.shape}")
    
    # Feature scaling
    print(f"\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print(f"\nTraining {classifier_name} classifier...")
    t_start = time.time()
    
    if classifier_name.lower() == 'logistic':
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
    elif classifier_name.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        # RF doesn't need scaling
        X_train_scaled = X_train
        X_test_scaled = X_test
    elif classifier_name.lower() == 'svm':
        clf = SVC(kernel='rbf', random_state=42, C=10)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    clf.fit(X_train_scaled, train_labels)
    train_time = time.time() - t_start
    print(f"Training took {train_time:.2f}s")
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    t_start = time.time()
    predictions = clf.predict(X_test_scaled)
    eval_time = time.time() - t_start
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision_macro = precision_score(test_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(test_labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    
    print(f"\nResults ({classifier_name.upper()}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro):    {recall_macro:.4f}")
    print(f"  F1-score (macro):  {f1_macro:.4f}")
    print(f"  Evaluation time: {eval_time:.2f}s")
    
    # Per-class performance
    print(f"\nPer-class performance:")
    unique_labels = np.unique(test_labels)
    for label in unique_labels:
        mask = test_labels == label
        acc = accuracy_score(test_labels[mask], predictions[mask])
        print(f"  Class {label}: {acc:.4f} ({np.sum(mask)} samples)")
    
    return {
        'dataset': dataset_name,
        'classifier': classifier_name,
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'train_time': train_time,
        'eval_time': eval_time,
        'model': clf,
        'scaler': scaler
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature-based baseline for node classification')
    parser.add_argument('--dataset', default='aifb', choices=['aifb', 'mutag', 'am', 'bgs'],
                       help='Dataset name')
    parser.add_argument('--classifier', default='logistic', choices=['logistic', 'rf', 'svm'],
                       help='Classifier type')
    
    args = parser.parse_args()
    
    results = run_feat_baseline(args.dataset, args.classifier)
    print(f"\n{'='*60}\nBaseline completed successfully!\n{'='*60}")
