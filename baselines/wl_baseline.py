"""
Weisfeiler-Lehman (WL) Baseline for Node Classification
Fixed version (working - correct RDF handling)
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import hashlib

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_node_classification_data


class WeisfeilerLehmanKernel:
    def __init__(self, num_iterations=2, hash_size=1024):
        self.num_iterations = num_iterations
        self.hash_size = hash_size

    def _hash_labels(self, label_multiset):
        sorted_labels = sorted(label_multiset, key=lambda x: str(x))
        label_str = ','.join(str(l) for l in sorted_labels)
        return int(hashlib.md5(label_str.encode()).hexdigest(), 16) % self.hash_size

    def fit_transform(self, edges, num_nodes):
        print(f"Computing WL features ({self.num_iterations} iterations)...")

        # Build adjacency
        adj_list = defaultdict(set)
        for s, t in edges:
            adj_list[s].add(t)
            adj_list[t].add(s)

        # Init labels = degree
        labels = {i: len(adj_list[i]) for i in range(num_nodes)}

        all_labels_set = set(labels.values())
        label_to_id = {l: i for i, l in enumerate(sorted(all_labels_set))}

        features_list = []
        node_label_counts = defaultdict(Counter)

        # iteration 0
        for node in range(num_nodes):
            node_label_counts[node][label_to_id[labels[node]]] += 1

        features_list.append(node_label_counts.copy())
        print(f"  Iteration 0: {len(all_labels_set)} unique labels")

        # WL iterations
        for it in range(self.num_iterations):
            print(f"  Iteration {it+1}/{self.num_iterations}...")

            new_labels = {}
            iteration_labels = set()

            for node in range(num_nodes):
                neigh = [labels[n] for n in adj_list[node]]
                multiset = [labels[node]] + sorted(neigh, key=lambda x: str(x))
                new_l = self._hash_labels(multiset)
                new_labels[node] = new_l
                iteration_labels.add(new_l)

            labels = new_labels
            print(f"    Generated {len(iteration_labels)} unique labels")

            all_labels_set.update(iteration_labels)
            label_to_id = {l: i for i, l in enumerate(sorted(all_labels_set))}

            node_label_counts = defaultdict(Counter)
            for node in range(num_nodes):
                node_label_counts[node][label_to_id[labels[node]]] += 1

            features_list.append(node_label_counts.copy())

        # build feature matrix
        total_labels = len(all_labels_set)
        X = np.zeros((num_nodes, total_labels), dtype=np.float32)

        for node in range(num_nodes):
            for node_label_count in features_list:
                for lid, cnt in node_label_count[node].items():
                    X[node, lid] += cnt

        print(f"  Final feature dimension: {X.shape}")
        print(f"  Total unique labels: {total_labels}")

        return X


from rdflib import URIRef, Literal

def build_undirected_graph(triples):
    edges = []
    edge_set = set()

    for s, r, o in triples:

        # 🔥 s, o đã là index (int)
        if s != o:

            if (s, o) not in edge_set:
                edges.append((s, o))
                edge_set.add((s, o))

            if (o, s) not in edge_set:
                edges.append((o, s))
                edge_set.add((o, s))

    print(f"DEBUG: total edges = {len(edges)}")

    return edges


def run_wl_baseline(dataset_name, wl_iterations=3, classifier_name='logistic'):
    print(f"\n{'='*60}")
    print(f"Weisfeiler-Lehman Baseline: {dataset_name.upper()}")
    print(f"WL Iterations: {wl_iterations}")
    print(f"Classifier: {classifier_name}")
    print(f"{'='*60}")

    print(f"\nLoading {dataset_name} data...")
    triples, (n2i, i2n), (r2i, i2r), train, test = load_node_classification_data(
        dataset_name, use_test_set=True, prune=False)

    # # 🔥 FIX QUAN TRỌNG: convert n2i → URIRef
    # import rdflib
    # n2i = {rdflib.term.URIRef(k): v for k, v in n2i.items()}

    print(f"  Loaded {len(n2i)} nodes, {len(r2i)} relations, {len(triples)} triples")

    print("\nBuilding undirected graph...")
    edges = build_undirected_graph(triples)
    print(f"  Graph built: {len(edges)} edges")

    print("\nComputing WL features...")
    wl = WeisfeilerLehmanKernel(num_iterations=wl_iterations)

    X = wl.fit_transform(edges, len(n2i))

    # 🔥 FIX: convert train/test node → URIRef
    import rdflib
    train_idx = np.array([n2i[node] for node, _ in train.items()])
    train_labels = np.array(list(train.values()))

    test_idx = np.array([n2i[node] for node, _ in test.items()])
    test_labels = np.array(list(test.values()))

    X_train = X[train_idx]
    X_test = X[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if classifier_name == 'logistic':
        clf = LogisticRegression(max_iter=1000)
    elif classifier_name == 'rf':
        clf = RandomForestClassifier(n_estimators=100)
    else:
        clf = SVC()

    clf.fit(X_train, train_labels)
    pred = clf.predict(X_test)

    print("\nResults:")
    print("Accuracy:", accuracy_score(test_labels, pred))
    print("F1:", f1_score(test_labels, pred, average='macro'))

    return accuracy_score(test_labels, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weisfeiler-Lehman baseline for node classification')
    parser.add_argument('--dataset', default='aifb', choices=['aifb', 'mutag', 'am', 'bgs'],
                       help='Dataset name')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of WL iterations')
    parser.add_argument('--classifier', default='logistic', choices=['logistic', 'rf', 'svm'],
                       help='Classifier type')
    args = parser.parse_args()

    run_wl_baseline(args.dataset, args.iterations, args.classifier)