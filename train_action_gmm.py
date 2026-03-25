"""
train_action_gmm.py — Train a Gaussian Mixture Model over action embeddings.

Replaces hardcoded ACTION_ANCHORS ({"question": [...], "reflection": [...]})
with emergent clusters discovered from the system's own output history.

Usage:
    python train_action_gmm.py [--data action_embeddings.jsonl] [--k 8] [--output action_gmm.pkl]

The trained GMM can be loaded by embeddings.py to classify actions geometrically
instead of via hardcoded anchor strings.

Requires: scikit-learn, numpy
"""

import json
import argparse
import pickle
import numpy as np
from pathlib import Path


def load_data(path: str) -> tuple:
    """Load action embeddings from JSONL file."""
    vecs = []
    latents = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            vecs.append(entry["vec"])
            latents.append(entry["latent"])
    return np.array(vecs), np.array(latents)


def train_gmm(vecs: np.ndarray, n_components: int = 8, random_state: int = 42):
    """Train GMM on action embeddings."""
    from sklearn.mixture import GaussianMixture
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",  # diagonal for efficiency with high-dim embeddings
        max_iter=200,
        n_init=5,
        random_state=random_state,
    )
    gmm.fit(vecs)
    return gmm


def analyze_clusters(gmm, vecs: np.ndarray, latents: np.ndarray):
    """Analyze what each cluster represents in terms of latent state."""
    labels = gmm.predict(vecs)
    n_clusters = gmm.n_components
    
    print(f"\n{'='*60}")
    print(f"ACTION SPACE ANALYSIS — {n_clusters} clusters from {len(vecs)} samples")
    print(f"{'='*60}\n")
    
    for k in range(n_clusters):
        mask = labels == k
        count = mask.sum()
        if count == 0:
            print(f"Cluster {k}: EMPTY")
            continue
        
        cluster_latents = latents[mask]
        mean_s = cluster_latents[:, 0].mean()
        mean_v = cluster_latents[:, 1].mean()
        mean_vel = cluster_latents[:, 2].mean()
        
        # Characterize the cluster by its latent state signature
        traits = []
        if mean_s > 0.5: traits.append("high-surprise")
        elif mean_s < 0.25: traits.append("low-surprise")
        if mean_v > 0.1: traits.append("positive")
        elif mean_v < -0.1: traits.append("negative")
        if mean_vel > 0.4: traits.append("fast")
        elif mean_vel < 0.15: traits.append("slow")
        
        trait_str = ", ".join(traits) if traits else "neutral"
        
        print(f"Cluster {k}: {count} samples ({count/len(vecs)*100:.0f}%)")
        print(f"  Latent signature: S={mean_s:.2f}  V={mean_v:+.2f}  Vel={mean_vel:.2f}")
        print(f"  Character: {trait_str}")
        print()
    
    # BIC/AIC for model selection
    print(f"BIC: {gmm.bic(vecs):.0f}")
    print(f"AIC: {gmm.aic(vecs):.0f}")
    
    return labels


def find_optimal_k(vecs: np.ndarray, k_range: range = range(3, 15)):
    """Find optimal number of clusters via BIC."""
    from sklearn.mixture import GaussianMixture
    
    bics = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="diag", max_iter=200, n_init=3, random_state=42)
        gmm.fit(vecs)
        bics.append((k, gmm.bic(vecs)))
        print(f"  k={k}: BIC={gmm.bic(vecs):.0f}")
    
    best_k = min(bics, key=lambda x: x[1])[0]
    print(f"\nOptimal k={best_k} (lowest BIC)")
    return best_k


def main():
    parser = argparse.ArgumentParser(description="Train GMM over action embeddings")
    parser.add_argument("--data", type=str, default="action_embeddings.jsonl", help="Path to JSONL data file")
    parser.add_argument("--k", type=int, default=0, help="Number of clusters (0 = auto-select via BIC)")
    parser.add_argument("--output", type=str, default="action_gmm.pkl", help="Output pickle file")
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        print(f"Run the system for 50+ cycles to generate data, then re-run this script.")
        return
    
    vecs, latents = load_data(args.data)
    print(f"Loaded {len(vecs)} action embeddings ({vecs.shape[1]} dims)")
    
    if len(vecs) < 20:
        print(f"Need at least 20 samples for meaningful clustering. Current: {len(vecs)}")
        return
    
    if args.k == 0:
        print("\nFinding optimal k...")
        k = find_optimal_k(vecs)
    else:
        k = args.k
    
    print(f"\nTraining GMM with k={k}...")
    gmm = train_gmm(vecs, n_components=k)
    
    labels = analyze_clusters(gmm, vecs, latents)
    
    # Save
    with open(args.output, "wb") as f:
        pickle.dump({"gmm": gmm, "k": k, "n_samples": len(vecs), "dim": vecs.shape[1]}, f)
    print(f"\nSaved GMM to {args.output}")
    print(f"Load in embeddings.py: gmm = pickle.load(open('{args.output}', 'rb'))['gmm']")


if __name__ == "__main__":
    main()
