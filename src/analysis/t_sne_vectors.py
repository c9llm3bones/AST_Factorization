import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

def load_mmap(vec_root: str):
    X = np.load(os.path.join(vec_root, "vectors.npy"), mmap_mode="r")
    with open(os.path.join(vec_root, "ids.json"), "r", encoding="utf8") as f:
        id2row = json.load(f)
    return X, id2row

def sample_rows(X_mmap, n: int, seed: int = 42):
    N = X_mmap.shape[0]
    n_eff = min(n, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_eff, replace=False) if n_eff < N else np.arange(N)
    X = X_mmap[idx]
    return X, idx

def tsne_for_method(
    method: str,
    variant: str,
    split: str,
    per_method: int,
    pca_dim: int,
    seed: int,
    trust_k: int = 15,
):
    vec_root = f"vectors/{method}_{variant}/{split}_mmap"
    if not os.path.isdir(vec_root):
        raise FileNotFoundError(f"Not found: {vec_root}")

    X_mmap, _ = load_mmap(vec_root)
    X_sample, _ = sample_rows(X_mmap, per_method, seed)

    if X_sample.dtype != np.float32:
        X_sample = X_sample.astype(np.float32, copy=False)

    pca_dim_eff = max(2, min(pca_dim, X_sample.shape[1], X_sample.shape[0] - 1))
    X_pca = PCA(
        n_components=pca_dim_eff,
        svd_solver="auto",
        random_state=seed,
    ).fit_transform(X_sample)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, X_pca.shape[0] // 10)),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        n_iter=1000,
        verbose=0,
    )
    emb = tsne.fit_transform(X_pca)

    trust = trustworthiness(
        X_pca,
        emb,
        n_neighbors=min(trust_k, X_pca.shape[0] - 1),
        metric="cosine",
    )

    return emb, trust


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True, help="e.g. tk wl bon bop c2v")
    p.add_argument("--variant", default="normalized")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--per-method", type=int, default=2000)
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-k", type=int, default=15)
    p.add_argument("--out", default="plots/tsne_methods.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_methods = len(args.methods)
    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    sns.set(style="white", rc={"figure.figsize": (5 * ncols, 5 * nrows)})
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)

    for i, method in enumerate(args.methods):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        try:
            emb, trust = tsne_for_method(
                method=method,
                variant=args.variant,
                split=args.split,
                per_method=args.per_method,
                pca_dim=args.pca,
                seed=args.seed,
                trust_k=args.trust_k,
            )

            ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6)
            ax.set_title(
                f"{method.upper()} ({args.variant})\n"
                f"Trust={trust:.3f}"
            )
            ax.set_xticks([])
            ax.set_yticks([])

            print(f"[{method.upper()}] Trustworthiness = {trust:.4f}")

        except Exception as e:
            ax.set_title(f"{method.upper()} error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(n_methods, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved t-SNE grid to {args.out}")

if __name__ == "__main__":
    main()
