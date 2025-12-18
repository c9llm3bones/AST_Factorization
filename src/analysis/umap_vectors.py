import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from umap import UMAP

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


def highlight_clone_pair(ax, emb, ids, id1, id2):
    if id1 not in ids or id2 not in ids:
        return False

    i1 = ids.index(id1)
    i2 = ids.index(id2)

    x = emb[[i1, i2], 0]
    y = emb[[i1, i2], 1]

    ax.scatter(x, y, s=80, c="red", zorder=5)
    ax.plot(x, y, c="red", lw=1.5, alpha=0.8)

    ax.text(x[0], y[0], id1, fontsize=8, color="red")
    ax.text(x[1], y[1], id2, fontsize=8, color="red")

    return True

def umap_for_method(
    method: str,
    variant: str,
    split: str,
    per_method: int,
    pca_dim: int,
    seed: int,
    tw_k: int = 15,
):
    vec_root = f"vectors/{method}_{variant}/{split}_mmap"
    X_mmap, id2row = load_mmap(vec_root)

    X_sample, rows = sample_rows(X_mmap, per_method, seed)

    if X_sample.dtype != np.float32:
        X_sample = X_sample.astype(np.float32, copy=False)

    pca_dim_eff = max(2, min(pca_dim, X_sample.shape[1], X_sample.shape[0] - 1))
    X_red = PCA(
        n_components=pca_dim_eff,
        random_state=seed
    ).fit_transform(X_sample)

    umap = UMAP(
        n_components=2,
        n_neighbors=tw_k,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    emb = umap.fit_transform(X_red)

    tw = trustworthiness(
        X_red,
        emb,
        n_neighbors=tw_k,
        metric="cosine",
    )

    row2id = {r: i for i, r in id2row.items()}
    ids = [row2id.get(int(r)) for r in rows]

    return emb, ids, tw

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--variant", default="normalized")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--per-method", type=int, default=8000)
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-k", type=int, default=15)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--out", default="plots/umap_methods.png")
    p.add_argument("--id1", type=str)
    p.add_argument("--id2", type=str)

    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_methods = len(args.methods)
    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    sns.set(style="white", rc={"figure.figsize": (5 * ncols, 5 * nrows)})
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig)

    for i, method in enumerate(args.methods):
        if i < 3:
            r = 0
            c0 = i * 2
            ax = fig.add_subplot(gs[r, c0:c0 + 2])
        else:
            r = 1
            c0 = (i - 3) * 3
            ax = fig.add_subplot(gs[r, c0:c0 + 3])

        try:
            emb, ids, tw = umap_for_method(
                method=method,
                variant=args.variant,
                split=args.split,
                per_method=args.per_method,
                pca_dim=args.pca,
                seed=args.seed,
            )

            ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.4)

            title = f"{method.upper()}  TW={tw:.3f}"
            if args.id1 and args.id2:
                _ = highlight_clone_pair(ax, emb, ids, args.id1, args.id2)
                # if found:
                #     title += " (clone)"

            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        except Exception as e:
            ax.set_title(f"{method.upper()} error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])


    # for j in range(n_methods, nrows * ncols):
    #     r, c = divmod(j, ncols)
    #     axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved UMAP grid to {args.out}")

if __name__ == "__main__":
    main()
