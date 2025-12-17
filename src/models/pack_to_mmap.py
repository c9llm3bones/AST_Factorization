import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vec-dir", required=True, help="Directory with individual .npy vectors")
    p.add_argument("--out", required=True, help="Output directory for mmap and ids.json")
    args = p.parse_args()

    vec_dir = args.vec_dir
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(vec_dir) if f.endswith(".npy"))
    assert files, f"No .npy files found in {vec_dir}"

    first = np.load(os.path.join(vec_dir, files[0]), allow_pickle=False)
    dim = first.shape[0]

    mmap_path = os.path.join(out_dir, "vectors.npy")
    id_map_path = os.path.join(out_dir, "ids.json")

    mmap = open_memmap(
        mmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(files), dim),
    )

    id2row = {}

    for i, fname in enumerate(tqdm(files, desc="Packing vectors")):
        id_ = os.path.splitext(fname)[0]
        vec = np.load(os.path.join(vec_dir, fname), allow_pickle=False).astype(np.float32, copy=False)
        if vec.shape[0] != dim:
            print(vec.shape)
            raise ValueError(f"Dim mismatch for {id_}: got {vec.shape[0]}, expected {dim}")
        mmap[i] = vec
        id2row[id_] = i

    mmap.flush()

    with open(id_map_path, "w", encoding="utf8") as f:
        json.dump(id2row, f)

    print(f"Saved mmap to {mmap_path}")
    print(f"Saved id map to {id_map_path}")
    print(f"Total vectors: {len(id2row)}, dim={dim}")

if __name__ == "__main__":
    main()
