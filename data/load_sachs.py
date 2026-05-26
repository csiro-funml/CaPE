import numpy as np
from typing import List, Tuple

# ---------------- Sachs reference (bnlearn, 11 nodes / 17 arcs) ----------------
SACHS_NODES: List[str] = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]


def load_sachs_observational(path_txt: str, standardize: bool = True) -> np.ndarray:
    with open(path_txt, "r") as f:
        header = f.readline().strip().split()
    if header != SACHS_NODES:
        raise ValueError(f"Unexpected header. Got {header}, expected {SACHS_NODES}")

    X = np.loadtxt(path_txt, skiprows=1)
    if standardize:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    return X

def save_sachs_as_npz():
    X = load_sachs_observational(path_txt="./data/sachs.data.txt", standardize=False)
    np.savez("./data/sachs_observational.npz", X=X, allow_pickle=True)

if __name__ == "__main__":
    save_sachs_as_npz()


