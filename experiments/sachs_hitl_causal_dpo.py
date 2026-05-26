#!/usr/bin/env python3
# End-to-end Sachs observational HITL experiment (11 nodes, 17 arcs reference).
# Uses your ParticlePosterior + 3-way BT likelihood.
# Optional: downloads Sachs observational data from bnlearn.

from __future__ import annotations

import argparse, gzip, json, os, time, urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from data.load_sachs import SACHS_NODES, load_sachs_observational

# ---- Your repo imports (adjust if your paths differ) ----
from inference.ParticlePosterior import ParticlePosterior
from generation.generation import screen_pairs_uncertain
from utils.utils import ess, entropy_categorical
from metrics.metrics import expected_true_class_prob, mean_brier_score
from metrics.structural_metrics import metrics_from_weighted_samples

from inference.static_baselines import init_static_schedule
from inference.candidate_selection import select_pair

import torch # For DAG GFN
from prior.train_dag_gfn import EdgePolicy
from prior.sample_dag_gfn import sample_dag_from_gfn


try:
    from prior.prior import sparse_prior_logprob
    _HAS_PRIOR = True
except Exception:
    _HAS_PRIOR = False


# From bnlearn model string:
# [PKC][PKA|PKC][Raf|PKC:PKA][Mek|PKC:PKA:Raf][Erk|Mek:PKA][Akt|Erk:PKA]
# [P38|PKC:PKA][Jnk|PKC:PKA][Plcg][PIP3|Plcg][PIP2|Plcg:PIP3]
# (bnlearn Sachs 2005 HOWTO)
REF_EDGES: List[Tuple[str, str]] = [
    ("PKC", "PKA"),
    ("PKC", "Raf"), ("PKA", "Raf"),
    ("PKC", "Mek"), ("PKA", "Mek"), ("Raf", "Mek"),
    ("Mek", "Erk"), ("PKA", "Erk"),
    ("Erk", "Akt"), ("PKA", "Akt"),
    ("PKC", "P38"), ("PKA", "P38"),
    ("PKC", "Jnk"), ("PKA", "Jnk"),
    ("Plcg", "PIP3"),
    ("Plcg", "PIP2"), ("PIP3", "PIP2"),
]


def build_adjacency(nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)
    for a, b in edges:
        A[idx[a], idx[b]] = 1
    return A


A_REF = build_adjacency(SACHS_NODES, REF_EDGES)


def oracle_label(i: int, j: int, A_ref: np.ndarray) -> int:
    """Matches your demo convention: 0=i->j, 1=j->i, 2=none."""
    if A_ref[i, j] == 1:
        return 0
    if A_ref[j, i] == 1:
        return 1
    return 2


# ---------------- Data download / load ----------------
SACHS_OBS_URL_GZ = "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz"


def maybe_download_sachs(data_dir: str, force: bool = False) -> str:
    os.makedirs(data_dir, exist_ok=True)
    gz_path = os.path.join(data_dir, "sachs.data.txt.gz")
    txt_path = os.path.join(data_dir, "sachs.data.txt")

    if (not force) and os.path.exists(txt_path):
        return txt_path

    if (not force) and os.path.exists(gz_path) and not os.path.exists(txt_path):
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())
        return txt_path

    print(f"Downloading {SACHS_OBS_URL_GZ} -> {gz_path}")
    urllib.request.urlretrieve(SACHS_OBS_URL_GZ, gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
        f_out.write(f_in.read())
    return txt_path


# ---------------- q0(W|X) via bootstrap + random order linear DAG ----------------

def ridge_coef(Xp: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if Xp.size == 0:
        return np.zeros((0,), dtype=float)
    XtX = Xp.T @ Xp
    XtX.flat[:: XtX.shape[0] + 1] += ridge
    Xty = Xp.T @ y
    return np.linalg.solve(XtX, Xty)


def sample_linear_dag(X: np.ndarray,
                      rng: np.random.Generator,
                      max_parents: int,
                      corr_screen_k: int,
                      ridge: float) -> np.ndarray:
    n, D = X.shape
    order = rng.permutation(D)
    W = np.zeros((D, D), dtype=float)

    for pos, j in enumerate(order):
        prev = order[:pos]
        if prev.size == 0:
            continue
        y = X[:, j]
        Xprev = X[:, prev]

        corrs = np.abs((Xprev.T @ y) / max(1, (n - 1)))
        k = min(corr_screen_k, prev.size)
        cand_idx = np.argpartition(-corrs, kth=k-1)[:k]
        cand_parents = prev[cand_idx]

        if cand_parents.size > max_parents:
            cand_corrs = corrs[cand_idx]
            top = np.argpartition(-cand_corrs, kth=max_parents-1)[:max_parents]
            parents = cand_parents[top]
        else:
            parents = cand_parents

        coef = ridge_coef(X[:, parents], y, ridge=ridge)
        for pnode, w in zip(parents, coef):
            if abs(w) > 1e-3:
                W[pnode, j] = float(w)

    return W


def make_q0_particles_bootstrap(X: np.ndarray,
                               S: int,
                               seed: int,
                               bootstrap_n: int | None,
                               max_parents: int,
                               corr_screen_k: int,
                               ridge: float) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n, _ = X.shape
    if bootstrap_n is None:
        bootstrap_n = n

    particles: List[np.ndarray] = []
    for _ in range(S):
        idx = rng.integers(0, n, size=bootstrap_n)
        Xb = X[idx]
        particles.append(sample_linear_dag(Xb, rng, max_parents, corr_screen_k, ridge))
    return particles


# builds DAG GFN prior
def sample_gfn(D:int, S: int, rng: np.random.Generator, max_edges: int=200):
    print("Sampling DAG GFN")
    ckpt = torch.load("prior/sachs_gfn_ckpt_std.pt", map_location="cpu")
    policy = EdgePolicy(D=D, hidden=256)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # sample S particles from GFN
    particles = [sample_dag_from_gfn(policy, D, rng, max_edges=max_edges) for _ in range(S)]

    return particles




# ---------------- HITL loop ----------------

@dataclass
class RunConfig:
    S: int
    T: int
    screen_k: int
    resample_threshold: float
    rejuvenate_samples: bool
    rejuvenate_steps: int
    max_parents: int
    corr_screen_k: int
    ridge: float
    bootstrap_n: int | None
    standardize_data: bool



def run_once(X: np.ndarray,
             policy: str,
             cfg: RunConfig,
             beta_edge: float,
             beta_dir: float,
             lam: float,
             seed: int,
             use_dag_gfn_prior: bool=False) -> Dict:
    rng = np.random.default_rng(seed)

    D = X.shape[1]

    if use_dag_gfn_prior:
        particles = sample_gfn(D=D, S=cfg.S, rng=rng)
    else:
        particles = make_q0_particles_bootstrap(
            X, S=cfg.S, seed=seed, bootstrap_n=cfg.bootstrap_n,
            max_parents=cfg.max_parents, corr_screen_k=cfg.corr_screen_k, ridge=cfg.ridge
        )

    posterior = ParticlePosterior(particles, weights=None)

    # --- snapshot t=0 marginals for before/after heatmaps ---
    posterior_marginals_init = posterior.edge_marginals().copy()
    init_metrics = metrics_from_weighted_samples(posterior.particles, posterior.weights, A_REF)

    history: List[Tuple[int, int, int]] = []
    logs: List[Dict] = []


    static_schedule = init_static_schedule(policy=policy, posterior=posterior, D=D,
                                           T=cfg.T, screen_k=cfg.screen_k,
                                           beta_edge=beta_edge, beta_dir=beta_dir,
                                           lam=lam, rng=rng)

    # --- prevent asking the same unordered pair twice ---
    asked: set[tuple[int, int]] = set()

    for t in range(1, cfg.T + 1):
        t0 = time.perf_counter()

        marg = posterior.edge_marginals()
        top_k = min(cfg.screen_k, D * (D - 1) // 2)
        cand = screen_pairs_uncertain(marg, top_k=top_k)

        # Remove previously queried unordered pairs {i,j}
        cand = [(i, j) for (i, j) in cand if (min(i, j), max(i, j)) not in asked]
        if not cand:
            raise RuntimeError("No candidate pairs left. Reduce T or increase screen_k.")

        (i, j), eig_val = select_pair(static_schedule=static_schedule,
                                      t=t,
                                      cand=cand,
                                      posterior=posterior,
                                      policy=policy,
                                      beta_edge=beta_edge,
                                      beta_dir=beta_dir,
                                      rng=rng,
                                      lam=lam)

        # record asked candidate
        asked.add((min(i, j), max(i, j)))

        y = oracle_label(i, j, A_REF)

        posterior.update_with_observation(i, j, y, beta_edge, beta_dir, lam)

        if ess(posterior.weights) / cfg.S < cfg.resample_threshold:
            posterior.resample(rng=rng)
            if cfg.rejuvenate_samples:
                if not _HAS_PRIOR:
                    raise RuntimeError("rejuvenate_samples=True but prior.prior.sparse_prior_logprob not importable")
                posterior.rejuvenate_particles(
                    q0_logprob=sparse_prior_logprob,
                    expert_history=history,
                    beta_edge=beta_edge,
                    beta_dir=beta_dir,
                    lam=lam,
                    round=t,
                    n_steps=cfg.rejuvenate_steps,
                    rng=rng,
                )

        history.append((i, j, y))

        etcp = expected_true_class_prob(posterior, beta_edge, beta_dir, lam, A_REF)
        brier = mean_brier_score(posterior, beta_edge, beta_dir, lam, A_REF)

        samp_metrics = metrics_from_weighted_samples(posterior.particles, posterior.weights, A_REF)
        avg_entropy = float(np.mean([
            entropy_categorical(posterior.predictive_answer_dist(a, b, beta_edge, beta_dir, lam))
            for (a, b) in cand
        ]))

        logs.append({
            "round": t,
            "pair": (int(i), int(j)),
            "answer_idx": int(y),
            "eig": float(eig_val),
            "avg_pred_entropy": avg_entropy,
            "ess": float(ess(posterior.weights)),
            "exp_true_class_prob": float(etcp),
            "brier": float(brier),
            "samp_skel_precision": float(samp_metrics["skel_precision"]),
            "samp_skel_recall": float(samp_metrics["skel_recall"]),
            "samp_skel_f1": float(samp_metrics["skel_f1"]),
            "samp_orient_precision": float(samp_metrics["orient_precision"]),
            "samp_orient_recall": float(samp_metrics["orient_recall"]),
            "samp_orient_f1": float(samp_metrics["orient_f1"]),
            "samp_shd": float(samp_metrics["shd"]),
        })

        t1 = time.perf_counter()
        print(f"[{policy}] t={t:02d}/{cfg.T} pair=({i},{j}) y={y} "
              f"EIG={eig_val:.4f} SHD={samp_metrics['shd']:.1f} "
              f"orientF1={samp_metrics['orient_f1']:.3f} dt={t1-t0:.3f}s")

    posterior_marginals_final = posterior.edge_marginals()
    return {
        "init": {
            "samp_shd": float(init_metrics["shd"]),
            "samp_skel_f1": float(init_metrics["skel_f1"]),
            "samp_orient_f1": float(init_metrics["orient_f1"]),
        },
        "policy": policy,
        "seed": seed,
        "settings": {
            "beta_edge": beta_edge,
            "beta_dir": beta_dir,
            "lam": lam,
            **cfg.__dict__,
        },
        "logs": logs,
        "final": logs[-1] if logs else {},
        # --- before/after snapshots ---
        "posterior_marginals_init": posterior_marginals_init.tolist(),
        "posterior_marginals_final": posterior_marginals_final.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--data_path", type=str, default="")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force_download", action="store_true")

    ap.add_argument("--outdir", type=str, default="results/results_sachs")

    ap.add_argument("--standardize_data", action="store_true")

    ap.add_argument("--S", type=int, default=500)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--screen_k", type=int, default=80)
    ap.add_argument("--resample_threshold", type=float, default=0.5)
    ap.add_argument("--rejuvenate_samples", action="store_true")
    ap.add_argument("--rejuvenate_steps", type=int, default=1)

    ap.add_argument("--max_parents", type=int, default=3)
    ap.add_argument("--corr_screen_k", type=int, default=6)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--bootstrap_n", type=int, default=0, help="0 => use full n")
    ap.add_argument("--use_dag_gfn_prior", action="store_true")

    ap.add_argument("--beta_edge", type=float, default=8.0)
    ap.add_argument("--beta_dir", type=float, default=-1.5)
    ap.add_argument("--lam", type=float, default=0.0)

    ap.add_argument("--policies", type=str, default="eig,uncertainty,random")
    ap.add_argument("--runs", type=int, default=5, help="runs per policy")
    ap.add_argument("--seed0", type=int, default=0)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.data_path:
        path = args.data_path
    elif args.download:
        path = maybe_download_sachs(args.data_dir, force=args.force_download)
    else:
        path = os.path.join(args.data_dir, "sachs.data.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Use --download or --data_path.")

    X = load_sachs_observational(path, standardize=args.standardize_data)
    print(f"Loaded observational Sachs: X.shape={X.shape} ({path})")

    cfg = RunConfig(
        S=args.S,
        T=args.T,
        screen_k=args.screen_k,
        resample_threshold=args.resample_threshold,
        rejuvenate_samples=args.rejuvenate_samples,
        rejuvenate_steps=args.rejuvenate_steps,
        max_parents=args.max_parents,
        corr_screen_k=args.corr_screen_k,
        ridge=args.ridge,
        bootstrap_n=None if args.bootstrap_n == 0 else int(args.bootstrap_n),
        standardize_data=args.standardize_data,
    )

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    # save metadata
    with open(os.path.join(args.outdir, "sachs_meta.json"), "w") as f:
        json.dump({
            "nodes": SACHS_NODES,
            "ref_edges": REF_EDGES,
            "ref_num_edges": len(REF_EDGES),
            "data_path": path,
        }, f, indent=2)

    all_runs: List[Dict] = []
    for policy in policies:
        for r in range(args.runs):
            seed = args.seed0 + r
            out = run_once(X=X, policy=policy, cfg=cfg,
                           beta_edge=args.beta_edge, beta_dir=args.beta_dir,
                           lam=args.lam, seed=seed,
                           use_dag_gfn_prior=args.use_dag_gfn_prior)
            all_runs.append(out)
            with open(os.path.join(args.outdir, f"sachs_{policy}_seed{seed}.json"), "w") as f:
                json.dump(out, f, indent=2)

    # summary
    summary: Dict[str, Dict[str, float]] = {}
    for policy in policies:
        finals = [r["final"] for r in all_runs if r["policy"] == policy]
        def _mean(key):
            vals = np.asarray([f.get(key, np.nan) for f in finals], float)
            return float(np.nanmean(vals))
        def _std(key):
            vals = np.asarray([f.get(key, np.nan) for f in finals], float)
            return float(np.nanstd(vals))
        summary[policy] = {
            "final_shd_mean": _mean("samp_shd"),
            "final_shd_std": _std("samp_shd"),
            "final_orient_f1_mean": _mean("samp_orient_f1"),
            "final_orient_f1_std": _std("samp_orient_f1"),
            "final_skel_f1_mean": _mean("samp_skel_f1"),
            "final_skel_f1_std": _std("samp_skel_f1"),
        }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Summary (mean ± std over runs) ===")
    for policy, s in summary.items():
        print(f"{policy:12s} SHD={s['final_shd_mean']:.2f}±{s['final_shd_std']:.2f} "
              f"orientF1={s['final_orient_f1_mean']:.3f}±{s['final_orient_f1_std']:.3f} "
              f"skelF1={s['final_skel_f1_mean']:.3f}±{s['final_skel_f1_std']:.3f}")

    print(f"Wrote results to: {args.outdir}")


if __name__ == "__main__":
    main()
