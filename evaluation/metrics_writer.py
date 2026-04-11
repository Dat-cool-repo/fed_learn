"""
metrics_writer.py  —  Federated PEFT Training · Results Logger
==============================================================
Drop this file into your training directory and import it.

USAGE (3 steps):
    1.  Create a logger at the start of each experiment run
    2.  Call .log_round() after every federation round
    3.  Call .save() when the run is done

EXAMPLE:
    from metrics_writer import MetricsWriter

    logger = MetricsWriter(seed=1, regime="mild", method="fedavg", peft="lora")

    for round_num in range(1, num_rounds + 1):
        # ... your federated training round ...

        logger.log_round(
            round_num       = round_num,
            rouge_l         = 0.312,                    # float: mean ROUGE-L this round
            update_norms    = [0.042, 0.038, 0.051],    # list[float]: one per participating client
            cosine_disagree = [0.21,  0.18,  0.25],    # list[float]: one per participating client
        )

    logger.save()   # writes results/seed_1/regime_mild/method_fedavg/peft_lora/metrics.json

That's it. The file is written automatically to the right folder.
"""

import json
import math
import numpy as np
from pathlib import Path


# ── Folder schema (must match 03_analysis.ipynb) ────────────────────────────
#   results/seed_{s}/regime_{r}/method_{m}/peft_{p}/metrics.json
RESULTS_ROOT = Path("results")


class MetricsWriter:
    """
    Accumulates per-round metrics during a federated training run and
    writes them to the agreed metrics.json schema.

    Parameters
    ----------
    seed    : int   — random seed for this run  (1, 2, or 3)
    regime  : str   — heterogeneity level        ("mild", "medium", "hard")
    method  : str   — aggregation algorithm      ("fedavg", "fedprox", "scaffold")
    peft    : str   — PEFT technique             ("lora", "softprompt")
    root    : Path  — root results folder        (default: "results/")
    """

    VALID = {
        "seed":   {1, 2},
        "regime": {"mild", "hard"},
        "method": {"fedavg", "fedprox", "scaffold"},
        "peft":   {"lora", "softprompt"},
    }

    def __init__(
        self,
        seed:   int,
        regime: str,
        method: str,
        peft:   str,
        root:   Path = RESULTS_ROOT,
    ):
        self._validate(seed, regime, method, peft)

        self.seed   = seed
        self.regime = regime
        self.method = method
        self.peft   = peft

        self.out_path: Path = (
            root
            / f"seed_{seed}"
            / f"regime_{regime}"
            / f"method_{method}"
            / f"peft_{peft}"
            / "metrics.json"
        )

        # Internal accumulators
        self._rounds:      list[int]         = []
        self._rouge_l:     list[float]       = []
        self._norms:       list[list[float]] = []   # [round_t][client_k]
        self._cosine_dis:  list[list[float]] = []   # [round_t][client_k]

        print(f"MetricsWriter ready  →  {self.out_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log_round(
        self,
        round_num:       int,
        rouge_l:         float,
        update_norms:    list[float],
        cosine_disagree: list[float],
    ) -> None:
        """
        Record metrics for one federation round.

        Parameters
        ----------
        round_num       : Current round number (1-indexed).
        rouge_l         : Mean ROUGE-L score across the evaluation set this round.
                          Compute with rouge_scorer or pass the value from eval.
        update_norms    : Per-client L2 update norm, normalized by sqrt(d).
                          One float per *participating* client this round.
                          Formula: ||w_k^t - w^t||_2 / sqrt(d)
        cosine_disagree : Per-client cosine disagreement from mean update.
                          One float per *participating* client this round.
                          Formula: 1 - cos(Δw_k, mean(Δw))
        """
        if len(update_norms) != len(cosine_disagree):
            raise ValueError(
                f"update_norms and cosine_disagree must have the same length "
                f"(got {len(update_norms)} vs {len(cosine_disagree)}). "
                f"Both should have one entry per participating client."
            )

        self._rounds.append(int(round_num))
        self._rouge_l.append(float(rouge_l))
        self._norms.append([float(v) for v in update_norms])
        self._cosine_dis.append([float(v) for v in cosine_disagree])

    def save(self) -> Path:
        """
        Write metrics.json to disk. Safe to call multiple times (overwrites).
        Returns the path to the written file.
        """
        if not self._rounds:
            raise RuntimeError("No rounds logged yet — call log_round() before save().")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "rounds":                        self._rounds,
            "rouge_l_per_round":             self._rouge_l,
            "update_norms_per_round":        self._norms,
            "cosine_disagreement_per_round": self._cosine_dis,
        }

        with open(self.out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved  {len(self._rounds)} rounds  →  {self.out_path}")
        return self.out_path

    def log_round_from_state_dicts(
        self,
        round_num:          int,
        rouge_l:            float,
        client_deltas:      list[dict],   # list of {layer: tensor/ndarray}
        global_weights:     dict,         # {layer: tensor/ndarray}  (unused, for reference)
    ) -> None:
        """
        Convenience overload: computes update_norms and cosine_disagree
        directly from raw PyTorch state dicts (or numpy dicts).

        Parameters
        ----------
        client_deltas   : List of delta dicts, one per participating client.
                          delta = local_weights - global_weights  (compute before calling)
        global_weights  : The global model weights for this round (used for d).

        Example
        -------
            delta_k = {name: local_sd[name] - global_sd[name] for name in global_sd}
            logger.log_round_from_state_dicts(
                round_num=t, rouge_l=0.31,
                client_deltas=[delta_k1, delta_k2, ...],
                global_weights=global_sd,
            )
        """
        norms, cosines = _compute_drift_metrics(client_deltas)
        self.log_round(round_num, rouge_l, norms, cosines)

    # ── Private ───────────────────────────────────────────────────────────────

    @classmethod
    def _validate(cls, seed, regime, method, peft):
        errors = []
        if seed not in cls.VALID["seed"]:
            errors.append(f"seed={seed!r} must be one of {cls.VALID['seed']}")
        if regime not in cls.VALID["regime"]:
            errors.append(f"regime={regime!r} must be one of {cls.VALID['regime']}")
        if method not in cls.VALID["method"]:
            errors.append(f"method={method!r} must be one of {cls.VALID['method']}")
        if peft not in cls.VALID["peft"]:
            errors.append(f"peft={peft!r} must be one of {cls.VALID['peft']}")
        if errors:
            raise ValueError("MetricsWriter config error:\n  " + "\n  ".join(errors))


# ── Drift helpers (mirror of 03_analysis.ipynb, usable from training code) ──

def _flatten(delta_w: dict) -> np.ndarray:
    """Flatten a param dict → 1-D numpy array. Handles torch tensors."""
    parts = []
    for v in delta_w.values():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        parts.append(np.asarray(v).ravel())
    return np.concatenate(parts)


def compute_update_norm(delta_w: dict) -> float:
    """
    L2 norm of a client update, normalized by sqrt(d).
    delta_w = {layer_name: (local_weights - global_weights)}
    """
    flat = _flatten(delta_w)
    d = flat.size
    return float(np.linalg.norm(flat) / math.sqrt(d))


def compute_cosine_disagreement(delta_w_k: dict, mean_delta_w: dict) -> float:
    """
    1 - cosine_similarity(delta_w_k, mean_delta_w).
    Returns 0 if either vector is zero.
    """
    flat_k    = _flatten(delta_w_k)
    flat_mean = _flatten(mean_delta_w)
    nk   = np.linalg.norm(flat_k)
    nm   = np.linalg.norm(flat_mean)
    if nk < 1e-12 or nm < 1e-12:
        return 0.0
    cos = float(np.dot(flat_k, flat_mean) / (nk * nm))
    return 1.0 - float(np.clip(cos, -1.0, 1.0))


def _compute_drift_metrics(
    client_deltas: list[dict],
) -> tuple[list[float], list[float]]:
    """
    Given a list of per-client delta dicts, compute norms and cosine disagreements.
    Returns (norms, cosine_disagreements) — both lists of length n_clients.
    """
    # Mean delta across clients
    flat_deltas = [_flatten(d) for d in client_deltas]
    mean_flat   = np.mean(flat_deltas, axis=0)

    # Reconstruct mean as a single-key dict for reuse of helper
    mean_dict = {"_": mean_flat}

    norms   = [compute_update_norm(d) for d in client_deltas]
    cosines = [
        compute_cosine_disagreement(d, mean_dict)
        for d in client_deltas
    ]
    return norms, cosines


# ── Batch writer: write all seeds at once (optional helper) ──────────────────

def create_all_writers(root: Path = RESULTS_ROOT) -> dict:
    """
    Pre-create MetricsWriter objects for all 54 experiment configurations
    (3 seeds × 3 regimes × 3 methods × 2 PEFT).

    Returns a dict keyed by (seed, regime, method, peft) tuples.

    Usage:
        writers = create_all_writers()
        w = writers[(1, "mild", "fedavg", "lora")]
        w.log_round(...)
        w.save()
    """
    from itertools import product
    writers = {}
    for seed, regime, method, peft in product(
        [1, 2, 3],
        ["mild", "medium", "hard"],
        ["fedavg", "fedprox", "scaffold"],
        ["lora", "softprompt"],
    ):
        writers[(seed, regime, method, peft)] = MetricsWriter(
            seed, regime, method, peft, root=root
        )
    print(f"✅ Created {len(writers)} MetricsWriter objects.")
    return writers


# ── Quick smoke test (run this file directly to verify) ──────────────────────

if __name__ == "__main__":
    import tempfile, shutil

    print("=" * 60)
    print("metrics_writer.py — smoke test")
    print("=" * 60)

    tmp = Path(tempfile.mkdtemp()) / "results"

    # Test 1: basic API
    logger = MetricsWriter(seed=1, regime="mild", method="fedavg", peft="lora", root=tmp)
    for t in range(1, 6):
        logger.log_round(
            round_num       = t,
            rouge_l         = 0.10 + t * 0.03,
            update_norms    = [0.05, 0.04, 0.06],
            cosine_disagree = [0.20, 0.18, 0.22],
        )
    path = logger.save()

    with open(path) as f:
        m = json.load(f)
    assert m["rounds"] == [1, 2, 3, 4, 5]
    assert len(m["rouge_l_per_round"]) == 5
    assert isinstance(m["update_norms_per_round"][0], list)
    print("Test 1 passed: basic log_round + save")

    # Test 2: state dict helper
    import numpy as np
    rng = np.random.default_rng(0)
    deltas = [
        {"lora_A": rng.normal(0, 0.01, (8, 64)), "lora_B": rng.normal(0, 0.01, (64, 8))}
        for _ in range(4)
    ]
    logger2 = MetricsWriter(seed=2, regime="hard", method="scaffold", peft="lora", root=tmp)
    logger2.log_round_from_state_dicts(round_num=1, rouge_l=0.22,
                                       client_deltas=deltas, global_weights={})
    logger2.save()
    print("✅ Test 2 passed: log_round_from_state_dicts")

    # Test 3: validation
    try:
        MetricsWriter(seed=99, regime="mild", method="fedavg", peft="lora", root=tmp)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"Test 3 passed: validation caught bad seed — {e}")

    shutil.rmtree(tmp.parent)
    print("\nAll smoke tests passed.")