"""Synthetic match generator: mostly legit players plus injected 'abnormal' ones.

Each row is one engagement (a fight). Ground-truth labels are returned separately
so detectors never see them.
"""
import numpy as np
import pandas as pd

BRACKETS = {0: "bronze", 1: "silver", 2: "gold", 3: "pro"}

# per-bracket legit behaviour: (hit_prob, headshot_prob, reaction_median_ms, linearity_mean, omega_mean)
LEGIT = {
    0: dict(hit=0.30, hs=0.12, react=270, lin=0.55, omega=260),
    1: dict(hit=0.38, hs=0.17, react=250, lin=0.60, omega=300),
    2: dict(hit=0.46, hs=0.24, react=235, lin=0.65, omega=350),
    3: dict(hit=0.58, hs=0.38, react=200, lin=0.72, omega=420),  # skilled humans, must NOT be flagged
}

CHEAT_TYPES = ["aimbot", "triggerbot", "wallhack", "mixed"]


def _engagements(rng, p, n, cheat=None, strength=1.0):
    shots = rng.integers(4, 14, n)
    hits = rng.binomial(shots, p["hit"])
    hs_p, react_med, lin, omega = p["hs"], p["react"], p["lin"], p["omega"]
    react_sigma, preaim_mu = 0.20, 32.0  # degrees of crosshair error before LOS opens

    if cheat in ("aimbot", "mixed"):
        hs_p = hs_p + (0.75 - hs_p) * strength
        lin = lin + (0.98 - lin) * strength
        omega = omega * (1 + 1.5 * strength)
        hits = rng.binomial(shots, min(0.95, p["hit"] + 0.35 * strength))
    if cheat in ("triggerbot", "mixed"):
        react_med = react_med + (95 - react_med) * strength
        react_sigma = 0.20 - 0.17 * strength
    if cheat in ("wallhack", "mixed"):
        preaim_mu = preaim_mu - 26 * strength

    headshots = rng.binomial(hits, np.clip(hs_p, 0, 1))
    react = rng.lognormal(np.log(react_med), react_sigma, n)
    linearity = np.clip(rng.normal(lin, 0.07, n), 0, 1)
    omega_peak = np.abs(rng.normal(omega, omega * 0.2, n))
    preaim = np.abs(rng.normal(preaim_mu, 14, n))
    null_preaim = np.abs(rng.normal(32, 14, n))  # crosshair error vs a random enemy spot
    return pd.DataFrame(dict(shots=shots, hits=hits, headshots=headshots, react_ms=react,
                             linearity=linearity, omega_peak=omega_peak,
                             preaim_err=preaim, null_preaim_err=null_preaim))


def _pos_int(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")


def simulate(n_players=400, n_matches=6, n_eng=40, cheater_rate=0.06, seed=42, strength_range=(0.5, 1.0)):
    for name, v in (("n_players", n_players), ("n_matches", n_matches), ("n_eng", n_eng)):
        _pos_int(name, v)
    if not 0 <= cheater_rate <= 1:
        raise ValueError(f"cheater_rate must be between 0 and 1, got {cheater_rate!r}")
    lo, hi = strength_range
    if not 0 <= lo <= hi <= 1:
        raise ValueError(f"strength_range must satisfy 0 <= low <= high <= 1, got {strength_range!r}")
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for pid in range(n_players):
        bracket = int(rng.choice(list(BRACKETS), p=[0.35, 0.3, 0.25, 0.10]))
        is_cheater = rng.random() < cheater_rate
        ctype = str(rng.choice(CHEAT_TYPES)) if is_cheater else None
        strength = float(rng.uniform(*strength_range)) if is_cheater else 0.0
        # each legit player has their own skill wobble
        skill = rng.normal(1.0, 0.06)
        p = {k: (v * skill if k in ("hit", "hs") else v) for k, v in LEGIT[bracket].items()}
        for m in range(n_matches):
            df = _engagements(rng, p, n_eng, ctype, strength)
            df["player_id"], df["match_idx"], df["bracket"] = f"p{pid}", m, bracket
            rows.append(df)
        labels.append(dict(player_id=f"p{pid}", bracket=bracket, is_cheater=is_cheater,
                           cheat_type=ctype, strength=strength))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(labels)
