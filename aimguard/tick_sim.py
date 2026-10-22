"""Generate RAW tick telemetry (crosshair path, shots, visibility events) for testing the extractor.
Synthetic: humans curve and hesitate; scripted cheats snap fast and straight; wallhacks pre-aim."""
import numpy as np
import pandas as pd
from .simulator import LEGIT, CHEAT_TYPES

HZ_MS = 1000 / 60
REL = np.arange(-600, 1500, HZ_MS)            # ms relative to first visibility
MOVE_FRAC = {0: 0.55, 1: 0.55, 2: 0.55, 3: 0.55}


def _player_match(rng, p, n_eng, cheat, strength, pid, mid, match_start):
    E, T = n_eng, len(REL)
    aim = cheat in ("aimbot", "mixed"); trig = cheat in ("triggerbot", "mixed"); wall = cheat in ("wallhack", "mixed")
    react_med, sig = p["react"], 0.20
    if trig:
        react_med, sig = react_med + (95 - react_med) * strength, 0.20 - 0.17 * strength
    R = rng.lognormal(np.log(react_med), sig, E)
    m = np.abs(rng.normal(32 - 26 * strength if wall else 32, 14, E))               # pre-aim error (deg)
    phi = rng.uniform(0, 2 * np.pi, E)
    c0y, c0p = rng.uniform(-170, 170, E), rng.uniform(-20, 20, E)
    Ty, Tp = c0y + m * np.cos(phi), c0p + m * np.sin(phi)
    r_start = 0.45 * R
    D = (R - r_start) * ((1 - 0.7 * strength) if aim else 1.0)
    D = np.maximum(D, 20.0)
    r_start = R - D
    amp = np.abs(rng.normal(0.01 if aim else 0.45, 0.005 if aim else 0.15, E))
    noise_sd = 0.02 if aim else 0.8
    u = np.clip((REL[None, :] - r_start[:, None]) / D[:, None], 0, 1)
    s = u * u * (3 - 2 * u)
    dy, dp = (Ty - c0y), (Tp - c0p)
    nrm = np.maximum(np.hypot(dy, dp), 1e-9)
    lat = amp[:, None] * nrm[:, None] * np.sin(np.pi * u)
    mv = ((u > 0) & (u < 1)).astype(float)
    yaw = c0y[:, None] + s * dy[:, None] + (-dp / nrm)[:, None] * lat + rng.normal(0, noise_sd, (E, T)) * mv
    pit = c0p[:, None] + s * dp[:, None] + (dy / nrm)[:, None] * lat + rng.normal(0, noise_sd, (E, T)) * mv
    jit = np.cumsum(rng.normal(0, 0.04, (E, T)), axis=1)
    yaw, pit = ((yaw + jit + 180) % 360) - 180, pit + jit
    t0 = match_start + 10_000 + np.arange(E) * 8_000
    tick_t = (t0[:, None] + REL[None, :])
    ticks = pd.DataFrame(dict(match_id=mid, player_id=pid, t_ms=tick_t.ravel(), yaw=yaw.ravel(), pitch=pit.ravel()))

    hit_p = min(0.95, p["hit"] + (0.35 * strength if aim else 0))
    hs_p = p["hs"] + ((0.75 - p["hs"]) * strength if aim else 0)
    srows = []
    for e in range(E):
        for k in range(int(rng.integers(2, 8))):
            hit = bool(rng.random() < hit_p)
            srows.append((mid, pid, t0[e] + R[e] + 110 * k, f"e{e}", hit, "head" if hit and rng.random() < hs_p else "body" if hit else ""))
    shots = pd.DataFrame(srows, columns=["match_id", "player_id", "t_ms", "target", "hit", "hitbox"])
    dm, dphi = np.abs(rng.normal(32, 14, E)), rng.uniform(0, 2 * np.pi, E)
    vis = pd.DataFrame(dict(match_id=mid, player_id=pid, t_ms=t0, target=[f"e{e}" for e in range(E)],
                            target_yaw=Ty, target_pitch=Tp, decoy_yaw=c0y + dm * np.cos(dphi), decoy_pitch=c0p + dm * np.sin(dphi)))
    return ticks, shots, vis


def simulate_ticks(n_players=40, n_matches=6, n_eng=20, cheater_rate=0.0, seed=0, brackets=(0, 1, 2, 3),
                   strength_range=(0.5, 1.0), start_index=0):
    """Yield (ticks, shots, vis) per player-match, plus a labels table. Streaming keeps memory small."""
    rng = np.random.default_rng(seed)
    labels, plan = [], []
    for i in range(n_players):
        pid = f"p{start_index + i}"
        b = int(rng.choice(brackets))
        cheat = bool(rng.random() < cheater_rate)
        ct = str(rng.choice(CHEAT_TYPES)) if cheat else None
        st = float(rng.uniform(*strength_range)) if cheat else 0.0
        skill = rng.normal(1.0, 0.06)
        p = {k: (v * skill if k in ("hit", "hs") else v) for k, v in LEGIT[b].items()}
        labels.append(dict(player_id=pid, bracket=b, is_cheater=cheat, cheat_type=ct, strength=st))
        plan.append((pid, p, ct, st))
    labels = pd.DataFrame(labels)

    def gen():
        for pid, p, ct, st in plan:
            for k in range(n_matches):
                yield _player_match(rng, p, n_eng, ct, st, pid, f"{pid}_m{k}", k * 3_600_000)
    return gen(), labels


def engagements_from_sim(n_players, cheater_rate, seed, **kw):
    """Simulate raw ticks and run the real extractor; returns (engagement rows, labels)."""
    from .ticks import extract_matches
    gen, labels = simulate_ticks(n_players, cheater_rate=cheater_rate, seed=seed, **kw)
    bracket_of = dict(zip(labels.player_id, labels.bracket))
    parts = [extract_matches(t, s, v, bracket_of) for t, s, v in gen]
    return pd.concat(parts, ignore_index=True), labels
