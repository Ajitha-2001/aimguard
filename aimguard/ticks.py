"""Raw server telemetry -> engagement rows (the columns the detectors consume).

Input tables (all times in server ms; angles in degrees; produced by YOUR game server):
  ticks: match_id, player_id, t_ms, yaw, pitch                    (crosshair, ~30-60 Hz)
  shots: match_id, player_id, t_ms, target, hit(bool), hitbox      (one row per shot fired)
  vis  : match_id, player_id, t_ms, target, target_yaw, target_pitch, decoy_yaw, decoy_pitch
         one row when `target` first becomes visible to `player_id`.
         target_*: direction from the player's eye to the target at that time.
         decoy_*:  direction to a plausible OTHER enemy position (e.g. a different occluded enemy);
                   it forms the null distribution for the pre-aim (wallhack) test.
"""
import numpy as np
import pandas as pd

PRE_MS = 500          # window before first visibility used for pre-aim
REACT_MAX_MS = 2000   # a shot later than this after visibility is not a reaction to it
NO_SHOT_WINDOW_MS = 500
ENG_COLS = ["shots", "hits", "headshots", "react_ms", "linearity", "omega_peak",
            "preaim_err", "null_preaim_err"]


def ang_err(yaw, pitch, tyaw, tpitch):
    dy = (np.asarray(yaw) - tyaw + 180) % 360 - 180
    return np.hypot(dy, np.asarray(pitch) - tpitch)


def _path_metrics(t, yaw, pitch):
    """(linearity, peak angular speed deg/s) of the crosshair path. Linearity 1.0 = perfectly straight."""
    if len(t) < 3:
        return np.nan, np.nan
    y = np.degrees(np.unwrap(np.radians(yaw)))
    dy, dp, dt = np.diff(y), np.diff(pitch), np.diff(t) / 1000.0
    ok = dt > 0
    if not ok.any():
        return np.nan, np.nan
    seg = np.hypot(dy, dp)[ok]
    path = seg.sum()
    lin = np.hypot(y[-1] - y[0], pitch[-1] - pitch[0]) / path if path > 1e-6 else np.nan
    return min(lin, 1.0), float((seg / dt[ok]).max())


def extract_player_match(ticks: pd.DataFrame, shots: pd.DataFrame, vis: pd.DataFrame) -> list:
    """One player's data for one match -> list of engagement dicts."""
    if ticks.empty or vis.empty:
        return []
    ticks = ticks.sort_values("t_ms").drop_duplicates("t_ms")
    t, yaw, pit = (ticks[c].to_numpy(float) for c in ("t_ms", "yaw", "pitch"))
    s_t = shots.t_ms.to_numpy(float) if len(shots) else np.array([])
    s_tg = shots.target.to_numpy() if len(shots) else np.array([])
    s_hit = shots.hit.to_numpy(bool) if len(shots) else np.array([], bool)
    s_head = (shots.hitbox.to_numpy() == "head") if len(shots) else np.array([], bool)
    out = []
    for v in vis.itertuples():
        tv = float(v.t_ms)
        lo, hi = np.searchsorted(t, tv - PRE_MS, "left"), np.searchsorted(t, tv, "right")
        if hi - lo < 2:
            continue
        pre_y, pre_p = yaw[lo:hi], pit[lo:hi]
        preaim = float(ang_err(pre_y, pre_p, v.target_yaw, v.target_pitch).mean())
        null = float(ang_err(pre_y, pre_p, v.decoy_yaw, v.decoy_pitch).mean())
        m = (s_tg == v.target) & (s_t >= tv) & (s_t <= tv + REACT_MAX_MS) if len(s_t) else np.array([], bool)
        n_shots = int(m.sum())
        if n_shots:
            t_first = s_t[m].min()
            react, end = t_first - tv, t_first
        else:
            react, end = np.nan, tv + NO_SHOT_WINDOW_MS
        a, b = np.searchsorted(t, tv, "left"), np.searchsorted(t, end, "right")
        lin, omega = _path_metrics(t[a:b], yaw[a:b], pit[a:b])
        out.append(dict(shots=n_shots, hits=int((m & s_hit).sum()), headshots=int((m & s_hit & s_head).sum()),
                        react_ms=react, linearity=lin, omega_peak=omega,
                        preaim_err=preaim, null_preaim_err=null))
    return out


def extract_matches(ticks, shots, vis, bracket_of) -> pd.DataFrame:
    """Many player-matches -> engagement table: player_id, match_id, match_ts, bracket + ENG_COLS.
    bracket_of: dict player_id -> skill bracket (from your matchmaking/MMR)."""
    tg = {k: g for k, g in ticks.groupby(["match_id", "player_id"])}
    sg = {k: g for k, g in shots.groupby(["match_id", "player_id"])} if len(shots) else {}
    rows = []
    for (mid, pid), vg in vis.groupby(["match_id", "player_id"]):
        tk = tg.get((mid, pid))
        if tk is None:
            continue
        sh = sg.get((mid, pid), shots.iloc[0:0])
        for e in extract_player_match(tk, sh, vg):
            e.update(player_id=pid, match_id=mid, match_ts=float(tk.t_ms.min()), bracket=int(bracket_of[pid]))
            rows.append(e)
    cols = ["player_id", "match_id", "match_ts", "bracket"] + ENG_COLS
    return pd.DataFrame(rows, columns=cols)


def assign_windows(eng: pd.DataFrame, n_matches: int) -> pd.DataFrame:
    """Keep each player's LAST n_matches matches (by match_ts) and number them 0..n-1.
    Players with fewer than n_matches are dropped (they can't be scored against a n-match baseline)."""
    if eng.empty:
        return eng.assign(match_idx=pd.Series(dtype=int))
    order = (eng.groupby(["player_id", "match_id"]).match_ts.min().reset_index()
             .sort_values(["player_id", "match_ts"]))
    order["rank_from_end"] = order.groupby("player_id").cumcount(ascending=False)
    keep = order[order.rank_from_end < n_matches]
    keep = keep[keep.groupby("player_id").match_id.transform("count") == n_matches].copy()
    keep["match_idx"] = keep.groupby("player_id").cumcount()
    return eng.merge(keep[["player_id", "match_id", "match_idx"]], on=["player_id", "match_id"])
