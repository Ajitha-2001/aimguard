"""Turn engagement logs into per player-per match features."""
import numpy as np
import pandas as pd

FEATURE_COLS = ["hs_rate", "accuracy", "react_med", "react_std", "linearity",
                "omega_p99", "preaim_med"]


def match_features(eng: pd.DataFrame) -> pd.DataFrame:
    g = eng.groupby(["player_id", "match_idx", "bracket"])
    out = g.apply(lambda d: pd.Series(dict(
        n_shots=d.shots.sum(), n_hits=d.hits.sum(), n_hs=d.headshots.sum(),
        hs_rate=d.headshots.sum() / max(d.hits.sum(), 1),
        accuracy=d.hits.sum() / max(d.shots.sum(), 1),
        react_med=d.react_ms.median(), react_std=d.react_ms.std(),
        linearity=d.linearity.mean(),
        omega_p99=(np.percentile(d.omega_peak.dropna(), 99) if d.omega_peak.notna().any() else np.nan),
        preaim_med=d.preaim_err.median())), include_groups=False)
    return out.reset_index()


def player_features(mf: pd.DataFrame) -> pd.DataFrame:
    agg = mf.groupby(["player_id", "bracket"]).agg(
        n_hits=("n_hits", "sum"), n_hs=("n_hs", "sum"),
        **{c: (c, "mean") for c in FEATURE_COLS}).reset_index()
    return agg
