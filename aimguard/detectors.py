"""Statistical + ML detectors, all calibrated on a SEPARATE reference population.

Nothing here is fitted on the players being scored, and no fixed share of players is
ever forced to be 'anomalous'.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .features import FEATURE_COLS, match_features, player_features


def headshot_pvalue(headshots, hits, baseline_rate):
    if hits < 100:  # minimum sample size
        return 1.0
    return binomtest(int(headshots), int(hits), baseline_rate, alternative="greater").pvalue


def preaim_pvalue(actual, null):
    if len(actual) < 50:
        return 1.0
    return mannwhitneyu(actual, null, alternative="less").pvalue


class Baseline:
    """Learns 'normal' from reference matches (ideally verified-clean players).

    Reference players are split: most fit the models/z-baselines, the held-out part
    sets the Isolation Forest alert threshold (a quantile of held-out legit scores).
    """

    MIN_FIT, MIN_CAL = 30, 20   # players per bracket needed to fit / to calibrate the threshold

    def __init__(self, iforest_quantile=0.995, cal_fraction=0.3, seed=42):
        if not 0 < cal_fraction < 1:
            raise ValueError(f"cal_fraction must be strictly between 0 and 1, got {cal_fraction!r}")
        if not 0 < iforest_quantile <= 1:
            raise ValueError(f"iforest_quantile must be in (0, 1], got {iforest_quantile!r}")
        self.q, self.cal_fraction, self.seed = iforest_quantile, cal_fraction, seed

    def _min_bracket_size(self):
        f = self.cal_fraction
        return next(n for n in range(1, 100000)
                    if n - round(f * n) >= self.MIN_FIT and round(f * n) >= self.MIN_CAL)

    def validate(self, eng: pd.DataFrame):
        """Reject data this baseline can't score correctly (window length, unseen brackets)."""
        counts = eng.groupby("player_id").match_idx.nunique()
        bad = counts[counts != self.n_matches]
        if len(bad):
            raise ValueError(f"scoring window mismatch: baseline was built on {self.n_matches} matches per "
                             f"player, but {len(bad)} player(s) have a different count "
                             f"(e.g. {bad.index[0]}: {int(bad.iloc[0])}). Use the last {self.n_matches} "
                             f"matches per player, or build a baseline for that window.")
        unseen = set(eng.bracket.unique()) - set(self.models)
        if unseen:
            raise ValueError(f"bracket(s) {sorted(unseen)} not present in the reference data")

    def fit(self, eng_ref: pd.DataFrame):
        counts = eng_ref.groupby("player_id").match_idx.nunique()
        if counts.nunique() != 1:
            raise ValueError("all reference players must have the same number of matches (the scoring window)")
        self.n_matches = int(counts.iloc[0])
        mf = match_features(eng_ref)
        pf = player_features(mf)
        rng = np.random.default_rng(self.seed)
        cal = set()
        for br, d in pf.groupby("bracket"):          # split fit/calibration WITHIN each bracket
            ids = d.player_id.to_numpy()
            n_cal = int(round(len(ids) * self.cal_fraction))
            if len(ids) - n_cal < self.MIN_FIT or n_cal < self.MIN_CAL:
                raise ValueError(
                    f"bracket {br}: only {len(ids)} reference players. Need >= {self.MIN_FIT} to fit and "
                    f">= {self.MIN_CAL} to calibrate (about {self._min_bracket_size()} per bracket at "
                    f"cal_fraction={self.cal_fraction}). Use a larger reference set.")
            cal.update(rng.choice(ids, n_cal, replace=False))
        fit_mf, fit_pf = mf[~mf.player_id.isin(cal)], pf[~pf.player_id.isin(cal)]
        cal_pf = pf[pf.player_id.isin(cal)]
        fit_eng = eng_ref[~eng_ref.player_id.isin(cal)]

        self.stats = {}
        for b, d in fit_mf.groupby("bracket"):
            for c in FEATURE_COLS:
                med = d[c].median()
                self.stats[(b, c)] = (med, (d[c] - med).abs().median())
        g = fit_eng.groupby("bracket")
        self.hs_base = (g.headshots.sum() / g.hits.sum()).to_dict()

        self.models, self.thresholds = {}, {}
        for b, d in fit_pf.groupby("bracket"):
            sc = RobustScaler().fit(d[FEATURE_COLS])
            m = IsolationForest(n_estimators=300, random_state=self.seed).fit(sc.transform(d[FEATURE_COLS]))
            self.models[b] = (sc, m)
            held_out = cal_pf[cal_pf.bracket == b]
            s = -m.score_samples(sc.transform(held_out[FEATURE_COLS]))
            self.thresholds[b] = float(np.quantile(s, self.q))
        return self

    def add_z(self, mf: pd.DataFrame) -> pd.DataFrame:
        mf = mf.copy()
        for c in FEATURE_COLS:
            med = mf.bracket.map({b: self.stats[(b, c)][0] for b in self.models})
            mad = mf.bracket.map({b: self.stats[(b, c)][1] for b in self.models})
            mf[f"z_{c}"] = 0.6745 * (mf[c] - med) / (mad + 1e-9)
        return mf

    def iforest_flags(self, pf: pd.DataFrame) -> dict:
        """player_id -> True if score exceeds the legit-calibrated threshold."""
        out = {}
        for b, d in pf.groupby("bracket"):
            sc, m = self.models[b]
            s = -m.score_samples(sc.transform(d[FEATURE_COLS]))
            out.update(dict(zip(d.player_id, s > self.thresholds[b])))
        return out

    def player_tests(self, eng: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for pid, d in eng.groupby("player_id"):
            b = int(d.bracket.iloc[0])
            rows.append(dict(player_id=pid,
                             p_headshot=headshot_pvalue(d.headshots.sum(), d.hits.sum(), self.hs_base[b]),
                             p_preaim=preaim_pvalue(d.preaim_err.values, d.null_preaim_err.values)))
        return pd.DataFrame(rows)
