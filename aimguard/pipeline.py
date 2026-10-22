import pandas as pd
from .simulator import simulate
from .features import match_features, player_features, FEATURE_COLS
from .detectors import Baseline
from .risk import score_players

REVIEW_THRESHOLD = 0.7


def build_baseline(n_ref=1500, n_matches=6, seed=1000) -> Baseline:
    """Reference population is simulated separately from anything later scored."""
    eng_ref, _ = simulate(n_ref, n_matches=n_matches, cheater_rate=0.0, seed=seed)
    return Baseline().fit(eng_ref)


def run(baseline, n_players=400, n_matches=None, cheater_rate=0.06, strength_range=(0.5, 1.0),
        seed=42, hard_evidence=None):
    n_matches = baseline.n_matches if n_matches is None else n_matches   # 0 / negatives are rejected by simulate()
    eng, labels = simulate(n_players, n_matches, cheater_rate=cheater_rate,
                           strength_range=strength_range, seed=seed)
    pf, risk = score_engagements(baseline, eng, hard_evidence)
    return eng, labels, pf, risk.merge(labels, on="player_id")


def score_engagements(baseline, eng, hard_evidence=None, unknown_modules=None, reported_hash=None):
    """Engagement table (player_id, match_idx, bracket + ENG_COLS) -> (player features, risk table).
    Players whose features can't be computed (too little data) are skipped and listed in risk.attrs."""
    baseline.validate(eng)
    mf = match_features(eng)
    bad = mf.loc[mf[FEATURE_COLS].isna().any(axis=1), "player_id"].unique()
    if len(bad):
        mf, eng = mf[~mf.player_id.isin(bad)], eng[~eng.player_id.isin(bad)]
    pf = player_features(mf)
    risk = score_players(baseline.add_z(mf), baseline.player_tests(eng), baseline.iforest_flags(pf),
                         hard_evidence, unknown_modules, reported_hash)
    risk.attrs["skipped_players"] = list(bad)
    return pf, risk


def evaluate(risk: pd.DataFrame, threshold=REVIEW_THRESHOLD) -> dict:
    f = risk.risk_score >= threshold
    c = risk.is_cheater
    tp, fp, fn = int((f & c).sum()), int((f & ~c).sum()), int((~f & c).sum())
    pro = risk[(risk.bracket == 3) & ~c]
    return dict(players=len(risk), cheaters=int(c.sum()), flagged=int(f.sum()), true_pos=tp,
                false_pos=fp, missed=fn, precision=tp / max(tp + fp, 1) if tp + fp else float("nan"),
                recall=tp / max(tp + fn, 1) if c.any() else float("nan"),
                legit_fp_rate=fp / max(int((~c).sum()), 1),
                pro_false_pos=int((pro.risk_score >= threshold).sum()), pro_legit=len(pro))
