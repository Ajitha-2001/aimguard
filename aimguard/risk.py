"""EXPERIMENTAL risk score (log-odds style, hand-set weights).

The output is a ranking score in 0..1, NOT a calibrated probability of cheating.
Calibrate against independent labelled data before reading it as a probability.
"""
import math
import pandas as pd

PRIOR_LOGIT = math.log(0.02 / 0.98)
DECAY = 0.98

# weights are placeholders to tune on real, independently labelled data
LLR = {"headshot_z": 1.5, "reaction": 2.0, "linearity": 1.5, "preaim": 2.0, "clean": -0.3}
LLR_PLAYER = {"iforest": 1.5, "p_headshot": 3.0, "p_preaim": 3.0,
              "hash_match": 5.0,        # VERIFIED known cheat-file hash (hard evidence)
              "reported_hash": 3.0,     # client-REPORTED blocklist hash: unverified, can be forged (soft)
              "unknown_modules": 1.0}   # soft signal: never enough on its own; each counted ONCE per window


def match_signals(r) -> dict:
    """Per-match evidence. Each match is genuinely separate data."""
    return {"headshot_z": r.z_hs_rate > 3.5,
            "reaction": r.z_react_med < -3.5 or r.z_react_std < -3.5,  # unusually fast OR unusually consistent, vs bracket
            "linearity": r.z_linearity > 3.5,
            "preaim": r.z_preaim_med < -3.5}


def update_risk(prev_logit, llr):
    return PRIOR_LOGIT + (prev_logit - PRIOR_LOGIT) * DECAY + llr


def to_score(logit):
    return 1 / (1 + math.exp(-logit))


def policy(score, hard_evidence=False):
    """Statistics alone never reach ban eligibility: that needs hard evidence
    (server-proven violation or verified cheat-file match), and still a human."""
    if hard_evidence:   # a verified signature match always gets human eyes
        return "ban_eligible_after_review" if score >= 0.95 else "human_review"
    if score < 0.3:
        return "none"
    if score < 0.7:
        return "monitor"
    return "human_review"


def score_players(mfz, tests, iforest_flags, hard_evidence=None, unknown_modules=None, reported_hash=None) -> pd.DataFrame:
    hard_evidence, unknown_modules, reported_hash = set(hard_evidence or []), set(unknown_modules or []), set(reported_hash or [])
    tests = tests.set_index("player_id")
    out = []
    for pid, d in mfz.sort_values("match_idx").groupby("player_id"):
        logit, fired = PRIOR_LOGIT, {}
        for _, r in d.iterrows():
            sig = match_signals(r)
            logit = update_risk(logit, sum(LLR[k] for k, v in sig.items() if v) or LLR["clean"])
            for k, v in sig.items():
                fired[k] = fired.get(k, 0) + int(v)
        t = tests.loc[pid]
        window = {"iforest": bool(iforest_flags.get(pid, False)),
                  "p_headshot": t.p_headshot < 1e-6, "p_preaim": t.p_preaim < 1e-6,
                  "hash_match": pid in hard_evidence, "reported_hash": pid in reported_hash,
                  "unknown_modules": pid in unknown_modules}
        for k, v in window.items():
            if v:
                logit += LLR_PLAYER[k]
                fired[k] = 1
        out.append(dict(player_id=pid, risk_score=to_score(logit), fired=fired,
                        p_headshot=t.p_headshot, p_preaim=t.p_preaim,
                        hard_evidence=pid in hard_evidence))
    res = pd.DataFrame(out)
    res["action"] = [policy(s, h) for s, h in zip(res.risk_score, res.hard_evidence)]
    return res.sort_values("risk_score", ascending=False).reset_index(drop=True)
