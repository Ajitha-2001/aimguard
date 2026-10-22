"""Glue: telemetry -> engagements -> scoring -> flags."""
from .ticks import extract_matches, assign_windows
from .pipeline import score_engagements
from .store import REPORTED_HASH


def ingest_telemetry(store, ticks, shots, vis, bracket_of):
    eng = extract_matches(ticks, shots, vis, bracket_of)
    return store.add_engagements(eng) if len(eng) else 0


def score_all(store, baseline, moderation):
    """Idempotent: re-scoring recomputes each player's score from stored data (nothing is double counted)."""
    eng = assign_windows(store.load_engagements(), baseline.n_matches)
    if eng.empty:
        return 0
    _, risk = score_engagements(baseline, eng, store.hard_evidence_players(), store.soft_signal_players(),
                             store.players_with((REPORTED_HASH,)))
    for r in risk.itertuples():
        store.set_risk(r.player_id, r.risk_score, r.action, r.fired)
    moderation.create_flags(risk)
    return len(risk)
