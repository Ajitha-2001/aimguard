"""Score real players.
    python score_real.py --reference clean.csv --live live.csv [--blocklist blocklist.txt --reports reports.json]
CSV columns: player_id, match_idx, bracket, shots, hits, headshots, react_ms, linearity,
             omega_peak, preaim_err, null_preaim_err   (one row per fight; see README)
"""
import argparse, json
import pandas as pd
from aimguard.detectors import Baseline
from aimguard.features import match_features, player_features
from aimguard.risk import score_players
from aimguard.signatures import load_blocklist, blocklist_matches, unknown_module_players
from aimguard.io_utils import read_engagement_csv, normalize_reports, read_id_list

ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True, help="CSV of verified-clean players")
ap.add_argument("--live", required=True, help="CSV of players to score")
ap.add_argument("--blocklist"); ap.add_argument("--reports")
ap.add_argument("--verified-players", help="text file of player IDs whose file-hash match a MODERATOR has verified (hard evidence)")
ap.add_argument("--out", default="review_queue.csv")
a = ap.parse_args()

base = Baseline().fit(read_engagement_csv(a.reference))
live = read_engagement_csv(a.live)
base.validate(live)                      # clear error if window length / brackets don't match

hard, soft, reported = set(), set(), set()
live_ids = set(live.player_id)
if a.reports:
    reports = normalize_reports(json.load(open(a.reports)))
    unknown = sorted(set(reports) - live_ids)
    if unknown:
        print(f"WARNING: {len(unknown)} report ID(s) not in the live CSV (ID mismatch?): {unknown[:5]}")
    soft = unknown_module_players(reports)
    if a.blocklist:   # client-reported hashes are SOFT: a modified client can forge or omit them
        reported = blocklist_matches(reports, load_blocklist(a.blocklist))
if a.verified_players:
    hard = read_id_list(a.verified_players)
    if hard - live_ids:
        print(f"WARNING: {len(hard - live_ids)} verified ID(s) not in the live CSV: {sorted(hard - live_ids)[:5]}")

mf = match_features(live)
pf = player_features(mf)
risk = score_players(base.add_z(mf), base.player_tests(live), base.iforest_flags(pf), hard, soft, reported)
risk.drop(columns=["fired"]).to_csv(a.out, index=False)
print(risk[["player_id", "risk_score", "action"]].head(20).to_string(index=False))
print(f"\nWrote {a.out}. risk_score is an experimental ranking, not a probability. Humans decide actions.")
