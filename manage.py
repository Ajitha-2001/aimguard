"""AimGuard management CLI.
  python manage.py demo                                   # synthetic end-to-end run (creates demo.db)
  python manage.py train-baseline --reference clean_engagements.csv --out baseline.pkl
  python manage.py make-token --db aimguard.db --role moderator --subject alice
  python manage.py serve --db aimguard.db --baseline baseline.pkl --blocklist blocklist.txt --port 8080
  python manage.py score --db aimguard.db --baseline baseline.pkl
  python manage.py ban-wave --db aimguard.db
Roles: game_server, moderator, admin, player.  Tokens are shown ONCE; only a hash is stored.
Only load baseline.pkl files you created yourself (pickle can run code)."""
import argparse, pickle, sys
import pandas as pd
from aimguard.api import make_server
from aimguard.detectors import Baseline
from aimguard.moderation import Moderation
from aimguard.service import score_all, ingest_telemetry
from aimguard.signatures import load_blocklist
from aimguard.store import Store
from aimguard.ticks import assign_windows
from aimguard.io_utils import read_engagement_csv


def load_baseline(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    tb = sub.add_parser("train-baseline"); tb.add_argument("--reference", required=True); tb.add_argument("--out", default="baseline.pkl"); tb.add_argument("--n-matches", type=int, default=6)
    mt = sub.add_parser("make-token"); mt.add_argument("--db", default="aimguard.db"); mt.add_argument("--role", required=True, choices=["game_server", "moderator", "admin", "player"]); mt.add_argument("--subject", required=True)
    sv = sub.add_parser("serve"); sv.add_argument("--db", default="aimguard.db"); sv.add_argument("--baseline", required=True); sv.add_argument("--blocklist"); sv.add_argument("--host", default="127.0.0.1"); sv.add_argument("--port", type=int, default=8080)
    sc = sub.add_parser("score"); sc.add_argument("--db", default="aimguard.db"); sc.add_argument("--baseline", required=True)
    bw = sub.add_parser("ban-wave"); bw.add_argument("--db", default="aimguard.db")
    dm = sub.add_parser("demo"); dm.add_argument("--db", default="demo.db")
    a = ap.parse_args()

    if a.cmd == "train-baseline":
        ref = assign_windows(read_engagement_csv(a.reference), a.n_matches)
        base = Baseline().fit(ref)
        with open(a.out, "wb") as f:
            pickle.dump(base, f)
        print(f"baseline trained on {ref.player_id.nunique()} players, window={base.n_matches} matches -> {a.out}")
    elif a.cmd == "make-token":
        print(Store(a.db).create_token(a.role, a.subject))
    elif a.cmd == "serve":
        srv = make_server(Store(a.db), load_baseline(a.baseline), blocklist=load_blocklist(a.blocklist) if a.blocklist else set(),
                          host=a.host, port=a.port)
        print(f"AimGuard API on http://{a.host}:{a.port}  (Ctrl+C to stop)")
        try: srv.serve_forever()
        except KeyboardInterrupt: pass
    elif a.cmd == "score":
        s = Store(a.db); print("players scored:", score_all(s, load_baseline(a.baseline), Moderation(s)))
    elif a.cmd == "ban-wave":
        print(Moderation(Store(a.db)).run_ban_wave())
    elif a.cmd == "demo":
        from aimguard.tick_sim import engagements_from_sim, simulate_ticks
        from aimguard.ticks import extract_matches
        print("1/4 training baseline on simulated clean players (about 15 s)...")
        ref, _ = engagements_from_sim(130, 0.0, seed=100, brackets=(0,))
        base = Baseline().fit(assign_windows(ref, 6))
        print("2/4 ingesting raw tick telemetry for 60 players (15% simulated cheaters)...")
        s = Store(a.db); mod = Moderation(s)
        gen, labels = simulate_ticks(60, cheater_rate=0.15, seed=7, brackets=(0,))
        bo = dict(zip(labels.player_id, labels.bracket))
        for t, sh, v in gen:
            ingest_telemetry(s, t, sh, v, bo)
        print("3/4 scoring...", score_all(s, base, mod), "players scored")
        q = mod.queue(); cheat = set(labels[labels.is_cheater].player_id)
        print(f"4/4 review queue: {len(q)} flagged; {sum(f['player_id'] in cheat for f in q)} of {len(cheat)} simulated cheaters caught,"
              f" {sum(f['player_id'] not in cheat for f in q)} legit players flagged")
        for f in q[:8]:
            print(f"   {f['player_id']:>5}  risk={f['severity']:.3f}  simulated cheater={f['player_id'] in cheat}")
        print(f"Database: {a.db}. Next: make tokens and run `serve` (see README).")


if __name__ == "__main__":
    sys.exit(main())
