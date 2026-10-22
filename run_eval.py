"""Held-out evaluation: separate reference data, several seeds, all-legit and weak-cheat scenarios."""
import numpy as np, pandas as pd
from aimguard.pipeline import build_baseline, run, evaluate

base = build_baseline()
SCEN = {"all legit (0% cheaters)": dict(cheater_rate=0.0),
        "weak cheats  (strength 0.15-0.3)": dict(cheater_rate=0.08, strength_range=(0.15, 0.3)),
        "medium cheats(strength 0.3-0.5)": dict(cheater_rate=0.08, strength_range=(0.3, 0.5)),
        "strong cheats(strength 0.6-1.0)": dict(cheater_rate=0.08, strength_range=(0.6, 1.0))}
rows = []
for name, kw in SCEN.items():
    for seed in range(1, 6):
        _, _, _, risk = run(base, n_players=400, seed=seed, **kw)
        m = evaluate(risk); m["scenario"], m["seed"] = name, seed
        rows.append(m)
df = pd.DataFrame(rows)
print(df.groupby("scenario").agg(cheaters=("cheaters", "sum"), tp=("true_pos", "sum"), fp=("false_pos", "sum"),
      pro_fp=("pro_false_pos", "sum"), recall=("recall", "mean"), legit_fp_rate=("legit_fp_rate", "mean")).round(3).to_string())
