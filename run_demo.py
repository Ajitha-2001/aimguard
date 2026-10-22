from aimguard.pipeline import build_baseline, run, evaluate
from aimguard.report import write_html

base = build_baseline()
eng, labels, pf, risk = run(base)
m = evaluate(risk)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
print(risk[["player_id", "bracket", "risk_score", "action", "is_cheater", "cheat_type", "strength"]].head(12).to_string(index=False))
risk.drop(columns=["fired"]).to_csv("flags.csv", index=False)
write_html(risk, m)
