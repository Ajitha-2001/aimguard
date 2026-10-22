import html
import pandas as pd

def write_html(risk: pd.DataFrame, metrics: dict, path="aimguard_report.html", top=40):
    rows = ""
    for _, r in risk.head(top).iterrows():
        rows += (f"<tr><td>{r.player_id}</td><td>{r.risk_score:.3f}</td><td>{r.action}</td>"
                 f"<td>{r.p_headshot:.1e}</td><td>{r.p_preaim:.1e}</td>"
                 f"<td>{html.escape(str(r.fired))}</td></tr>")
    m = " | ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
    page = f"""<!doctype html><meta charset=utf-8><title>AimGuard report</title>
<style>body{{font:14px system-ui;margin:2rem;max-width:1000px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #8884;padding:6px 8px;text-align:left}}</style>
<h1>AimGuard: review queue</h1><p><b>Risk score is an experimental, uncalibrated ranking, not a probability of cheating.</b></p><p>{m}</p>
<table><tr><th>Player</th><th>Risk score (experimental)</th><th>Action</th><th>HS p-value</th><th>Pre-aim p</th><th>Signals fired</th></tr>{rows}</table>"""
    open(path, "w").write(page)
