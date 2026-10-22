"""Scan this computer for likely cheat tools (read-only; nothing deleted or uploaded).
    python scan_my_pc.py --game pubg --blocklist blocklist.txt
Run it while the game is open. Use --check-signatures on Windows to also check Authenticode.
"""
import argparse, json
from aimguard.localscan import scan
from aimguard.signatures import load_blocklist

ap = argparse.ArgumentParser()
ap.add_argument("--game", help="part of the game's process name, e.g. pubg (to inspect modules loaded into it)")
ap.add_argument("--blocklist", help="file of known-cheat SHA-256 hashes")
ap.add_argument("--check-signatures", action="store_true")
ap.add_argument("--out", default="scan_report.json")
a = ap.parse_args()

bl = load_blocklist(a.blocklist) if a.blocklist else set()
res = scan(bl, a.game, a.check_signatures)
json.dump(res, open(a.out, "w"), indent=2)
if not res:
    print("No findings. (This does not prove the machine is clean; hidden cheats can evade user-mode scans.)")
for f in res:
    print(f"[{f['level']:6}] {f['kind']:14} {f['name']}\n         {f['path']}\n         -> {f['why']}")
print(f"\n{len(res)} finding(s) saved to {a.out}. Findings are leads to check, not proof. Nothing was changed.")
