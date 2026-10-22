"""Known-cheat signature matching (hashes only; never store or ship cheat files).

Client reports look like:
    {"player_id": {"hashes": ["<sha256>", ...], "unknown_modules": ["name.dll", ...]}}
- hashes: SHA-256 of running executables reported by the client. Client-reported, so SOFT until verified.
- unknown_modules: files mapped into the game process that are not on your allow-list.
  Soft signal only.
"""
import hashlib


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_blocklist(path):
    """One SHA-256 per line; '#' starts a comment (use it to record the sample's source)."""
    out = set()
    with open(path) as f:
        for line in f:
            h = line.split("#")[0].strip().lower()
            if h:
                out.add(h)
    return out


def blocklist_matches(reports, blocklist):
    """Players whose REPORTED hashes appear on the blocklist.
    TRUST: a report is only a client's claim; a modified client can forge or omit it. Treat matches as SOFT
    evidence until a moderator verifies them (Moderation.verify_hash_evidence); only verified matches or
    server-observed violations are hard evidence."""
    return {pid for pid, r in reports.items()
            if blocklist & {h.lower() for h in r.get("hashes", [])}}


def unknown_module_players(reports):
    return {pid for pid, r in reports.items() if r.get("unknown_modules")}
