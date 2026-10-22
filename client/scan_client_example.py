"""EXAMPLE of a scoped client-side check (Windows). A real client belongs in native code inside
your game, with player consent disclosed in your terms. Needs: pip install -r requirements-client.txt

Privacy rules this follows: scans only RUNNING executables and modules mapped into YOUR game process,
never documents; the server gets hashes of blocklist matches and allow-list misses, never paths.
"""
import json, hashlib
import psutil


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_running_processes(blocklist):
    hits = set()
    for p in psutil.process_iter(["exe"]):
        exe = p.info.get("exe")
        if not exe:
            continue
        try:
            h = sha256_file(exe)
        except OSError:      # protected or closed: skip
            continue
        if h in blocklist:
            hits.add(h)
    return sorted(hits)


def unknown_modules(game_pid, allowed_names):
    try:
        maps = psutil.Process(game_pid).memory_maps()
    except (psutil.Error, OSError):
        return []
    names = {m.path.rsplit("\\", 1)[-1].lower() for m in maps}
    return sorted(n for n in names if n not in allowed_names)


if __name__ == "__main__":
    from aimguard.signatures import load_blocklist  # run from the project root
    report = {"hashes": scan_running_processes(load_blocklist("blocklist.txt")), "unknown_modules": []}
    print(json.dumps(report, indent=2))
