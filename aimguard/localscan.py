"""Scan YOUR OWN computer for likely cheat tools. Read-only: nothing is deleted, nothing is uploaded.

Findings are LEADS for you to check, not proof. Legit software (Discord/Steam overlays, OBS, GPU tools)
also loads into games, and a well-hidden cheat can evade any user-mode scan.
"""
import hashlib, os, re, subprocess, sys

KEYWORDS = {"aimbot", "wallhack", "triggerbot", "injector", "esp", "radar", "cheat", "hack",
            "hacks", "spoofer", "hwid", "unlocker", "loader", "bypass", "norecoil"}
DUAL_USE = {"cheatengine", "x64dbg", "x32dbg", "ollydbg", "processhacker", "extremeinjector"}
_WIN = re.compile(r"^[a-z]:\\")
_WIN_TRUSTED = re.compile(r"^[a-z]:\\(windows|program files|program files \(x86\))(\\|$)")
_WIN_RISKY = re.compile(r"^[a-z]:\\(users\\[^\\]+\\(downloads|desktop|appdata\\local\\temp)|windows\\temp|temp|tmp)(\\|$)")
_POSIX_TRUSTED = ("/usr/", "/lib/", "/lib64/", "/opt/", "/bin/", "/sbin/")
_POSIX_RISKY = ("/tmp/", "/var/tmp/", "/dev/shm/")


def norm(path):
    """Windows-style paths are compared case-insensitively with '\\'; POSIX paths are left as they are."""
    p = path or ""
    return p.lower().replace("/", "\\") if (_WIN.match(p.lower().replace("/", "\\")) or "\\" in p) else p


def in_dir(path, directory):
    """True only if `path` is the directory or INSIDE it. Boundary-aware: C:\\Games\\PUBG_cheats is NOT in C:\\Games\\PUBG."""
    n, d = norm(path), norm(directory).rstrip("\\/")
    sep = "\\" if "\\" in d else "/"
    return bool(d) and (n == d or n.startswith(d + sep))


def is_trusted_location(path):
    """Anchored to the drive root: C:\\Windows\\, C:\\Program Files\\ ... never a folder merely NAMED like that deeper down."""
    n = norm(path)
    return bool(_WIN_TRUSTED.match(n)) or n.startswith(_POSIX_TRUSTED)


def is_risky_location(path):
    n = norm(path)
    return bool(_WIN_RISKY.match(n)) or n.startswith(_POSIX_RISKY) or bool(re.match(r"^/home/[^/]+/downloads/", n, re.I))


def name_tokens(path):
    base = re.split(r"[\\/]", path or "")[-1].lower()
    base = re.sub(r"\.(exe|dll|sys)$", "", base)
    return {t for t in re.split(r"[^a-z0-9]+", base) if t}, base.replace(" ", "").replace("_", "").replace("-", "")


def classify_file(path, sha256=None, blocklist=frozenset()):
    """Return list of (level, reason). Levels: MATCH (blocklist hash), REVIEW, INFO."""
    out = []
    if sha256 and sha256.lower() in blocklist:
        out.append(("MATCH", "hash is on your cheat blocklist"))
    tokens, squashed = name_tokens(path)
    if tokens & KEYWORDS:
        out.append(("REVIEW", f"name contains cheat-style word(s): {sorted(tokens & KEYWORDS)}"))
    if any(d in squashed for d in DUAL_USE):
        out.append(("INFO", "dual-use tool (debugger / memory editor / injector), not proof of cheating"))
    if is_risky_location(path):
        out.append(("INFO", "runs from a temp/downloads/desktop folder"))
    return out


MODULE_EXT = re.compile(r"\.(dll|sys|dylib)$|\.so[\d.]*$", re.I)


def module_findings(module_paths, game_dir, blocklist, hash_fn):
    """Every module mapped into the game is HASHED and checked against the blocklist FIRST, wherever it lives
    (a blocklisted DLL inside the game folder or C:\\Windows is still a MATCH). Only afterwards is location used,
    to decide which remaining modules count as 'unknown' (outside the game folder and trusted system folders)."""
    out = []
    for m in sorted({m for m in module_paths if m and MODULE_EXT.search(m)}):
        try:
            h = hash_fn(m)
        except OSError:
            h = None
        unknown = not is_trusted_location(m) and not in_dir(m, game_dir)
        reasons = [r for r in classify_file(m, h, blocklist) if unknown or r[0] == "MATCH"]
        if unknown and not any(lvl in ("MATCH", "REVIEW") for lvl, _ in reasons):
            reasons.append(("REVIEW", "unknown file loaded into the game (outside the game and system folders)"))
        if unknown and h is None:
            reasons.append(("INFO", "file could not be read, so its hash was not checked"))
        for level, why in reasons:
            out.append(dict(level=level, kind="module-in-game", name=re.split(r"[\\/]", m)[-1], path=m, sha256=h, why=why))
    return out


def unknown_game_modules(module_paths, game_dir):
    """Modules outside the game folder and trusted system folders (location only; use module_findings for hashing)."""
    return sorted(m for m in set(module_paths) if m and MODULE_EXT.search(m) and not is_trusted_location(m) and not in_dir(m, game_dir))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signature_status(path):
    """Windows only: Authenticode status via PowerShell ('Valid', 'NotSigned', ...). None if unavailable."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-AuthenticodeSignature -LiteralPath $args[0]).Status", path],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def is_self(exe, own=None):
    """True for this scanner's own Python interpreter (and its venv launcher), so it never reports itself."""
    own = {os.path.normcase(os.path.realpath(x)) for x in (own or [sys.executable, getattr(sys, "_base_executable", "")]) if x}
    return os.path.normcase(os.path.realpath(exe)) in own


def scan(blocklist=frozenset(), game_name=None, check_signatures=False):
    import psutil
    findings, cache, game_pids = [], {}, []
    for p in psutil.process_iter(["pid", "name", "exe"]):
        exe = p.info.get("exe")
        if not exe or is_self(exe):
            continue
        if exe not in cache:
            try:
                cache[exe] = sha256_file(exe)
            except OSError:
                cache[exe] = None
        reasons = classify_file(exe, cache[exe], blocklist)
        if cache[exe] is None and not is_trusted_location(exe):
            reasons.append(("INFO", "executable could not be read, so its hash was not checked"))
        if check_signatures and reasons and cache[exe]:
            st = signature_status(exe)
            if st and st != "Valid":
                reasons.append(("REVIEW", f"executable signature status: {st}"))
        for level, why in reasons:
            findings.append(dict(level=level, kind="process", name=p.info["name"], path=exe,
                                 sha256=cache[exe], why=why))
        if game_name and game_name.lower() in (p.info["name"] or "").lower():
            game_pids.append((p.info["pid"], exe))
    def cached_hash(path):
        if path not in cache:
            try:
                cache[path] = sha256_file(path)
            except OSError:
                cache[path] = None
        if cache[path] is None:
            raise OSError("unreadable")
        return cache[path]

    for pid, exe in game_pids:
        try:
            mods = [m.path for m in psutil.Process(pid).memory_maps()]
        except (psutil.Error, OSError):
            findings.append(dict(level="INFO", kind="module", name=str(pid), path=exe, sha256=None,
                                 why="could not read modules (run PowerShell as Administrator)"))
            continue
        game_dir = re.split(r"[\\/][^\\/]*$", exe)[0]
        findings.extend(module_findings(mods, game_dir, blocklist, cached_hash))
    order = {"MATCH": 0, "REVIEW": 1, "INFO": 2}
    return sorted(findings, key=lambda f: order[f["level"]])
