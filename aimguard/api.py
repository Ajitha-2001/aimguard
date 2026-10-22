"""HTTP API (stdlib http.server; fine for a prototype / behind a reverse proxy; swap in FastAPI for production).
Auth: 'Authorization: Bearer <token>'. Roles: game_server, moderator, admin, player."""
import json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import pandas as pd

from .attestation import NonceStore, evaluate_play_integrity
from .moderation import Moderation
from .network import RateLimiter
from .service import ingest_telemetry, score_all
from .signatures import blocklist_matches
from .store import REPORTED_HASH

MAX_BODY = 25 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg); self.status = status


def _need(body, *keys):
    miss = [k for k in keys if k not in body]
    if miss:
        raise ApiError(400, f"missing field(s): {miss}")


def _frame(d, cols, name):
    try:
        df = pd.DataFrame(d)
    except ValueError as e:
        raise ApiError(400, f"{name}: columns must be equal-length lists") from e
    if df.empty:
        return df.reindex(columns=cols)
    if not set(cols) <= set(df.columns):
        raise ApiError(400, f"{name} needs columns {cols}")
    return df


def make_server(store, baseline, moderation=None, blocklist=frozenset(), decoder=None, package="com.example.game",
                cert_digests=None, host="127.0.0.1", port=8080, report_rate=(5 / 60, 5)):
    mod = moderation or Moderation(store)
    nonces, limiters = NonceStore(), {}
    MODS, VIEW = {"moderator"}, {"moderator", "admin"}

    def telemetry(ctx):
        _need(ctx["body"], "players")
        ticks, shots, vis, bracket_of = [], [], [], {}
        for pid, d in ctx["body"]["players"].items():
            _need(d, "bracket", "ticks", "vis")
            bracket_of[pid] = int(d["bracket"])
            for lst, key, cols in ((ticks, "ticks", ["t_ms", "yaw", "pitch"]),
                                   (vis, "vis", ["t_ms", "target", "target_yaw", "target_pitch", "decoy_yaw", "decoy_pitch"]),
                                   (shots, "shots", ["t_ms", "target", "hit", "hitbox"])):
                df = _frame(d.get(key, {c: [] for c in cols}), cols, key)
                df["match_id"], df["player_id"] = ctx["m"][0], pid
                lst.append(df)
        n = ingest_telemetry(store, pd.concat(ticks), pd.concat(shots), pd.concat(vis), bracket_of)
        return 200, {"engagements_added": n}

    def risk(ctx):
        pid = ctx["m"][0]
        r = store.q("SELECT * FROM players WHERE player_id=?", (pid,))
        if not r:
            raise ApiError(404, "unknown player")
        r = r[0]
        return 200, {"player_id": pid, "risk_score": r["risk_score"], "action": r["action"], "signals": json.loads(r["fired"]),
                     "evidence": store.q("SELECT kind, at FROM evidence WHERE player_id=?", (pid,)),
                     "weighted_reports": store.weighted_report_score(pid),
                     "note": "risk_score is an experimental, uncalibrated ranking, not a probability"}

    def flags(ctx):
        q = ctx["query"]
        return 200, {"flags": mod.queue(q.get("status", ["open"])[0], float(q.get("min_severity", ["0"])[0]))}

    def review(ctx):
        b = ctx["body"]; _need(b, "player_id", "decision")
        return 200, {"review_id": mod.submit_review(ctx["subject"], b["player_id"], b["decision"], b.get("notes", ""), b.get("flag_id"))}

    def report(ctx):
        b = ctx["body"]; _need(b, "target_id")
        lim = limiters.setdefault(ctx["subject"], RateLimiter(*report_rate))
        if not lim.allow():
            raise ApiError(429, "too many reports; slow down")
        store.add_report(ctx["subject"], b["target_id"], str(b.get("reason", ""))[:500])
        return 200, {"ok": True}      # reports only add weighted context to the review queue; they never trigger action

    def nonce(ctx):
        _need(ctx["body"], "player_id")
        return 200, {"nonce": nonces.issue(ctx["body"]["player_id"])}

    def attest(ctx):
        b = ctx["body"]; _need(b, "player_id", "nonce", "token")
        if decoder is None:
            raise ApiError(501, "no Play Integrity decoder configured (needs Google credentials)")
        if not nonces.consume(b["player_id"], b["nonce"]):
            raise ApiError(400, "invalid, expired or reused nonce")
        ok, reasons = evaluate_play_integrity(decoder(b["token"]), b["nonce"], package, cert_digests)
        if not ok:
            store.add_evidence(b["player_id"], "attestation_failed", ",".join(sorted(reasons)))   # soft signal only
        return 200, {"ok": ok, "reasons": reasons}

    def evidence(ctx):
        b = ctx["body"]; _need(b, "player_id", "kind")
        pid = b["player_id"]
        if b["kind"] == "file_hash":      # a client-REPORTED hash: soft until a moderator verifies it
            if not blocklist_matches({pid: {"hashes": b.get("hashes", [])}}, blocklist):
                return 200, {"recorded": False}
            ref = ",".join(sorted(h.lower() for h in b["hashes"]))
            eid = store.add_evidence(pid, REPORTED_HASH, ref)
            return 200, {"recorded": eid is not None, "evidence_id": eid, "status": "reported_unverified"}
        if b["kind"] == "server_violation":   # observed by the game server itself
            eid = store.add_evidence(pid, "server_violation", str(b.get("ref", "")))
            return 200, {"recorded": eid is not None, "evidence_id": eid}
        if b["kind"] == "unknown_modules":
            eid = store.add_evidence(pid, "unknown_modules", str(b.get("ref", "")))
            return 200, {"recorded": eid is not None}
        raise ApiError(400, "unknown evidence kind")

    def verify_evidence(ctx):
        b = ctx["body"]; _need(b, "player_id", "evidence_id")
        return 200, {"verified_id": mod.verify_hash_evidence(ctx["subject"], b["player_id"], int(b["evidence_id"]))}

    def ban(ctx):
        _need(ctx["body"], "player_id")
        return 200, {"enforcement_id": mod.request_ban(ctx["subject"], ctx["body"]["player_id"])}

    def appeal(ctx):
        b = ctx["body"]; _need(b, "enforcement_id", "text")
        return 200, {"appeal_id": mod.file_appeal(ctx["subject"], int(b["enforcement_id"]), b["text"])}

    def resolve(ctx):
        b = ctx["body"]; _need(b, "outcome")
        mod.resolve_appeal(ctx["subject"], int(ctx["m"][0]), b["outcome"], b.get("notes", ""))
        return 200, {"ok": True}

    def status(ctx):
        n = mod.player_notice(ctx["subject"])
        return 200, {"notice": n, "enforcements": store.q("SELECT id, status, appealable_until FROM enforcement WHERE player_id=?", (ctx["subject"],)) if n else []}

    routes = [
        ("POST", r"/v1/matches/([^/]+)/telemetry", {"game_server"}, telemetry),
        ("GET", r"/v1/players/([^/]+)/risk", VIEW, risk),
        ("GET", r"/v1/flags", VIEW, flags),
        ("POST", r"/v1/reviews", MODS, review),
        ("POST", r"/v1/reports", {"player", "game_server"}, report),
        ("POST", r"/v1/attestation/nonce", {"game_server"}, nonce),
        ("POST", r"/v1/attestation/verify", {"game_server"}, attest),
        ("POST", r"/v1/evidence", {"game_server"}, evidence),
        ("POST", r"/v1/evidence/verify", MODS, verify_evidence),
        ("POST", r"/v1/bans", MODS, ban),
        ("POST", r"/v1/appeals", {"player"}, appeal),
        ("POST", r"/v1/appeals/(\d+)/resolve", MODS, resolve),
        ("GET", r"/v1/me/status", {"player"}, status),
        ("POST", r"/v1/admin/score", {"admin"}, lambda c: (200, {"players_scored": score_all(store, baseline, mod)})),
        ("POST", r"/v1/admin/ban_wave", {"admin"}, lambda c: (200, mod.run_ban_wave())),
    ]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            data = json.dumps(obj, default=str).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def _handle(self, method):
            try:
                u = urlparse(self.path)
                for m, pat, roles, fn in routes:
                    mt = re.fullmatch(pat, u.path)
                    if m != method or not mt:
                        continue
                    h = self.headers.get("Authorization", "")
                    who = store.auth(h[7:]) if h.startswith("Bearer ") else None
                    if not who:
                        raise ApiError(401, "missing or invalid token")
                    if who[0] not in roles:
                        raise ApiError(403, "this role may not use this endpoint")
                    body = {}
                    if method == "POST":
                        n = int(self.headers.get("Content-Length", 0))
                        if n > MAX_BODY:
                            raise ApiError(413, "body too large")
                        try:
                            body = json.loads(self.rfile.read(n) or b"{}")
                        except json.JSONDecodeError as e:
                            raise ApiError(400, "invalid JSON") from e
                        if not isinstance(body, dict):
                            raise ApiError(400, "JSON body must be an object")
                    if who[0] == "player" and body.get("player_id", who[1]) != who[1]:
                        raise ApiError(403, "players may only act on their own account")
                    status, obj = fn({"m": mt.groups(), "query": parse_qs(u.query), "body": body, "subject": who[1], "role": who[0]})
                    return self._send(status, obj)
                raise ApiError(404, "no such endpoint")
            except ApiError as e:
                self._send(e.status, {"error": str(e)})
            except ValueError as e:                      # workflow rule violations
                self._send(400, {"error": str(e)})
            except Exception:
                self._send(500, {"error": "internal error"})   # never leak internals

        def do_GET(self): self._handle("GET")
        def do_POST(self): self._handle("POST")

    srv = ThreadingHTTPServer((host, port), H)
    srv.moderation, srv.nonces = mod, nonces
    return srv


def serve_in_thread(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start(); return t
