"""End-to-end: raw telemetry over HTTP -> extraction -> scoring -> flags -> moderators -> bans -> appeals."""
import json, urllib.request, urllib.error, hashlib
import pytest
from aimguard.api import make_server, serve_in_thread
from aimguard.detectors import Baseline
from aimguard.moderation import Moderation
from aimguard.store import Store
from aimguard.tick_sim import engagements_from_sim, simulate_ticks
from aimguard.ticks import assign_windows

BAD_FILE_HASH = hashlib.sha256(b"captured cheat sample").hexdigest()


def fake_decoder(token):                       # stands in for Google's decode call
    kind, nonce = token.split(":")
    good = kind == "good"
    return {"requestDetails": {"nonce": nonce, "requestPackageName": "com.example.game"},
            "appIntegrity": {"appRecognitionVerdict": "PLAY_RECOGNIZED" if good else "UNEVALUATED", "certificateSha256Digest": ["abc"]},
            "deviceIntegrity": {"deviceRecognitionVerdict": ["MEETS_DEVICE_INTEGRITY"] if good else []},
            "accountDetails": {"appLicensingVerdict": "LICENSED"}}


class Sys:
    def call(self, method, path, tok=None, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if tok: req.add_header("Authorization", f"Bearer {tok}")
        try:
            with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e: return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def sysm():
    S = Sys()
    ref, _ = engagements_from_sim(130, 0.0, seed=100, brackets=(0,))
    S.base = Baseline().fit(assign_windows(ref, 6))
    S.clock = [1_000_000.0]
    S.store = Store(clock=lambda: S.clock[0])
    S.mod = Moderation(S.store)
    srv = make_server(S.store, S.base, S.mod, blocklist={BAD_FILE_HASH}, decoder=fake_decoder, port=0)
    serve_in_thread(srv); S.srv = srv; S.url = f"http://127.0.0.1:{srv.server_address[1]}"
    S.t = {r: S.store.create_token(r, s) for r, s in [("game_server", "gs1"), ("moderator", "mod1"), ("admin", "admin1")]}
    S.t["mod2"], S.t["mod3"] = S.store.create_token("moderator", "mod2"), S.store.create_token("moderator", "mod3")
    # live telemetry: group per match index across players, exactly like a game server would upload it
    gen, S.labels = simulate_ticks(60, cheater_rate=0.15, seed=7, brackets=(0,))
    S.matches = {}
    for t, s, v in gen:
        k = int(t.match_id.iloc[0].split("_m")[1]); pid = t.player_id.iloc[0]
        S.matches.setdefault(k, {})[pid] = {"bracket": 0, "ticks": t[["t_ms", "yaw", "pitch"]].to_dict("list"),
            "shots": s[["t_ms", "target", "hit", "hitbox"]].to_dict("list"),
            "vis": v[["t_ms", "target", "target_yaw", "target_pitch", "decoy_yaw", "decoy_pitch"]].to_dict("list")}
    for k, players in S.matches.items():
        st, out = S.call("POST", f"/v1/matches/M{k}/telemetry", S.t["game_server"], {"players": players})
        assert st == 200 and out["engagements_added"] > 0
    st, out = S.call("POST", "/v1/admin/score", S.t["admin"], {})
    assert st == 200 and out["players_scored"] >= 55
    S.cheaters = S.labels[S.labels.is_cheater].player_id.tolist()
    S.legit = S.labels[~S.labels.is_cheater].player_id.tolist()
    flagged = {f["player_id"] for f in S.call("GET", "/v1/flags?status=open", S.t["moderator"])[1]["flags"]}
    S.fc = [c for c in S.cheaters if c in flagged]            # cheaters that really are in the review queue
    assert len(S.fc) >= 4
    return S


def test_auth_and_role_separation(sysm):
    S = sysm
    assert S.call("GET", "/v1/flags")[0] == 401
    assert S.call("GET", "/v1/flags", "wrong")[0] == 401
    ptok = S.store.create_token("player", "p1")
    assert S.call("GET", "/v1/flags", ptok)[0] == 403
    assert S.call("GET", "/v1/flags", S.t["game_server"])[0] == 403
    assert S.call("POST", "/v1/reviews", S.t["admin"], {"player_id": "p1", "decision": "cheater"})[0] == 403   # separation of duties
    assert S.call("GET", "/v1/nope", S.t["admin"])[0] == 404
    assert S.call("POST", "/v1/reviews", S.t["moderator"], raw=b"{not json")[0] == 400
    assert S.call("POST", "/v1/reviews", S.t["moderator"], {})[0] == 400

def test_reupload_is_idempotent(sysm):
    S = sysm
    st, out = S.call("POST", "/v1/matches/M0/telemetry", S.t["game_server"], {"players": S.matches[0]})
    assert st == 200 and out["engagements_added"] == 0

def test_rescoring_does_not_change_scores(sysm):
    S = sysm; pid = S.cheaters[0]
    before = S.call("GET", f"/v1/players/{pid}/risk", S.t["moderator"])[1]["risk_score"]
    S.call("POST", "/v1/admin/score", S.t["admin"], {})
    assert S.call("GET", f"/v1/players/{pid}/risk", S.t["moderator"])[1]["risk_score"] == pytest.approx(before)

def test_queue_contains_cheaters_not_legit_players(sysm):
    S = sysm
    flagged = {f["player_id"] for f in S.call("GET", "/v1/flags?status=open", S.t["moderator"])[1]["flags"]}
    assert len(flagged & set(S.cheaters)) >= 0.8 * len(S.cheaters)
    assert len(flagged & set(S.legit)) <= 1

def test_statistics_alone_need_two_distinct_moderators_and_ban_is_delayed(sysm):
    S = sysm; pid = S.fc[0]
    assert S.call("POST", "/v1/reviews", S.t["moderator"], {"player_id": pid, "decision": "cheater"})[0] == 200
    st, out = S.call("POST", "/v1/bans", S.t["moderator"], {"player_id": pid})
    assert st == 400 and "2 distinct" in out["error"]
    assert S.call("POST", "/v1/reviews", S.t["mod2"], {"player_id": pid, "decision": "cheater"})[0] == 200
    st, out = S.call("POST", "/v1/bans", S.t["moderator"], {"player_id": pid}); assert st == 200
    eid = out["enforcement_id"]
    S.call("POST", "/v1/admin/ban_wave", S.t["admin"], {})                                    # too early for this ban
    assert S.store.q("SELECT status FROM enforcement WHERE id=?", (eid,))[0]["status"] == "scheduled"
    ptok = S.store.create_token("player", pid)
    assert S.call("GET", "/v1/me/status", ptok)[1]["notice"] is None                         # not enforced yet
    S.clock[0] += 73 * 3600
    assert S.call("POST", "/v1/admin/ban_wave", S.t["admin"], {})[1]["enforced"] >= 1
    assert S.store.q("SELECT status FROM enforcement WHERE id=?", (eid,))[0]["status"] == "active"   # THIS player's ban
    status = S.call("GET", "/v1/me/status", ptok)[1]
    assert "fair play" in status["notice"] and "risk" not in json.dumps(status).lower()     # no detection details leak
    # appeal: only the owner, not someone else, and not the original reviewers resolving it
    other = S.store.create_token("player", "someone_else")
    assert S.call("POST", "/v1/appeals", other, {"player_id": pid, "enforcement_id": eid, "text": "x"})[0] == 403
    st, out = S.call("POST", "/v1/appeals", ptok, {"enforcement_id": eid, "text": "I did not cheat"}); assert st == 200
    aid = out["appeal_id"]
    st, out = S.call("POST", f"/v1/appeals/{aid}/resolve", S.t["moderator"], {"outcome": "upheld"})
    assert st == 400 and "not involved" in out["error"]
    assert S.call("POST", f"/v1/appeals/{aid}/resolve", S.t["mod3"], {"outcome": "overturned"})[0] == 200
    assert S.call("GET", "/v1/me/status", ptok)[1]["notice"] is None                         # ban reverted
    labels = S.mod.export_labels(); assert labels[labels.player_id == pid].label.iloc[0] == "legit"   # feedback label
    # the overturned case is CLOSED: re-scoring must not re-flag it from the same old matches, and old reviews can't re-ban
    S.call("POST", "/v1/admin/score", S.t["admin"], {})
    assert not [f for f in S.call("GET", "/v1/flags?status=open", S.t["moderator"])[1]["flags"] if f["player_id"] == pid]
    S.call("POST", "/v1/reviews", S.t["moderator"], {"player_id": pid, "decision": "cheater"})
    S.call("POST", "/v1/reviews", S.t["mod2"], {"player_id": pid, "decision": "cheater"})
    st, out = S.call("POST", "/v1/bans", S.t["moderator"], {"player_id": pid})
    assert st == 400 and "no open case" in out["error"]
    acts = {r["action"] for r in S.store.q("SELECT action FROM audit_log")}
    assert {"review", "ban_requested", "ban_enforced", "appeal_filed", "appeal_resolved"} <= acts

def test_client_reported_hash_is_soft_until_a_moderator_verifies_it(sysm):
    S = sysm; pid = S.fc[1]
    assert S.call("POST", "/v1/evidence", S.t["game_server"], {"player_id": pid, "kind": "file_hash", "hashes": ["00" * 32]})[1]["recorded"] is False
    out = S.call("POST", "/v1/evidence", S.t["game_server"], {"player_id": pid, "kind": "file_hash", "hashes": [BAD_FILE_HASH.upper()]})[1]
    assert out["recorded"] is True and out["status"] == "reported_unverified"
    assert pid not in S.store.hard_evidence_players()                                     # a client's claim is not hard evidence
    S.call("POST", "/v1/reviews", S.t["moderator"], {"player_id": pid, "decision": "cheater"})
    st, err = S.call("POST", "/v1/bans", S.t["moderator"], {"player_id": pid})
    assert st == 400 and "2 distinct" in err["error"]                                     # still needs two confirmations
    assert S.call("POST", "/v1/evidence/verify", S.t["game_server"], {"player_id": pid, "evidence_id": out["evidence_id"]})[0] == 403
    assert S.call("POST", "/v1/evidence/verify", S.t["moderator"], {"player_id": pid, "evidence_id": out["evidence_id"]})[0] == 200
    assert S.call("POST", "/v1/bans", S.t["moderator"], {"player_id": pid})[0] == 200     # verified: one confirmation suffices

def test_disagreeing_moderators_block_a_ban(sysm):
    S = sysm; pid = S.fc[2]
    S.call("POST", "/v1/reviews", S.t["moderator"], {"player_id": pid, "decision": "cheater"})
    S.call("POST", "/v1/reviews", S.t["mod2"], {"player_id": pid, "decision": "legit"})
    st, out = S.call("POST", "/v1/bans", S.t["moderator"], {"player_id": pid})
    assert st == 400 and "disagree" in out["error"]

def test_reports_are_rate_limited_and_only_add_context(sysm):
    S = sysm; rt = S.store.create_token("player", "reporter1"); target = S.legit[0]
    codes = [S.call("POST", "/v1/reports", rt, {"target_id": target, "reason": "sus"})[0] for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]
    assert S.store.q("SELECT action FROM players WHERE player_id=?", (target,))[0]["action"] != "ban"   # mass reports never auto-punish

def test_attestation_nonce_binding_and_soft_signal(sysm):
    S = sysm; pid = S.legit[1]
    n = S.call("POST", "/v1/attestation/nonce", S.t["game_server"], {"player_id": pid})[1]["nonce"]
    assert S.call("POST", "/v1/attestation/verify", S.t["game_server"], {"player_id": pid, "nonce": n, "token": f"good:{n}"})[1]["ok"] is True
    assert S.call("POST", "/v1/attestation/verify", S.t["game_server"], {"player_id": pid, "nonce": n, "token": f"good:{n}"})[0] == 400   # replay
    n2 = S.call("POST", "/v1/attestation/nonce", S.t["game_server"], {"player_id": pid})[1]["nonce"]
    out = S.call("POST", "/v1/attestation/verify", S.t["game_server"], {"player_id": pid, "nonce": n2, "token": f"bad:{n2}"})[1]
    assert out["ok"] is False and "device_integrity_failed" in out["reasons"]
    assert pid in S.store.soft_signal_players() and pid not in S.store.hard_evidence_players()   # soft, never hard

def test_retention_purge_removes_old_raw_data_and_is_isolated():
    import pandas as pd
    from aimguard.store import Store, ENG_DB_COLS
    st = Store()                                   # own database: must not disturb the shared system under test
    row = lambda mid, ts: dict(player_id="p", match_id=mid, match_ts=ts, bracket=0, shots=1, hits=1, headshots=0,
                               react_ms=200.0, linearity=0.6, omega_peak=300.0, preaim_err=30.0, null_preaim_err=30.0)
    st.add_engagements(pd.DataFrame([row("old", 1.0), row("new", 9e9)])[ENG_DB_COLS])
    assert st.purge_before(1e6) == 1
    assert st.load_engagements().match_id.tolist() == ["new"]
